"""SDFT trainer — Megatron Bridge version with DDP support.

Orchestrates:
    1. Env rollout (rank 0 only, broadcast to all ranks)
       - ENV_TYPE=rag: RagEnv (vLLM generation + optional reflector)
       - ENV_TYPE=api_adapter: ApiAdapterEnv (multi-turn adapter loop)
    2. Student forward pass (Megatron-Core GPTModel via PyTorch DDP)
    3. Teacher log-probs (HTTP from logprob server, each rank independently)
    4. Reverse KL loss + backward (with gradient accumulation + no_sync)
    5. Step-level weight sync to both servers (rank 0 only)

Launch: torchrun --nproc_per_node=N -m megatron_trainer.trainer
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from megatron_trainer.collator import SDFTCollator
from megatron_trainer.config import (
    BATCH_SIZE,
    ENV_TYPE,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    HF_MODEL_PATH,
    HINDSIGHT_FIELD,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_NAME,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SAVE_EVERY,
    STUDENT_MAX_PROMPT_LEN,
    TEACHER_MAX_PROMPT_LEN,
    TRAIN_DATA_PATH,
    VLLM_BASE_URL,
    WANDB_PROJECT,
    WANDB_ENTITY,
    WANDB_NAME,
)
from megatron_trainer.env import ApiAdapterEnv, RagEnv
from megatron_trainer.model_utils import (
    cleanup,
    init_distributed_trainer,
    load_model,
    save_hf_checkpoint,
)
from megatron_trainer.logprob_client import (
    init_logprob_weight_engine,
    request_teacher_log_probs_batch_http,
    request_teacher_log_probs_http,
    sync_weights_to_logprob_server,
    wait_for_logprob_server,
)
from megatron_trainer.vllm_utils import (
    init_vllm_weight_engine,
    sync_weights_to_vllm,
    wait_for_vllm,
)

# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

KL_CHUNK = 128


def compute_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    completion_ids: list[int],
    eos_token_id: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Chunked reverse KL(student || teacher) averaged over token positions."""
    C = student_logits.size(0)
    device = student_logits.device
    token_ids = torch.tensor(completion_ids, device=device, dtype=torch.long)

    per_token_kl = torch.zeros(C, device=device, dtype=torch.float32)
    policy_logp = torch.zeros(C, device=device, dtype=torch.float32)
    critic_logp = torch.zeros(C, device=device, dtype=torch.float32)

    for i in range(0, C, KL_CHUNK):
        j = min(i + KL_CHUNK, C)
        s_chunk = student_logits[i:j].float()
        t_chunk = teacher_log_probs[i:j].float()

        s_log = F.log_softmax(s_chunk, dim=-1)
        s_prob = s_log.exp()
        per_token_kl[i:j] = (s_prob * (s_log - t_chunk)).sum(dim=-1)

        chunk_ids = token_ids[i:j]
        idx = torch.arange(j - i, device=device)
        policy_logp[i:j] = s_log[idx, chunk_ids].detach()
        critic_logp[i:j] = t_chunk[idx, chunk_ids]

    loss = per_token_kl.mean()

    with torch.no_grad():
        signal = critic_logp - policy_logp
        metrics = {
            "sdpo/signal_mean": signal.mean().item(),
            "sdpo/signal_std": signal.std().item(),
            "sdpo/len_signal_mean": signal.sum().item() / C,
            "sdpo/policy_logp": policy_logp.mean().item(),
            "sdpo/critic_logp": critic_logp.mean().item(),
        }
        if eos_token_id is not None:
            eos_mask = token_ids == eos_token_id
            if eos_mask.any():
                metrics["sdpo/eos_signal_mean"] = signal[eos_mask].mean().item()
                metrics["sdpo/eos_logp_mean"] = policy_logp[eos_mask].mean().item()
                metrics["sdpo/eos_logratio_mean"] = signal[eos_mask].mean().item()

    return loss, metrics


def forward_student(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    completion_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Student forward pass with Megatron-Core GPTModel.

    Calls model(input_ids, position_ids, attention_mask=None, labels=None)
    to get full logits, then slices at completion positions.

    Returns: (C, V) tensor with gradient attached.
    """
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=STUDENT_MAX_PROMPT_LEN,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)
    C = len(completion_ids)

    comp_ids_t = torch.tensor(completion_ids, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)

    completion_logits = logits[0, prompt_len - 1 : prompt_len + C - 1, :]
    return completion_logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add("logs/trainer.log", level=log_level)

    logger.info("=== SDFT Megatron Trainer Starting ===")

    # ---- Initialize torch.distributed via torchrun ----
    local_rank = init_distributed_trainer()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    assert GRAD_ACCUM_STEPS % world_size == 0, (
        f"GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}) must be divisible by "
        f"num_trainers ({world_size})"
    )
    local_accum_steps = GRAD_ACCUM_STEPS // world_size

    logger.info(f"DDP: rank={rank}/{world_size}, local_rank={local_rank}, "
                f"local_accum_steps={local_accum_steps}")

    # ---- Model + tokenizer ----
    logger.info(f"Loading model: {HF_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    model = load_model(HF_MODEL_PATH)
    model.train()

    # Use padded vocab size from model (Megatron pads for TP alignment)
    unwrapped = model.module if hasattr(model, 'module') else model
    vocab_size = unwrapped.vocab_size
    logger.info(
        f"Model loaded. vocab_size={vocab_size} (tokenizer={tokenizer.vocab_size}) "
        f"GPU mem after load: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
    )

    # ---- PyTorch DDP wrapping ----
    ddp_model = DDP(model, device_ids=[local_rank])
    logger.info("PyTorch DDP wrapping complete.")

    # ---- Optimizer (8-bit Adam, same as Phase 1) ----
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"8-bit Adam optimizer ready. LR={LEARNING_RATE}")

    # ---- Dataset (all ranks load, only rank 0 iterates) ----
    logger.info(f"Loading dataset: {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    collator = SDFTCollator(tokenizer=tokenizer, hindsight_field=HINDSIGHT_FIELD)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, drop_last=True,
    )
    steps_per_epoch = len(dataset) // GRAD_ACCUM_STEPS
    logger.info(f"Dataset: {len(dataset)} examples, {steps_per_epoch} steps/epoch")

    # ---- Rank 0 only: wandb, vLLM, logprob server ----
    vllm_group = None
    logprob_comm = None
    success_cache: dict[str, str] = {}

    if rank == 0:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_NAME or f"sdft-megatron-{MODEL_NAME.split('/')[-1]}-e{NUM_EPOCHS}",
            config={
                "model": MODEL_NAME,
                "env_type": ENV_TYPE,
                "backend": "megatron-bridge-ddp",
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
                "num_trainers": world_size,
                "local_accum_steps": local_accum_steps,
                "num_epochs": NUM_EPOCHS,
                "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
                "loss": "reverse_kl",
                "dataset": TRAIN_DATA_PATH,
                "hindsight_field": HINDSIGHT_FIELD,
            },
        )

        logger.info("Waiting for vLLM server...")
        wait_for_vllm()
        logger.info("Initializing vLLM weight transfer engine...")
        vllm_group = init_vllm_weight_engine(device)
        logger.info("vLLM weight engine ready.")

        logger.info("Waiting for logprob server...")
        wait_for_logprob_server()
        logger.info("Initializing logprob weight transfer engine...")
        logprob_comm = init_logprob_weight_engine(device)
        logger.info("Logprob weight engine ready.")

    # Barrier: all ranks wait for rank 0 to finish setup
    dist.barrier()

    # ---- Training loop ----
    optimizer_step = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_loss_sum: float = 0.0
        epoch_samples: int = 0

        data_iter = iter(dataloader) if rank == 0 else None

        for _step in range(steps_per_epoch):
            t_step_start = time.monotonic()

            # ---- Rank 0: rollout + build broadcast data ----
            t_gen_start = time.monotonic()
            full_pass_rate = None
            if rank == 0:
                items = [next(data_iter) for _ in range(GRAD_ACCUM_STEPS)]

                if ENV_TYPE == "rag":
                    use_reflector = HINDSIGHT_FIELD == "online_feedback"
                    envs = [
                        RagEnv(
                            prompt_text=item["prompt_texts"][0],
                            vllm_base_url=VLLM_BASE_URL,
                            privileged_information_prompt=item["conditional_texts"][0],
                            raw_question=item["raw_questions"][0],
                            golden_answer=item["golden_answers"][0],
                            normalized_messages=item["normalized_messages"][0],
                            tokenizer=tokenizer,
                            use_reflector=use_reflector,
                        )
                        for item in items
                    ]
                else:
                    envs = [
                        ApiAdapterEnv(
                            prompt_text=item["prompt_texts"][0],
                            vllm_base_url=VLLM_BASE_URL,
                            raw_question=item["raw_questions"][0],
                            golden_answer=item["golden_answers"][0],
                            tokenizer=tokenizer,
                            success_cache=success_cache,
                        )
                        for item in items
                    ]

                with ThreadPoolExecutor(max_workers=min(16, len(envs))) as executor:
                    list(executor.map(lambda e: e.run(), envs))

                if ENV_TYPE == "api_adapter":
                    # Cache successful adapter responses
                    for env in envs:
                        if env.episode_result and env.completion_text:
                            parsed_verdict, parsed_feedback = env.parse_adapter_response(env.completion_text)
                            if parsed_verdict:
                                cached_text = f"Verdict: {parsed_verdict}\nFeedback: {parsed_feedback}"
                                success_cache[env.raw_question] = cached_text

                # Build broadcast payload
                rollout_data = [
                    {
                        "prompt_text": env.prompt_text,
                        "completion_text": env.completion_text,
                        "privileged_information_prompt": env.privileged_information_prompt,
                    }
                    for env in envs
                ]

                # Compute pass_rate (env-type dependent)
                if ENV_TYPE == "api_adapter":
                    pass_results = [env.episode_result for env in envs if env.episode_result is not None]
                    full_pass_rate = sum(1.0 for r in pass_results if r) / max(len(pass_results), 1)
                elif ENV_TYPE == "rag" and envs[0].reflector_result is not None:
                    verdicts = [env.reflector_result["verdict"] for env in envs if env.reflector_result]
                    full_pass_rate = sum(1.0 for v in verdicts if v == "PASS") / max(len(verdicts), 1)
            else:
                rollout_data = [None] * GRAD_ACCUM_STEPS
            t_generation = time.monotonic() - t_gen_start

            # ---- Broadcast rollout data to all ranks ----
            t_bcast_start = time.monotonic()
            dist.broadcast_object_list(rollout_data, src=0)
            t_broadcast = time.monotonic() - t_bcast_start

            # ---- Each rank slices its portion ----
            my_items = rollout_data[rank * local_accum_steps : (rank + 1) * local_accum_steps]

            # ---- Pre-tokenize all micro-step items ----
            optimizer.zero_grad()
            accum_loss_sum: float = 0.0
            accum_samples: int = 0
            accum_comp_len_sum: int = 0
            accum_metrics: dict[str, list[float]] = {}
            t_student_sum: float = 0.0
            t_loss_bwd_sum: float = 0.0

            micro_data: list[tuple[dict, list[int], list[int]]] = []
            for item_data in my_items:
                completion_ids: list[int] = tokenizer.encode(
                    item_data["completion_text"], add_special_tokens=False,
                )
                if len(completion_ids) == 0:
                    logger.warning("Empty completion, skipping")
                    continue
                completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]
                cond_ids: list[int] = tokenizer.encode(
                    item_data["privileged_information_prompt"],
                    add_special_tokens=False, truncation=True, max_length=TEACHER_MAX_PROMPT_LEN,
                )
                micro_data.append((item_data, completion_ids, cond_ids))

            # ---- Batched teacher log-probs (one HTTP call) ----
            t0 = time.monotonic()
            teacher_items = [(cond + comp, len(cond)) for _, comp, cond in micro_data]
            all_teacher_logprobs = request_teacher_log_probs_batch_http(
                teacher_items, vocab_size, device,
            )
            t_teacher = time.monotonic() - t0

            # ---- Gradient accumulation loop ----
            for micro_step, ((item_data, completion_ids, _cond_ids), teacher_log_probs) in enumerate(
                zip(micro_data, all_teacher_logprobs),
            ):
                # Student forward pass (use raw model — DDP hooks are on params)
                t0 = time.monotonic()
                student_logits = forward_student(
                    model, tokenizer, item_data["prompt_text"], completion_ids, device,
                )  # (C, V)
                t_student_sum += time.monotonic() - t0

                # Reverse KL loss
                t0 = time.monotonic()
                loss, step_metrics = compute_kl(
                    student_logits, teacher_log_probs.detach(),
                    completion_ids, tokenizer.eos_token_id,
                )

                # no_sync on non-final micro-steps (skip allreduce)
                is_final = (micro_step == len(micro_data) - 1)
                ctx = nullcontext() if is_final else ddp_model.no_sync()
                with ctx:
                    scaled_loss = loss / local_accum_steps
                    scaled_loss.backward()
                t_loss_bwd_sum += time.monotonic() - t0

                loss_val = loss.item()
                accum_loss_sum += loss_val
                accum_samples += 1
                accum_comp_len_sum += len(completion_ids)
                epoch_loss_sum += loss_val
                epoch_samples += 1
                for k, v in step_metrics.items():
                    accum_metrics.setdefault(k, []).append(v)

            # ---- Optimizer step ----
            t0 = time.monotonic()
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer_step += 1
            t_optimizer = time.monotonic() - t0

            # ---- Aggregate loss across ranks ----
            t0 = time.monotonic()
            loss_tensor = torch.tensor(
                [accum_loss_sum, float(accum_samples)], device=device,
            )
            dist.all_reduce(loss_tensor)
            t_allreduce = time.monotonic() - t0

            t_weight_sync: float = 0.0
            if rank == 0:
                total_loss = loss_tensor[0].item()
                total_samples = max(loss_tensor[1].item(), 1)
                avg_loss = total_loss / total_samples
                avg_comp_len = accum_comp_len_sum / max(accum_samples, 1)

                log_dict: dict = {
                    "train/loss": avg_loss,
                    "train/completion_length": avg_comp_len,
                    "train/epoch": epoch,
                    "train/lr": LEARNING_RATE,
                }
                if full_pass_rate is not None:
                    key = "reflector/pass_rate" if ENV_TYPE == "rag" else "episode/pass_rate"
                    log_dict[key] = full_pass_rate
                for k, vals in accum_metrics.items():
                    log_dict[k] = sum(vals) / len(vals)

                if optimizer_step % 10 == 0 and envs:
                    env = envs[-1]
                    if hasattr(env, "adapter_history"):
                        table = wandb.Table(
                            columns=["step", "question", "golden_answer", "num_turns", "verdict", "conversation"],
                        )
                        conversation = "\n".join(str(msg) for msg in env.adapter_history)
                        table.add_data(
                            optimizer_step, env.raw_question, env.golden_answer,
                            len(env.adapter_history), env.verdict, conversation,
                        )
                        log_dict["episode/sample"] = table

                wandb.log(log_dict, step=optimizer_step)
                logger.info(f"opt_step={optimizer_step} loss={avg_loss:.4f} comp_len={avg_comp_len:.0f}")

                # ---- Sync weights (rank 0 only) ----
                t0 = time.monotonic()
                sync_weights_to_logprob_server(model, logprob_comm)
                sync_weights_to_vllm(model, device, vllm_group)
                t_weight_sync = time.monotonic() - t0

                # ---- Checkpoint ----
                if optimizer_step % SAVE_EVERY == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
                    save_hf_checkpoint(model, ckpt_dir, tokenizer)

            # All ranks wait for rank 0 weight sync before next step
            t0 = time.monotonic()
            dist.barrier()
            t_barrier = time.monotonic() - t0

            t_total = time.monotonic() - t_step_start
            if rank == 0:
                logger.info(
                    f"TIMING step={optimizer_step} | total={t_total:.1f}s gen={t_generation:.1f}s "
                    f"teacher={t_teacher:.1f}s student={t_student_sum:.1f}s "
                    f"loss_bwd={t_loss_bwd_sum:.1f}s optim={t_optimizer:.1f}s wsync={t_weight_sync:.1f}s"
                )

        # ---- Epoch summary ----
        avg_epoch_loss = epoch_loss_sum / max(epoch_samples, 1)
        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} done. "
            f"avg_loss={avg_epoch_loss:.4f} samples={epoch_samples}"
        )
        if rank == 0:
            wandb.log(
                {"epoch/avg_loss": avg_epoch_loss, "epoch/number": epoch + 1},
                step=optimizer_step,
            )
            ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
            save_hf_checkpoint(model, ckpt_dir, tokenizer)

    # ---- Final checkpoint + shutdown ----
    if rank == 0:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
        save_hf_checkpoint(model, ckpt_dir, tokenizer)
        wandb.finish()

    cleanup()
    logger.info("Training complete.")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    train()
