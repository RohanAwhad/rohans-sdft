"""SDFT trainer — Megatron Bridge version with DDP support.

Orchestrates:
    1. ApiAdapterEnv rollout (rank 0 only, broadcast to all ranks)
    2. Student forward pass (Megatron-Core GPTModel via MegatronDDP)
    3. Teacher log-probs (HTTP from logprob server, each rank independently)
    4. Reverse KL loss + backward (with gradient accumulation + no_sync)
    5. Step-level weight sync to both servers (rank 0 only)

Launch: torchrun --nproc_per_node=N -m megatron_trainer.trainer
"""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from megatron_trainer.collator import SDFTCollator
from megatron_trainer.config import (
    BATCH_SIZE,
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
    TRAIN_DATA_PATH,
    VLLM_BASE_URL,
    WANDB_PROJECT,
    WANDB_ENTITY,
    WANDB_NAME,
)
from megatron_trainer.env import ApiAdapterEnv
from megatron_trainer.model_utils import (
    cleanup,
    init_distributed_trainer,
    load_model,
    save_hf_checkpoint,
)
from megatron_trainer.logprob_client import (
    init_logprob_weight_engine,
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
        max_length=2048,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)
    C = len(completion_ids)

    comp_ids_t = torch.tensor(completion_ids, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    def _fwd(ids, pos_ids):
        return model(input_ids=ids, position_ids=pos_ids, attention_mask=None)

    logits = torch.utils.checkpoint.checkpoint(
        _fwd, input_ids, position_ids, use_reentrant=False,
    )

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

    # ---- MegatronDDP wrapping ----
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import DistributedDataParallelConfig

    ddp_config = DistributedDataParallelConfig()
    ddp_model = MegatronDDP(
        config=model.config, ddp_config=ddp_config, module=model,
    )
    logger.info("MegatronDDP wrapping complete.")

    # ---- Megatron distributed optimizer (replaces bitsandbytes) ----
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=LEARNING_RATE,
        bf16=True,
        clip_grad=MAX_GRAD_NORM,
        use_distributed_optimizer=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.01,
    )
    optimizer = get_megatron_optimizer(opt_config, model_chunks=[ddp_model])
    logger.info(f"Megatron distributed optimizer ready. LR={LEARNING_RATE}")

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
            # ---- Rank 0: rollout + build broadcast data ----
            if rank == 0:
                items = [next(data_iter) for _ in range(GRAD_ACCUM_STEPS)]

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
                        "episode_result": env.episode_result,
                        "raw_question": env.raw_question,
                        "golden_answer": env.golden_answer,
                        "verdict": env.verdict,
                        "feedback": env.feedback,
                    }
                    for env in envs
                ]

                # Compute full pass_rate before slicing (rank 0 has all data)
                pass_results = [d["episode_result"] for d in rollout_data if d["episode_result"] is not None]
                full_pass_rate = sum(1.0 for r in pass_results if r) / max(len(pass_results), 1)
            else:
                rollout_data = [None] * GRAD_ACCUM_STEPS

            # ---- Broadcast rollout data to all ranks ----
            dist.broadcast_object_list(rollout_data, src=0)

            # ---- Each rank slices its portion ----
            my_items = rollout_data[rank * local_accum_steps : (rank + 1) * local_accum_steps]

            # ---- Gradient accumulation loop ----
            ddp_model.zero_grad_buffer()
            accum_loss_sum: float = 0.0
            accum_samples: int = 0
            accum_comp_len_sum: int = 0
            accum_metrics: dict[str, list[float]] = {}

            for micro_step, item_data in enumerate(my_items):
                completion_ids: list[int] = tokenizer.encode(
                    item_data["completion_text"], add_special_tokens=False,
                )
                if len(completion_ids) == 0:
                    logger.warning(f"Empty completion, skipping micro_step {micro_step}")
                    continue
                completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

                # Teacher log-probs via HTTP (each rank independently)
                cond_ids: list[int] = tokenizer.encode(
                    item_data["privileged_information_prompt"],
                    add_special_tokens=False, truncation=True, max_length=2048,
                )
                teacher_log_probs = request_teacher_log_probs_http(
                    token_ids=cond_ids + completion_ids,
                    prompt_len=len(cond_ids),
                    vocab_size=vocab_size,
                    device=device,
                )  # (C, V)

                # Student forward pass (use raw model — DDP hooks are on params)
                student_logits = forward_student(
                    model, tokenizer, item_data["prompt_text"], completion_ids, device,
                )  # (C, V)

                # Reverse KL loss
                loss, step_metrics = compute_kl(
                    student_logits, teacher_log_probs.detach(),
                    completion_ids, tokenizer.eos_token_id,
                )

                # no_sync on non-final micro-steps (skip allreduce)
                is_final = (micro_step == local_accum_steps - 1)
                ctx = nullcontext() if is_final else ddp_model.no_sync()
                with ctx:
                    scaled_loss = loss / local_accum_steps
                    scaled_loss.backward()

                loss_val = loss.item()
                accum_loss_sum += loss_val
                accum_samples += 1
                accum_comp_len_sum += len(completion_ids)
                epoch_loss_sum += loss_val
                epoch_samples += 1
                for k, v in step_metrics.items():
                    accum_metrics.setdefault(k, []).append(v)

            # ---- Optimizer step (grad clipping is internal) ----
            optimizer.step()
            optimizer_step += 1

            # ---- Aggregate loss across ranks ----
            loss_tensor = torch.tensor(
                [accum_loss_sum, float(accum_samples)], device=device,
            )
            dist.all_reduce(loss_tensor)

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
                    "episode/pass_rate": full_pass_rate,
                }
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
                sync_weights_to_logprob_server(model, logprob_comm)
                sync_weights_to_vllm(model, device, vllm_group)

                # ---- Checkpoint ----
                if optimizer_step % SAVE_EVERY == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
                    save_hf_checkpoint(model, ckpt_dir, tokenizer)

            # All ranks wait for rank 0 weight sync before next step
            dist.barrier()

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
