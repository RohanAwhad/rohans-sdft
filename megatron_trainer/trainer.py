"""SDFT trainer — Megatron Bridge version.

Orchestrates:
    1. ApiAdapterEnv rollout (multi-turn adapter + external API)
    2. Student forward pass (Megatron-Core GPTModel, with gradients)
    3. Teacher log-probs (NCCL from logprob server)
    4. Reverse KL loss + backward
    5. Step-level weight sync to both servers
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from megatron_trainer.collator import SDFTCollator
from megatron_trainer.config import (
    BATCH_SIZE,
    EMA_ALPHA,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    HF_MODEL_PATH,
    HINDSIGHT_FIELD,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_NAME,
    NCCL_MASTER_PORT,
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
    init_distributed,
    load_model,
    save_hf_checkpoint,
)
from megatron_trainer.nccl_comm import (
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    broadcast_weights_ema,
    request_teacher_log_probs,
    send_command,
)
from megatron_trainer.vllm_utils import (
    init_vllm_weight_engine,
    sync_weights_to_vllm,
    wait_for_vllm,
)

DEVICE = torch.device("cuda:0")

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

    # ---- Initialize Megatron + torch.distributed ----
    init_distributed(rank=0, world_size=2, master_port=NCCL_MASTER_PORT)
    logger.info("Distributed init complete (trainer=rank0).")

    # ---- Model + tokenizer ----
    logger.info(f"Loading model: {HF_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    model = load_model(HF_MODEL_PATH)
    model.train()

    # Use padded vocab size from model (Megatron pads for TP alignment)
    # The model's output logits have this dimension, not tokenizer.vocab_size
    unwrapped = model.module if hasattr(model, 'module') else model
    vocab_size = unwrapped.vocab_size
    logger.info(
        f"Model loaded. vocab_size={vocab_size} (tokenizer={tokenizer.vocab_size}) "
        f"GPU mem after load: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB"
    )

    # ---- Dataset ----
    logger.info(f"Loading dataset: {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    collator = SDFTCollator(tokenizer=tokenizer, hindsight_field=HINDSIGHT_FIELD)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, drop_last=True
    )
    logger.info(f"Dataset: {len(dataset)} examples")

    # ---- Optimizer (8-bit Adam) ----
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)

    # ---- LR scheduler (constant) ----
    steps_per_epoch = math.ceil(len(dataset) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    logger.info(f"Constant LR: {total_steps} total steps, LR={LEARNING_RATE}")

    # ---- wandb ----
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_NAME or f"sdft-megatron-{MODEL_NAME.split('/')[-1]}-e{NUM_EPOCHS}",
        config={
            "model": MODEL_NAME,
            "backend": "megatron-bridge",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "num_epochs": NUM_EPOCHS,
            "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
            "loss": "reverse_kl",
            "lr_scheduler": "constant",
            "total_optimizer_steps": total_steps,
            "dataset": TRAIN_DATA_PATH,
            "hindsight_field": HINDSIGHT_FIELD,
        },
    )

    # ---- Wait for vLLM ----
    logger.info("Waiting for vLLM server...")
    wait_for_vllm()

    # ---- Init vLLM weight engine (separate NCCL group) ----
    logger.info("Initializing vLLM weight transfer engine...")
    vllm_group = init_vllm_weight_engine(DEVICE)
    logger.info("vLLM weight engine ready.")

    # ---- Training loop ----
    optimizer_step = 0
    success_cache: dict[str, str] = {}  # raw_question -> adapter verdict+feedback text

    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_loss_sum: float = 0.0
        epoch_samples: int = 0
        accum_loss_sum: float = 0.0
        accum_samples: int = 0
        accum_comp_len_sum: int = 0
        accum_metrics: dict[str, list[float]] = {}

        optimizer.zero_grad()
        data_iter = iter(dataloader)

        try:
          while True:
            items = []
            for _ in range(GRAD_ACCUM_STEPS):
              item = next(data_iter)
              items.append(item)

            # --- Rollout: generate completions via ApiAdapterEnv ---
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

            # Cache successful adapter responses for future hindsight
            for env in envs:
              if env.episode_result and env.completion_text:
                parsed_verdict, parsed_feedback = env.parse_adapter_response(env.completion_text)
                if parsed_verdict:
                  cached_text = f"Verdict: {parsed_verdict}\nFeedback: {parsed_feedback}"
                  success_cache[env.raw_question] = cached_text

            for micro_step, env in enumerate(envs):
              completion_ids: list[int] = tokenizer.encode(env.completion_text, add_special_tokens=False)
              if len(completion_ids) == 0:
                logger.warning(f"Empty completion, skipping micro_step {micro_step}")
                continue
              completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

              # Teacher log-probs via NCCL
              cond_ids: list[int] = tokenizer.encode(
                env.privileged_information_prompt, add_special_tokens=False, truncation=True, max_length=2048,
              )
              teacher_log_probs = request_teacher_log_probs(
                token_ids=cond_ids + completion_ids,
                prompt_len=len(cond_ids),
                vocab_size=vocab_size,
                device=DEVICE,
              )  # (C, V)

              # Student forward pass
              student_logits = forward_student(model, tokenizer, env.prompt_text, completion_ids, DEVICE)  # (C, V)

              # Reverse KL loss
              loss, step_metrics = compute_kl(
                student_logits, teacher_log_probs.detach(),
                completion_ids, tokenizer.eos_token_id,
              )
              scaled_loss = loss / GRAD_ACCUM_STEPS
              scaled_loss.backward()

              loss_val = loss.item()
              accum_loss_sum += loss_val
              accum_samples += 1
              accum_comp_len_sum += len(completion_ids)
              epoch_loss_sum += loss_val
              epoch_samples += 1
              for k, v in step_metrics.items():
                accum_metrics.setdefault(k, []).append(v)
              if env.episode_result is not None:
                accum_metrics.setdefault("episode/pass_rate", []).append(
                  1.0 if env.episode_result else 0.0
                )

            # Optimizer step
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1

            avg_loss = accum_loss_sum / max(accum_samples, 1)
            avg_comp_len = accum_comp_len_sum / max(accum_samples, 1)

            log_dict = {
              "train/loss": avg_loss,
              "train/completion_length": avg_comp_len,
              "train/epoch": epoch,
              "train/lr": scheduler.get_last_lr()[0],
            }
            for k, vals in accum_metrics.items():
              log_dict[k] = sum(vals) / len(vals)

            if optimizer_step % 10 == 0 and hasattr(env, "adapter_history"):
              table = wandb.Table(columns=["step", "question", "golden_answer", "num_turns", "verdict", "conversation"])
              conversation = "\n".join(str(msg) for msg in env.adapter_history)
              table.add_data(optimizer_step, env.raw_question, env.golden_answer, len(env.adapter_history), env.verdict, conversation)
              log_dict["episode/sample"] = table

            wandb.log(log_dict, step=optimizer_step)
            logger.info(f"opt_step={optimizer_step} loss={avg_loss:.4f} comp_len={avg_comp_len:.0f}")

            accum_loss_sum = 0.0
            accum_samples = 0
            accum_comp_len_sum = 0
            accum_metrics = {}

            # Sync weights
            send_command(CMD_SYNC_WEIGHTS, DEVICE)
            broadcast_weights_ema(model, alpha=EMA_ALPHA, src=0)
            sync_weights_to_vllm(model, DEVICE, vllm_group)

            # Checkpoint
            if optimizer_step % SAVE_EVERY == 0:
              ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
              save_hf_checkpoint(model, ckpt_dir, tokenizer)
        except StopIteration:
          pass

        # ---- Epoch summary ----
        avg_epoch_loss = epoch_loss_sum / max(epoch_samples, 1)
        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} done. "
            f"avg_loss={avg_epoch_loss:.4f} samples={epoch_samples}"
        )
        wandb.log(
            {"epoch/avg_loss": avg_epoch_loss, "epoch/number": epoch + 1},
            step=optimizer_step,
        )

        # ---- Epoch checkpoint ----
        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
        save_hf_checkpoint(model, ckpt_dir, tokenizer)

    # ---- Final checkpoint ----
    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
    save_hf_checkpoint(model, ckpt_dir, tokenizer)

    # ---- Shutdown ----
    send_command(CMD_SHUTDOWN, DEVICE)
    cleanup()
    wandb.finish()
    logger.info("Training complete.")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    train()
