"""SDFT trainer — main training loop.

Orchestrates:
    1. vLLM rollout generation (HTTP)
    2. Student forward pass (local, with gradients)
    3. Teacher log-probs (NCCL from logprob server)
    4. Reverse KL loss + backward
    5. Epoch-level weight sync to both servers
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from src.collator import SDFTCollator
from src.config import (
    BATCH_SIZE,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MODEL_NAME,
    NCCL_MASTER_PORT,
    NUM_EPOCHS,
    OUTPUT_DIR,
    HINDSIGHT_FIELD,
    SAVE_EVERY,
    TRAIN_DATA_PATH,
)
from src.config import EMA_ALPHA
from src.nccl_comm import (
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    broadcast_weights_ema,
    cleanup,
    init_nccl,
    request_teacher_log_probs,
    send_command,
)
from src.vllm_utils import (
    init_vllm_weight_engine,
    sync_weights_to_vllm,
    vllm_generate,
    wait_for_vllm,
)

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


KL_CHUNK = 128  # tokens per chunk to avoid OOM on large models


def compute_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    completion_ids: list[int],
    eos_token_id: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Chunked reverse KL(student || teacher) averaged over token positions.

    Processes KL_CHUNK tokens at a time to avoid materializing full (C, V)
    intermediates. Gradient flows through slice assignment into per_token_kl.

    Args:
        student_logits:    (C, V) bfloat16, with gradient
        teacher_log_probs: (C, V) bfloat16, detached (log-softmax)
        completion_ids:    token IDs of the completion (for signal metrics)
        eos_token_id:      EOS token ID

    Returns:
        (loss, metrics_dict)
    """
    C = student_logits.size(0)
    device = student_logits.device
    token_ids = torch.tensor(completion_ids, device=device, dtype=torch.long)

    # Pre-allocate outputs
    per_token_kl = torch.zeros(C, device=device, dtype=torch.float32)
    policy_logp = torch.zeros(C, device=device, dtype=torch.float32)
    critic_logp = torch.zeros(C, device=device, dtype=torch.float32)

    for i in range(0, C, KL_CHUNK):
        j = min(i + KL_CHUNK, C)
        s_chunk = student_logits[i:j].float()       # (chunk, V) — has grad
        t_chunk = teacher_log_probs[i:j].float()     # (chunk, V) — detached

        s_log = F.log_softmax(s_chunk, dim=-1)
        s_prob = s_log.exp()
        # KL(p_s || p_t) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
        per_token_kl[i:j] = (s_prob * (s_log - t_chunk)).sum(dim=-1)

        # Signal metrics at generated tokens (detached)
        chunk_ids = token_ids[i:j]
        idx = torch.arange(j - i, device=device)
        policy_logp[i:j] = s_log[idx, chunk_ids].detach()
        critic_logp[i:j] = t_chunk[idx, chunk_ids]

    loss = per_token_kl.mean()

    # --- SDPO-style signal metrics ---
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
    """Student forward pass. Returns logits at completion positions.

    Uses backbone (model.model) + selective lm_head to avoid allocating
    the full (1, S, V) logits tensor. For 8B models with V=152K, the full
    logits can be >1 GB vs ~30 MB for hidden states.

    Returns: (C, V) tensor with gradient attached.
    """
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"][0]  # (P,)
    prompt_len = prompt_ids.size(0)
    C = len(completion_ids)

    comp_ids_t = torch.tensor(completion_ids, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)  # (1, P+C)
    attn_mask = torch.ones_like(input_ids)

    # Run backbone under torch.utils.checkpoint so the full (1, S, hidden_dim)
    # hidden states are recomputed during backward instead of stored.
    def _backbone_fwd(ids, mask):
        return model.model(input_ids=ids, attention_mask=mask)[0]

    hidden = torch.utils.checkpoint.checkpoint(
        _backbone_fwd, input_ids, attn_mask, use_reentrant=False,
    )  # (1, S, hidden_dim) — recomputed on backward, not stored

    # Extract completion positions, apply lm_head selectively
    # Position P-1 predicts completion token 0, ..., P+C-2 predicts token C-1
    completion_hidden = hidden[0, prompt_len - 1 : prompt_len + C - 1, :]  # (C, hidden_dim)
    completion_logits = model.lm_head(completion_hidden)  # (C, V)
    return completion_logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add("logs/trainer.log", level=log_level)

    logger.info("=== SDFT Trainer Starting ===")

    # ---- Model + tokenizer ----
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=DEVICE,  # load directly to GPU, skip CPU→GPU copy
    )
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    vocab_size = model.config.vocab_size
    logger.info(
        f"Model loaded. vocab_size={vocab_size} "
        f"GPU mem after load: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB"
    )

    # ---- Dataset ----
    logger.info(f"Loading dataset: {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    collator = SDFTCollator(tokenizer=tokenizer, hindsight_field=HINDSIGHT_FIELD)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    logger.info(f"Dataset: {len(dataset)} examples, ~{len(dataset)} steps/epoch")

    # ---- Optimizer (8-bit Adam to fit 8B models on single GPU) ----
    import bitsandbytes as bnb
    import math
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)

    # ---- LR scheduler (constant) ----
    steps_per_epoch = math.ceil(len(dataset) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    logger.info(f"Constant LR: {total_steps} total steps, LR={LEARNING_RATE}")

    # ---- wandb ----
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sdft-online"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get(
            "WANDB_NAME", f"sdft-{MODEL_NAME.split('/')[-1]}-e{NUM_EPOCHS}"
        ),
        config={
            "model": MODEL_NAME,
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
        },
    )

    # ---- Wait for vLLM ----
    logger.info("Waiting for vLLM server...")
    wait_for_vllm()

    # ---- Init NCCL for logprob server (rank 0) ----
    logger.info("Initializing NCCL for logprob server...")
    init_nccl(rank=0, world_size=2, master_port=NCCL_MASTER_PORT)
    logger.info("NCCL initialized (trainer=rank0, server=rank1).")

    # ---- Init vLLM weight engine (separate NCCL group) ----
    logger.info("Initializing vLLM weight transfer engine...")
    vllm_group = init_vllm_weight_engine(DEVICE)
    logger.info("vLLM weight engine ready.")

    # ---- Training loop ----
    global_step = 0
    optimizer_step = 0
    offline_overfit = os.environ.get("OFFLINE_OVERFIT", "0") == "1"
    cached_data: list[tuple[str, list[int], torch.Tensor]] = []  # (prompt, comp_ids, teacher_lp)

    if offline_overfit:
        logger.info("OFFLINE_OVERFIT mode: epoch 1 caches data, epochs 2+ replay (no gen/teacher/sync)")

    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_loss_sum = 0.0
        epoch_samples = 0
        accum_loss_sum = 0.0
        accum_samples = 0
        accum_comp_len_sum = 0
        accum_metrics: dict[str, list[float]] = {}

        optimizer.zero_grad()

        # Determine data source: dataloader (epoch 1 or normal) vs cache
        use_cache = offline_overfit and epoch > 0
        data_iter = cached_data if use_cache else dataloader

        for batch_idx, item in enumerate(data_iter):
            if use_cache:
                prompt_text, completion_ids, teacher_log_probs = item
                completion_text = ""  # not needed for cached replay
            else:
                prompt_text = item["prompt_texts"][0]  # batch_size=1
                conditional_text = item["conditional_texts"][0]

                # 1. Generate completion via vLLM
                completion_text = vllm_generate(prompt_text)
                completion_ids = tokenizer.encode(
                    completion_text, add_special_tokens=False
                )

                # Append EOS so model learns to stop
                if tokenizer.eos_token_id is not None:
                    completion_ids.append(tokenizer.eos_token_id)

                if len(completion_ids) == 0:
                    logger.warning(f"Empty completion at step {global_step}, skipping")
                    continue

                # Truncate if needed
                completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

                # 3. Teacher log-probs via NCCL (no grad)
                cond_ids = tokenizer.encode(
                    conditional_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=2048,
                )
                teacher_log_probs = request_teacher_log_probs(
                    token_ids=cond_ids + completion_ids,
                    prompt_len=len(cond_ids),
                    vocab_size=vocab_size,
                    device=DEVICE,
                )  # (C, V) float32

                # Cache for offline overfit replay
                if offline_overfit:
                    cached_data.append((prompt_text, completion_ids, teacher_log_probs.detach().cpu()))

            # 2. Student forward pass (with gradients)
            student_logits = forward_student(
                model, tokenizer, prompt_text, completion_ids, DEVICE
            )  # (C, V)

            # Move teacher log-probs to device if from cache
            t_lp = teacher_log_probs.to(DEVICE) if use_cache else teacher_log_probs

            # 4. Reverse KL loss + signal metrics
            loss, step_metrics = compute_kl(
                student_logits, t_lp.detach(),
                completion_ids, tokenizer.eos_token_id,
            )
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            # Accumulate metrics
            loss_val = loss.item()
            accum_loss_sum += loss_val
            accum_samples += 1
            accum_comp_len_sum += len(completion_ids)
            for k, v in step_metrics.items():
                accum_metrics.setdefault(k, []).append(v)
            epoch_loss_sum += loss_val
            epoch_samples += 1
            global_step += 1

            # 5. Optimizer step at accumulation boundary
            if global_step % GRAD_ACCUM_STEPS == 0:
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                avg_loss = accum_loss_sum / accum_samples
                avg_comp_len = accum_comp_len_sum / accum_samples

                log_dict = {
                    "train/loss": avg_loss,
                    "train/completion_length": avg_comp_len,
                    "train/epoch": epoch,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                for k, vals in accum_metrics.items():
                    log_dict[k] = sum(vals) / len(vals)

                # Sample preview every 10 optimizer steps
                if optimizer_step % 10 == 0:
                    log_dict["samples/prompt"] = wandb.Html(
                        f"<pre>{_escape(prompt_text[:1000])}</pre>"
                    )
                    log_dict["samples/completion"] = wandb.Html(
                        f"<pre>{_escape(completion_text[:1000])}</pre>"
                    )

                wandb.log(log_dict, step=optimizer_step)
                logger.info(
                    f"opt_step={optimizer_step} loss={avg_loss:.4f} "
                    f"comp_len={avg_comp_len:.0f}"
                )

                accum_loss_sum = 0.0
                accum_samples = 0
                accum_comp_len_sum = 0
                accum_metrics = {}

                # ---- Step-level weight sync (skip in offline overfit) ----
                if not offline_overfit:
                    send_command(CMD_SYNC_WEIGHTS, DEVICE)
                    broadcast_weights_ema(model, alpha=EMA_ALPHA, src=0)
                    sync_weights_to_vllm(model, DEVICE, vllm_group)

                # ---- Step-level checkpoint ----
                if optimizer_step % SAVE_EVERY == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"Checkpoint saved: {ckpt_dir}")

        # Flush remaining accumulated gradients
        if global_step % GRAD_ACCUM_STEPS != 0:
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1
            if not offline_overfit:
                send_command(CMD_SYNC_WEIGHTS, DEVICE)
                broadcast_weights_ema(model, alpha=EMA_ALPHA, src=0)
                sync_weights_to_vllm(model, DEVICE, vllm_group)

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

    # ---- Final checkpoint ----
    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info(f"Final checkpoint saved: {ckpt_dir}")

    # ---- Shutdown ----
    send_command(CMD_SHUTDOWN, DEVICE)
    cleanup()
    wandb.finish()
    logger.info("Training complete.")


def _escape(text: str) -> str:
    """Minimal HTML escaping for wandb preview."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    train()
