"""SDFT trainer — Megatron Bridge version.

Orchestrates:
    1. vLLM rollout generation (HTTP)
    2. Student forward pass (Megatron-Core GPTModel, with gradients)
    3. Teacher log-probs (NCCL from logprob server)
    4. Reverse KL loss + backward
    5. Step-level weight sync to both servers
"""

import math
import os
import shutil

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
)
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
    vllm_generate,
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
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
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
        project=os.environ.get("WANDB_PROJECT", "sdft-online"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get(
            "WANDB_NAME", f"sdft-megatron-{MODEL_NAME.split('/')[-1]}-e{NUM_EPOCHS}"
        ),
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
    global_step = 0
    optimizer_step = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_loss_sum = 0.0
        epoch_samples = 0
        accum_loss_sum = 0.0
        accum_samples = 0
        accum_comp_len_sum = 0
        accum_metrics: dict[str, list[float]] = {}

        optimizer.zero_grad()

        for batch_idx, item in enumerate(dataloader):
            prompt_text = item["prompt_texts"][0]
            conditional_text = item["conditional_texts"][0]

            # 1. Generate completion via vLLM
            completion_text = vllm_generate(prompt_text)
            completion_ids = tokenizer.encode(
                completion_text, add_special_tokens=False
            )

            if tokenizer.eos_token_id is not None:
                completion_ids.append(tokenizer.eos_token_id)

            if len(completion_ids) == 0:
                logger.warning(f"Empty completion at step {global_step}, skipping")
                continue

            completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

            # 2. Teacher log-probs via NCCL
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
            )

            # 3. Student forward pass
            student_logits = forward_student(
                model, tokenizer, prompt_text, completion_ids, DEVICE
            )

            # 4. Reverse KL loss
            loss, step_metrics = compute_kl(
                student_logits, teacher_log_probs.detach(),
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

                # ---- Weight sync ----
                send_command(CMD_SYNC_WEIGHTS, DEVICE)
                broadcast_weights_ema(model, alpha=EMA_ALPHA, src=0)
                sync_weights_to_vllm(model, DEVICE, vllm_group)

                # ---- Step-level checkpoint ----
                if optimizer_step % SAVE_EVERY == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
                    save_hf_checkpoint(model, ckpt_dir, tokenizer)

        # Flush remaining accumulated gradients
        if global_step % GRAD_ACCUM_STEPS != 0:
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1
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

        # ---- Epoch checkpoint ----
        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
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
