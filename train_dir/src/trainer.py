"""SDFT trainer — main training loop.

Orchestrates:
    1. vLLM rollout generation (HTTP)
    2. Student forward pass (local, with gradients)
    3. Teacher log-probs (NCCL from logprob server)
    4. Reverse KL loss + backward
    5. Epoch-level weight sync to both servers
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import load_dataset
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
import math

import wandb
from src.collator import SDFTCollator
from src.env import RagEnv
from src.loss import compute_kl
from src.student import forward_student
from src.config import (
    BATCH_SIZE,
    EMA_ALPHA,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LR_SCHEDULER,
    MAX_GRAD_NORM,
    MODEL_NAME,
    NCCL_MASTER_PORT,
    NUM_EPOCHS,
    ONLINE_HINDSIGHT_FIELDS,
    OUTPUT_DIR,
    HINDSIGHT_FIELD,
    SAVE_EVERY,
    STUDENT_MAX_PROMPT_LEN,
    TEACHER_MAX_PROMPT_LEN,
    TOPK_K,
    TRAIN_DATA_PATH,
    VLLM_BASE_URL,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from src.nccl_comm import (
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    broadcast_weights_ema,
    cleanup,
    init_nccl,
    request_teacher_log_probs,
    request_teacher_log_probs_topk,
    send_command,
)
from src.vllm_utils import (
    init_vllm_weight_engine,
    sync_weights_to_vllm,
    wait_for_vllm,
)

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
  # === Model loading and setup ===
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
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  vocab_size = model.config.vocab_size
  logger.info(
      f"Model loaded. vocab_size={vocab_size} "
      f"GPU mem after load: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB"
  )

  if TOPK_K != 0:
    assert 0 < TOPK_K < vocab_size, \
        f"TOPK_K must be in [0, {vocab_size}), got {TOPK_K}"
    logger.info(f"Top-K reverse KL enabled: k={TOPK_K}")
  else:
    logger.info("Top-K reverse KL disabled (full vocab)")

  dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
  collator = SDFTCollator(tokenizer=tokenizer, hindsight_field=HINDSIGHT_FIELD)
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, drop_last=True)
  logger.info(f"Dataset: {len(dataset)} examples")

  # full AdamW OOMs on longer sequences; 8-bit keeps optimizer states in int8
  optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=WEIGHT_DECAY)
  steps_per_epoch = math.ceil(len(dataset) / GRAD_ACCUM_STEPS)
  total_steps = steps_per_epoch * NUM_EPOCHS
  if LR_SCHEDULER == "cosine":
    warmup_steps = min(WARMUP_STEPS, int(0.1 * total_steps))
    def _cosine_with_warmup(step: int) -> float:
      if step < warmup_steps:
        return step / warmup_steps
      progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
      return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_with_warmup)
    logger.info(f"Cosine LR: {total_steps} total steps, {warmup_steps} warmup, LR={LEARNING_RATE}")
  else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    logger.info(f"Constant LR: {total_steps} total steps, LR={LEARNING_RATE}")

  wandb.init(
      project=os.environ.get("WANDB_PROJECT", "sdft-online"),
      entity=os.environ.get("WANDB_ENTITY"),
      name=os.environ.get(
          "WANDB_NAME", f"sdft-{MODEL_NAME.split('/')[-1]}-e{NUM_EPOCHS}"
      ),
      config={
          "model": MODEL_NAME,
          "learning_rate": LEARNING_RATE,
          "optimizer": "AdamW8bit",
          "optimizer_betas": (0.9, 0.95),
          "optimizer_eps": 1e-8,
          "weight_decay": WEIGHT_DECAY,
          "batch_size": BATCH_SIZE,
          "grad_accum_steps": GRAD_ACCUM_STEPS,
          "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
          "num_epochs": NUM_EPOCHS,
          "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
          "loss": "reverse_kl",
          "topk_k": TOPK_K,
          "lr_scheduler": LR_SCHEDULER,
          "warmup_steps": warmup_steps if LR_SCHEDULER == "cosine" else 0,
          "total_optimizer_steps": total_steps,
          "dataset": TRAIN_DATA_PATH,
          "hindsight_field": HINDSIGHT_FIELD,
          "student_max_prompt_len": STUDENT_MAX_PROMPT_LEN,
          "teacher_max_prompt_len": TEACHER_MAX_PROMPT_LEN,
      },
  )
  if HINDSIGHT_FIELD in ONLINE_HINDSIGHT_FIELDS:
    from src.config import REFLECTOR_MODEL
    logger.info(f"Hindsight: {HINDSIGHT_FIELD} (reflector={REFLECTOR_MODEL})")
  else:
    logger.info(f"Hindsight: {HINDSIGHT_FIELD} (static)")
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


  # === Training Loop ===
  optimizer_step = 0
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

        # --- Rollout: generate completions + optional reflector feedback ---
        is_online = HINDSIGHT_FIELD in ONLINE_HINDSIGHT_FIELDS
        envs = [
          RagEnv(
            prompt_text=item["prompt_texts"][0],
            vllm_base_url=VLLM_BASE_URL,
            privileged_information_prompt=item["conditional_texts"][0] if not is_online else None,
            raw_question=item["raw_questions"][0],
            golden_answer=item["golden_answers"][0],
            normalized_messages=item["normalized_messages"][0],
            tokenizer=tokenizer,
            hindsight_field=HINDSIGHT_FIELD if is_online else None,
          )
          for item in items
        ]
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=min(64, len(envs))) as executor:
          list(executor.map(lambda e: e.run(), envs))
        end = time.monotonic()
        logger.info(f"Rollout Time: {end-start:0.2f} secs")

        start = time.monotonic()
        for micro_step, env in enumerate(envs):
          # TODO: (rohan) we only have support for BATCH_SIZE=1
          completion_ids: list[int] = tokenizer.encode(env.completion_text, add_special_tokens=False)
          if tokenizer.eos_token_id is not None:
            completion_ids.append(tokenizer.eos_token_id)
          if len(completion_ids) == 0:
            logger.warning(f"Empty completion, skipping micro_step {micro_step}")
            continue
          completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

          cond_ids: list[int] = tokenizer.encode(
            env.privileged_information_prompt, add_special_tokens=False, truncation=True,
            max_length=TEACHER_MAX_PROMPT_LEN,
          )

          if TOPK_K > 0:
            # Top-k: student forward first (need logits for top-k indices)
            start_stud_micro_step_time = time.monotonic()
            student_logits = forward_student(model, tokenizer, env.prompt_text, completion_ids, DEVICE)
            with torch.no_grad():
              _, topk_indices = torch.topk(student_logits.detach(), TOPK_K, dim=-1)

            start_teacher_lp_time = time.monotonic()
            teacher_log_probs = request_teacher_log_probs_topk(
              token_ids=cond_ids + completion_ids,
              prompt_len=len(cond_ids),
              topk_indices=topk_indices,
              device=DEVICE,
            )
            end_teacher_lp_time = time.monotonic()
            logger.info(f"  Teacher LogProb Gen Time: {end_teacher_lp_time - start_teacher_lp_time:0.2f} secs")

            loss, step_metrics = compute_kl(
              student_logits, teacher_log_probs.detach(),
              completion_ids, tokenizer.eos_token_id,
              topk_indices=topk_indices,
            )
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            end_stud_micro_step_time = time.monotonic()
            logger.info(f"  Micro Step Time: {end_stud_micro_step_time - start_stud_micro_step_time:0.2f} secs")
          else:
            # Full-vocab: teacher first (overlap), then student (current behavior)
            start_teacher_lp_time = time.monotonic()
            teacher_log_probs = request_teacher_log_probs(
              token_ids=cond_ids + completion_ids,
              prompt_len=len(cond_ids),
              vocab_size=vocab_size,
              device=DEVICE,
            )
            end_teacher_lp_time = time.monotonic()
            logger.info(f"  Teacher LogProb Gen Time: {end_teacher_lp_time - start_teacher_lp_time:0.2f} secs")

            start_stud_micro_step_time = time.monotonic()
            student_logits = forward_student(model, tokenizer, env.prompt_text, completion_ids, DEVICE)

            loss, step_metrics = compute_kl(
              student_logits, teacher_log_probs.detach(),
              completion_ids, tokenizer.eos_token_id,
            )
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()
            end_stud_micro_step_time = time.monotonic()
            logger.info(f"  Micro Step Time: {end_stud_micro_step_time - start_stud_micro_step_time:0.2f} secs")

          loss_val = loss.item()
          accum_loss_sum += loss_val
          accum_samples += 1
          accum_comp_len_sum += len(completion_ids)
          epoch_loss_sum += loss_val
          epoch_samples += 1
          for k, v in step_metrics.items():
            accum_metrics.setdefault(k, []).append(v)
          if env.reflector_result is not None:
            accum_metrics.setdefault("reflector/pass_rate", []).append(
              1.0 if env.reflector_result["verdict"] == "PASS" else 0.0
            )

        # backward pass
        # Optimizer step
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        optimizer_step += 1
        end = time.monotonic()
        logger.info(f"Optimization Step Time: {end-start:0.2f} secs")


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
          os.makedirs(ckpt_dir, exist_ok=True)
          model.save_pretrained(ckpt_dir)
          tokenizer.save_pretrained(ckpt_dir)
          logger.info(f"Checkpoint saved: {ckpt_dir}")
    except StopIteration:
      pass
    avg_epoch_loss = epoch_loss_sum / max(epoch_samples, 1)
    logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} done. avg_loss={avg_epoch_loss:.4f} samples={epoch_samples}")
    wandb.log({"epoch/avg_loss": avg_epoch_loss, "epoch/number": epoch + 1}, step=optimizer_step)

  # Final checkpoint
  ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
  os.makedirs(ckpt_dir, exist_ok=True)
  model.save_pretrained(ckpt_dir)
  tokenizer.save_pretrained(ckpt_dir)
  logger.info(f"Final checkpoint saved: {ckpt_dir}")

  send_command(CMD_SHUTDOWN, DEVICE)
  cleanup()
  wandb.finish()
  logger.info("Training complete.")


if __name__ == '__main__':
  if BATCH_SIZE != 1: raise ValueError(f'ONLY batch size == 1 is supported. we got {BATCH_SIZE}')
  train()
