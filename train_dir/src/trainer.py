"""SDFT trainer — main training loop.

Orchestrates:
    1. vLLM rollout generation (HTTP)
    2. Student forward pass (local, with gradients)
    3. Teacher log-probs (NCCL from logprob server)
    4. Reverse KL loss + backward
    5. Epoch-level weight sync to both servers
"""

import os

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
    TRAIN_DATA_PATH,
)
from src.nccl_comm import (
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    broadcast_weights,
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


def compute_reverse_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Reverse KL(student || teacher) averaged over token positions.

    Args:
        student_logits:    (C, V) bfloat16, with gradient
        teacher_log_probs: (C, V) float32, detached

    Returns:
        scalar loss (float32)
    """
    # Upcast student to float32 for numerical stability
    s_log = F.log_softmax(student_logits.float(), dim=-1)  # (C, V)
    s_prob = s_log.exp()  # (C, V)
    # KL(p_s || p_t) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
    per_token_kl = (s_prob * (s_log - teacher_log_probs)).sum(dim=-1)  # (C,)
    return per_token_kl.mean()


def forward_student(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    completion_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Student forward pass. Returns logits at completion positions.

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

    comp_ids_t = torch.tensor(completion_ids, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)  # (1, P+C)
    attn_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits[0]  # (P+C, V)

    # Position P-1 predicts completion token 0, ..., P+C-2 predicts token C-1
    C = len(completion_ids)
    completion_logits = logits[prompt_len - 1 : prompt_len + C - 1, :]  # (C, V)
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
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    vocab_size = model.config.vocab_size
    logger.info(f"Model loaded. vocab_size={vocab_size}")

    # ---- Dataset ----
    logger.info(f"Loading dataset: {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    collator = SDFTCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    logger.info(f"Dataset: {len(dataset)} examples, ~{len(dataset)} steps/epoch")

    # ---- Optimizer (constant LR) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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

    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_loss_sum = 0.0
        epoch_samples = 0
        accum_loss_sum = 0.0
        accum_samples = 0
        accum_comp_len_sum = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            prompt_text = batch["prompt_texts"][0]  # batch_size=1
            conditional_text = batch["conditional_texts"][0]

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

            # 2. Student forward pass (with gradients)
            student_logits = forward_student(
                model, tokenizer, prompt_text, completion_ids, DEVICE
            )  # (C, V)

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

            # 4. Reverse KL loss
            loss = compute_reverse_kl(student_logits, teacher_log_probs.detach())
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            # Accumulate metrics
            loss_val = loss.item()
            accum_loss_sum += loss_val
            accum_samples += 1
            accum_comp_len_sum += len(completion_ids)
            epoch_loss_sum += loss_val
            epoch_samples += 1
            global_step += 1

            # 5. Optimizer step at accumulation boundary
            if global_step % GRAD_ACCUM_STEPS == 0:
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                avg_loss = accum_loss_sum / accum_samples
                avg_comp_len = accum_comp_len_sum / accum_samples

                log_dict = {
                    "train/loss": avg_loss,
                    "train/completion_length": avg_comp_len,
                    "train/epoch": epoch,
                }

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

        # Flush remaining accumulated gradients
        if global_step % GRAD_ACCUM_STEPS != 0:
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

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

        # ---- Weight sync ----
        logger.info("Syncing weights to logprob server...")
        send_command(CMD_SYNC_WEIGHTS, DEVICE)
        broadcast_weights(model, src=0)

        logger.info("Syncing weights to vLLM...")
        sync_weights_to_vllm(model, DEVICE, vllm_group)
        logger.info("Weight sync complete.")

        # ---- Checkpoint ----
        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Checkpoint saved: {ckpt_dir}")

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
