"""Reproduce the long-completion KL backward memory peak with fixed tensors.

Launch with four GPUs so the model and optimizer match the production FSDP
layout. A short warmup step initializes Adam state before the fixed 8192-token
forward: 2048 prompt tokens followed by 6144 completion tokens.
"""

import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from megatron_trainer.config import HF_MODEL_PATH
from megatron_trainer.model_utils import load_model, register_fsdp_module_mappings


PROMPT_LEN = 2048
COMPLETION_LEN = 6144
TOTAL_LEN = PROMPT_LEN + COMPLETION_LEN
WARMUP_LEN = 512
WARMUP_LOSS_TOKENS = 128
KL_CHUNK = 128
ACCUM_STEPS = 8


def log_memory(label: str, rank: int, device: torch.device) -> None:
    logger.info(
        f"rank={rank} {label} "
        f"allocated={torch.cuda.memory_allocated(device) / 2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved(device) / 2**30:.2f}GiB "
        f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB"
    )


def fixed_tokens(length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(0)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(1, length),
        generator=generator,
        device=device,
        dtype=torch.long,
    )


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    position_ids = torch.arange(
        input_ids.size(1), device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
    )


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Match trainer.compute_kl's chunked reverse-KL loss path."""
    completion_len = student_logits.size(0)
    per_token_kl = torch.zeros(
        completion_len,
        device=student_logits.device,
        dtype=torch.float32,
    )

    for start in range(0, completion_len, KL_CHUNK):
        end = min(start + KL_CHUNK, completion_len)
        student_log_probs = F.log_softmax(student_logits[start:end].float(), dim=-1)
        student_probs = student_log_probs.exp()
        per_token_kl[start:end] = (
            student_probs
            * (student_log_probs - teacher_log_probs[start:end].float())
        ).sum(dim=-1)

    return per_token_kl.mean()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs("logs", exist_ok=True)

    assert world_size == 4, f"Expected four FSDP ranks, got {world_size}"
    assert TOTAL_LEN == 8192

    logger.info(
        f"rank={rank} starting fixed repro model={HF_MODEL_PATH} "
        f"prompt={PROMPT_LEN} completion={COMPLETION_LEN} total={TOTAL_LEN}"
    )

    model = load_model(HF_MODEL_PATH)
    logger.add(
        f"logs/repro_fixed_8192_rank{rank}.log",
        level=os.environ.get("LOGGING_LEVEL", "DEBUG"),
    )
    model.train()
    unwrapped = model.module if hasattr(model, "module") else model
    vocab_size = unwrapped.vocab_size
    logger.info(
        f"rank={rank} config_identity={unwrapped.config is unwrapped.decoder.config} "
        f"recompute={unwrapped.decoder.config.recompute_granularity}/"
        f"{unwrapped.decoder.config.recompute_method}/"
        f"{unwrapped.decoder.config.recompute_num_layers}"
    )
    log_memory("after_model_load", rank, device)

    from megatron.core.distributed import (
        DistributedDataParallelConfig,
        TorchFullyShardedDataParallel,
    )

    register_fsdp_module_mappings()
    fsdp_model = TorchFullyShardedDataParallel(
        unwrapped.config,
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        model,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    log_memory("after_fsdp_wrap", rank, device)

    warmup_ids = fixed_tokens(WARMUP_LEN, vocab_size, device)
    optimizer.zero_grad()
    warmup_logits = forward_logits(model, warmup_ids)
    warmup_loss = F.cross_entropy(
        warmup_logits[0, -WARMUP_LOSS_TOKENS - 1 : -1],
        warmup_ids[0, -WARMUP_LOSS_TOKENS:],
    )
    log_memory("warmup_before_backward", rank, device)
    warmup_loss.backward()
    fsdp_model.finish_grad_sync()
    log_memory("warmup_after_backward", rank, device)
    optimizer.step()
    optimizer.zero_grad()
    del warmup_ids, warmup_logits, warmup_loss
    torch.cuda.empty_cache()
    log_memory("after_warmup_optimizer_step", rank, device)

    input_ids = fixed_tokens(TOTAL_LEN, vocab_size, device)
    optimizer.zero_grad()
    for micro_step in range(ACCUM_STEPS):
        teacher_log_probs = torch.full(
            (COMPLETION_LEN, vocab_size),
            fill_value=-math.log(vocab_size),
            device=device,
            dtype=torch.bfloat16,
        )
        log_memory(f"micro={micro_step} after_teacher", rank, device)

        logits = forward_logits(model, input_ids)
        student_logits = logits[
            0,
            PROMPT_LEN - 1 : PROMPT_LEN + COMPLETION_LEN - 1,
            :,
        ]
        logger.info(
            f"rank={rank} micro={micro_step} logits_shape={tuple(logits.shape)} "
            f"dtype={logits.dtype} student_shape={tuple(student_logits.shape)}"
        )
        log_memory(f"micro={micro_step} after_student", rank, device)

        loss = compute_kl_loss(student_logits, teacher_log_probs)
        logger.info(f"rank={rank} micro={micro_step} loss={loss.item():.6f}")
        log_memory(f"micro={micro_step} before_backward", rank, device)

        is_final = micro_step == ACCUM_STEPS - 1
        sync_context = nullcontext() if is_final else fsdp_model.no_sync()
        with sync_context:
            (loss / ACCUM_STEPS).backward()
        log_memory(f"micro={micro_step} after_backward", rank, device)

    fsdp_model.finish_grad_sync()
    log_memory("after_fixed_backward", rank, device)
    logger.info(f"rank={rank} fixed repro completed without OOM")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
