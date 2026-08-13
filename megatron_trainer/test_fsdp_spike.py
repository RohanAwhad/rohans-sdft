"""FSDP spike for gpt-oss-20b — validate MCore native FSDP end-to-end.

Checks:
    1. TorchFullyShardedDataParallel wrap over a bridge-loaded MCore GPTModel
    2. torch AdamW over FSDP2 DTensor params (no bnb)
    3. no_sync()/finish_grad_sync() micro-step semantics
    4. clip_grad_norm_ on sharded grads
    5. HF-format weight export (DTensor -> full gather)
    6. Peak GPU memory per rank

Launch: torchrun --nproc_per_node=4 -m megatron_trainer.test_fsdp_spike
"""

import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    from loguru import logger

    from megatron_trainer.config import HF_MODEL_PATH
    from megatron_trainer.model_utils import cleanup, load_model
    from megatron.core.distributed import (
        DistributedDataParallelConfig,
        TorchFullyShardedDataParallel,
    )

    logger.info(f"=== FSDP spike start rank={rank}/{world_size} model={HF_MODEL_PATH} ===")

    t0 = time.monotonic()
    model = load_model(HF_MODEL_PATH)
    model.train()
    logger.info(
        f"model loaded in {time.monotonic() - t0:.1f}s "
        f"has_config={hasattr(model, 'config')} "
        f"mem={torch.cuda.memory_allocated(device) / 1e9:.1f}GB"
    )

    unwrapped = model.module if hasattr(model, "module") else model
    vocab_size = unwrapped.vocab_size
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    fsdp_model = TorchFullyShardedDataParallel(unwrapped.config, ddp_config, model)
    logger.info(f"FSDP wrap OK: {type(fsdp_model).__name__}")

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=5e-5)
    logger.info("AdamW created over FSDP params")

    # ---- Fake micro-steps (no_sync pattern like trainer.py) ----
    torch.manual_seed(0)
    seq_len = 512
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    NUM_MICRO = 3
    for i in range(NUM_MICRO):
        is_final = i == NUM_MICRO - 1
        ctx = nullcontext() if is_final else fsdp_model.no_sync()
        with ctx:
            logits = fsdp_model(
                input_ids=input_ids, position_ids=position_ids, attention_mask=None
            )
            loss = torch.nn.functional.cross_entropy(
                logits[0, -1:], input_ids[0, -1:]
            )
            (loss / NUM_MICRO).backward()
        logger.info(
            f"micro {i} done loss={loss.item():.4f} "
            f"mem={torch.cuda.memory_allocated(device) / 1e9:.1f}GB "
            f"peak={torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB"
        )

    if hasattr(fsdp_model, "finish_grad_sync"):
        fsdp_model.finish_grad_sync()
        logger.info("finish_grad_sync() OK")

    g0 = next(p.grad for p in fsdp_model.parameters() if p.grad is not None)
    logger.info(f"grad sample: dtype={g0.dtype} type={type(g0).__name__}")

    clip_grad_norm_(fsdp_model.parameters(), 10.0)
    logger.info("clip_grad_norm_ OK")
    optimizer.step()
    logger.info("optimizer step OK")

    # ---- Weight export / DTensor gather check (rank 0) ----
    if rank == 0:
        p = next(iter(fsdp_model.parameters()))
        try:
            full = p.full_tensor()
            logger.info(f"param.full_tensor() OK: {full.shape}")
            del full
        except Exception as e:
            logger.error(f"full_tensor failed: {type(e).__name__}: {e}")

        try:
            from megatron_trainer.model_utils import get_hf_weight_metadata

            names, dtypes, shapes = get_hf_weight_metadata(model)
            logger.info(
                f"export OK: {len(names)} tensors "
                f"first={names[0]} {shapes[0]} last={names[-1]} {shapes[-1]}"
            )
        except Exception as e:
            logger.error(f"export failed: {type(e).__name__}: {e}")

    # ---- Global peak memory ----
    peak = torch.tensor([torch.cuda.max_memory_allocated(device)], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        logger.info(f"GLOBAL peak memory: {peak.item() / 1e9:.2f} GB")

    dist.barrier()
    cleanup()
    logger.info("=== FSDP spike done ===")


if __name__ == "__main__":
    main()
