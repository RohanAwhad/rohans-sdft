"""T1b: DDP vs FSDP forward-loss parity (Qwen3-0.6B, 4 ranks).

Same model weights + same input, one forward pass under each backend.
Assert |loss_fsdp - loss_ddp| < 1e-2 (catches wrong no_sync/accum or
wrap semantics in the FSDP branch — pure numeric equivalence check).

NOTE: Qwen3 ties embeddings. MCore's tied output layer forwards the
at-rest embedding DTensor into a native matmul under FSDP (mixed
Tensor/DTensor error), so this script patches
GPTModel.shared_embedding_or_output_weight to gather the shared weight
(what FSDP's own all-gather would do for an owned weight). Test-only patch.

Exit code 0 = parity holds.

Launch: bash megatron_trainer/test_ddp_fsdp_parity.sh
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def _patch_tied_embedding_weight() -> None:
    """Gather the tied embedding weight when FSDP shards it (test-only)."""
    import megatron.core.models.gpt.gpt_model as gpt_model_mod
    from torch.distributed.tensor import DTensor

    orig = gpt_model_mod.GPTModel.shared_embedding_or_output_weight

    def patched(self):
        w = orig(self)
        if isinstance(w, DTensor):
            return w.full_tensor()
        return w

    gpt_model_mod.GPTModel.shared_embedding_or_output_weight = patched


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    from loguru import logger

    from megatron_trainer.config import HF_MODEL_PATH
    from megatron_trainer.model_utils import cleanup, load_model
    from megatron.core.distributed import (
        DistributedDataParallelConfig,
        TorchFullyShardedDataParallel,
    )

    _patch_tied_embedding_weight()

    def run_forward(wrap_fn) -> torch.Tensor:
        model = load_model(HF_MODEL_PATH)
        model.train()
        wrapped = wrap_fn(model)
        unwrapped = model.module if hasattr(model, "module") else model
        vocab_size = unwrapped.vocab_size
        torch.manual_seed(0)
        with torch.no_grad():
            loss = F.cross_entropy(
                wrapped(input_ids=input_ids, position_ids=position_ids, attention_mask=None)[0, -1:],
                input_ids[0, -1:],
            ).float()
        dist.all_reduce(loss)
        return loss / dist.get_world_size()

    logger.info(f"=== T1b parity start rank={rank} model={HF_MODEL_PATH} ===")

    torch.manual_seed(0)
    input_ids = torch.randint(0, 1024, (1, 512), device=device)
    position_ids = torch.arange(512, device=device, dtype=torch.long).unsqueeze(0)

    t0 = time.monotonic()
    ddp_loss = run_forward(lambda m: DDP(m, device_ids=[local_rank]))
    logger.info(f"DDP fwd loss={ddp_loss.item():.6f} ({time.monotonic() - t0:.1f}s)")

    def fsdp_wrap(m: torch.nn.Module):
        unwrapped = m.module if hasattr(m, "module") else m
        return TorchFullyShardedDataParallel(
            unwrapped.config,
            DistributedDataParallelConfig(use_distributed_optimizer=False),
            m,
        )

    t0 = time.monotonic()
    fsdp_loss = run_forward(fsdp_wrap)
    logger.info(f"FSDP fwd loss={fsdp_loss.item():.6f} ({time.monotonic() - t0:.1f}s)")

    diff = abs(ddp_loss.item() - fsdp_loss.item())
    if rank == 0:
        if diff < 1e-2:
            logger.info(f"PARITY PASSED: diff={diff:.6f}")
        else:
            logger.error(f"PARITY FAILED: ddp={ddp_loss.item():.6f} fsdp={fsdp_loss.item():.6f} diff={diff:.6f}")
            raise SystemExit(1)

    cleanup()


if __name__ == "__main__":
    main()
