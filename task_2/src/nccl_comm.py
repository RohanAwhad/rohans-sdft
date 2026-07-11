"""NCCL communication utilities for weight transfer between processes."""

import os

import torch
import torch.distributed as dist


def init_nccl(rank: int, world_size: int = 2, master_port: int = 29500) -> None:
    """Initialize NCCL process group for intra-node GPU-to-GPU communication."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)  # cuda:0 after CUDA_VISIBLE_DEVICES filtering
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def broadcast_weights(model: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all model parameters from src rank to all other ranks.

    Must be called on ALL ranks simultaneously — NCCL broadcast is a collective op.
    On src rank: sends param.data.
    On other ranks: receives into param.data (in-place overwrite).
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def cleanup() -> None:
    """Destroy the NCCL process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
