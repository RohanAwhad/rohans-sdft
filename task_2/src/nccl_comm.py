"""NCCL communication utilities for weight and data transfer between processes."""

import os

import torch
import torch.distributed as dist

# Command signals (broadcast from rank 0 to coordinate the two processes)
CMD_LOGPROBS = 1.0
CMD_SYNC_WEIGHTS = 2.0
CMD_SHUTDOWN = -1.0


def init_nccl(rank: int, world_size: int = 2, master_port: int = 29500) -> None:
    """Initialize NCCL process group for intra-node GPU-to-GPU communication."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)  # cuda:0 after CUDA_VISIBLE_DEVICES filtering
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def send_command(cmd: float, device: torch.device) -> None:
    """Broadcast a command signal from rank 0."""
    signal = torch.tensor([cmd], device=device)
    dist.broadcast(signal, src=0)


def recv_command(device: torch.device) -> float:
    """Receive a command signal (broadcast from rank 0). Called by rank 1."""
    signal = torch.zeros(1, device=device)
    dist.broadcast(signal, src=0)
    return signal.item()


def broadcast_weights(model: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all model parameters from src rank to all other ranks.

    Must be called on ALL ranks simultaneously — NCCL broadcast is a collective op.
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def cleanup() -> None:
    """Destroy the NCCL process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
