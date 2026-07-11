"""NCCL communication protocol for trainer <-> logprob server.

Two-process group (torch.distributed):
    rank 0 = trainer   (GPU_TRAINER)
    rank 1 = server    (GPU_LOGPROB_SERVER)

Commands:
    CMD_TEACHER_LOGPROBS  — get full log-softmax at completion positions
    CMD_SYNC_WEIGHTS      — push trainer weights to server
    CMD_SHUTDOWN          — exit
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

CMD_TEACHER_LOGPROBS = 1.0
CMD_SYNC_WEIGHTS = 2.0
CMD_SHUTDOWN = -1.0


def init_nccl(rank: int, world_size: int = 2, master_port: int = 29500) -> None:
    """Initialize torch.distributed NCCL process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)  # cuda:0 after CUDA_VISIBLE_DEVICES filtering
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def send_command(cmd: float, device: torch.device) -> None:
    """Rank 0 broadcasts a command signal."""
    signal = torch.tensor([cmd], device=device)
    dist.broadcast(signal, src=0)


def recv_command(device: torch.device) -> float:
    """Rank 1 receives a command signal."""
    signal = torch.zeros(1, device=device)
    dist.broadcast(signal, src=0)
    return signal.item()


# ---------------------------------------------------------------------------
# CMD_TEACHER_LOGPROBS — full log-softmax transfer
# ---------------------------------------------------------------------------

def request_teacher_log_probs(
    token_ids: list[int],
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Rank 0 (trainer): send [cond_prompt + completion], receive teacher
    log-probs at completion positions.

    Returns: (completion_len, vocab_size) float32 tensor (detached).
    """
    send_command(CMD_TEACHER_LOGPROBS, device)

    seq_len = len(token_ids)
    completion_len = seq_len - prompt_len

    # 1. Send metadata
    meta = torch.tensor([seq_len, prompt_len], device=device, dtype=torch.long)
    dist.broadcast(meta, src=0)

    # 2. Send token IDs
    ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    dist.broadcast(ids, src=0)

    # 3. Receive teacher log-probs (server broadcasts from src=1)
    log_probs = torch.zeros(
        completion_len, vocab_size, device=device, dtype=torch.float32
    )
    dist.broadcast(log_probs, src=1)

    return log_probs


def handle_teacher_log_probs(model: torch.nn.Module, device: torch.device) -> None:
    """Rank 1 (server): receive sequence, compute log-softmax at completion
    positions, broadcast back."""
    # 1. Receive metadata
    meta = torch.tensor([0, 0], device=device, dtype=torch.long)
    dist.broadcast(meta, src=0)
    seq_len = int(meta[0].item())
    prompt_len = int(meta[1].item())
    completion_len = seq_len - prompt_len

    # 2. Receive token IDs
    ids = torch.zeros(seq_len, device=device, dtype=torch.long)
    dist.broadcast(ids, src=0)

    # 3. Forward pass
    with torch.no_grad():
        logits = model(ids.unsqueeze(0)).logits[0]  # (seq_len, V)

    # 4. Extract completion logits + log_softmax
    #    position prompt_len-1 predicts completion token 0
    completion_logits = logits[prompt_len - 1 : prompt_len + completion_len - 1, :]
    log_probs = F.log_softmax(completion_logits.float(), dim=-1)  # (C, V) float32

    # 5. Broadcast back from rank 1
    dist.broadcast(log_probs, src=1)


# ---------------------------------------------------------------------------
# CMD_SYNC_WEIGHTS
# ---------------------------------------------------------------------------

def broadcast_weights(model: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all model parameters from src rank.
    Must be called on ALL ranks simultaneously."""
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
