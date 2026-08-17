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
CMD_TEACHER_LOGPROBS_TOPK = 3.0
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

    # 3. Receive teacher log-probs (server broadcasts from src=1, bf16 to save memory)
    log_probs = torch.zeros(
        completion_len, vocab_size, device=device, dtype=torch.bfloat16
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
    log_probs = F.log_softmax(completion_logits.float(), dim=-1).bfloat16()  # (C, V) bf16

    # 5. Broadcast back from rank 1 (bf16 to save trainer memory)
    dist.broadcast(log_probs, src=1)


# ---------------------------------------------------------------------------
# CMD_SYNC_WEIGHTS
# ---------------------------------------------------------------------------

def broadcast_weights(model: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all model parameters from src rank.
    Must be called on ALL ranks simultaneously."""
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def broadcast_weights_ema(
    model: torch.nn.Module, alpha: float = 0.01, src: int = 0
) -> None:
    """Receive weights from src and EMA-blend into local model.

    phi = alpha * theta_received + (1 - alpha) * phi_local

    On the src rank, this just broadcasts (model is the student).
    On the dst rank, this receives into a buffer and blends.
    Must be called on ALL ranks simultaneously.
    """
    rank = dist.get_rank()
    for param in model.parameters():
        if rank == src:
            # Sender: broadcast own weights
            dist.broadcast(param.data, src=src)
        else:
            # Receiver: receive into buffer, EMA blend
            incoming = torch.empty_like(param.data)
            dist.broadcast(incoming, src=src)
            param.data.lerp_(incoming, alpha)


# ---------------------------------------------------------------------------
# CMD_TEACHER_LOGPROBS_TOPK — top-k log-softmax transfer
# ---------------------------------------------------------------------------


def request_teacher_log_probs_topk(
    token_ids: list[int],
    prompt_len: int,
    topk_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Rank 0 (trainer): send token IDs + top-k indices, receive teacher
    log-probs at those indices.

    Args:
        token_ids:    [cond_prompt + completion] token IDs.
        prompt_len:   length of condition prompt.
        topk_indices: (C, k) int64 tensor of top-k vocab IDs per position.
        device:       cuda device.

    Returns: (C, k) bfloat16 tensor of teacher log-probs (detached).
    """
    send_command(CMD_TEACHER_LOGPROBS_TOPK, device)

    seq_len = len(token_ids)
    completion_len = seq_len - prompt_len
    C, k = topk_indices.shape
    assert (
        topk_indices.device == device
    ), f"topk_indices on {topk_indices.device}, expected {device}"

    # 1. Send metadata [seq_len, prompt_len, k]
    meta = torch.tensor([seq_len, prompt_len, k], device=device, dtype=torch.long)
    dist.broadcast(meta, src=0)

    # 2. Send token IDs
    ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    dist.broadcast(ids, src=0)

    # 3. Send topk_indices (flattened to 1D)
    topk_flat = topk_indices.reshape(-1).contiguous()
    dist.broadcast(topk_flat, src=0)

    # 4. Receive teacher log-probs at top-k indices
    log_probs = torch.zeros(
        completion_len, k, device=device, dtype=torch.bfloat16
    )
    dist.broadcast(log_probs, src=1)

    return log_probs


def handle_teacher_log_probs_topk(
    model: torch.nn.Module, device: torch.device
) -> None:
    """Rank 1 (server): receive sequence + top-k indices, compute log-softmax
    at those positions, broadcast back."""
    # 1. Receive metadata [seq_len, prompt_len, k]
    meta = torch.tensor([0, 0, 0], device=device, dtype=torch.long)
    dist.broadcast(meta, src=0)
    seq_len = int(meta[0].item())
    prompt_len = int(meta[1].item())
    k = int(meta[2].item())
    completion_len = seq_len - prompt_len

    # 2. Receive token IDs
    ids = torch.zeros(seq_len, device=device, dtype=torch.long)
    dist.broadcast(ids, src=0)

    # 3. Receive topk_indices (flattened), reshape to (C, k)
    topk_flat = torch.zeros(completion_len * k, device=device, dtype=torch.long)
    dist.broadcast(topk_flat, src=0)
    topk_indices = topk_flat.reshape(completion_len, k)

    # 4. Forward pass
    with torch.no_grad():
        logits = model(ids.unsqueeze(0)).logits[0]  # (seq_len, V)

    # 5. Extract completion logits, compute log-softmax via logsumexp trick
    completion_logits = logits[
        prompt_len - 1 : prompt_len + completion_len - 1, :
    ].float()  # (C, V)
    log_z = completion_logits.logsumexp(dim=-1, keepdim=True)  # (C, 1)
    log_probs = (completion_logits.gather(1, topk_indices) - log_z).bfloat16()  # (C, k) bf16

    # 6. Broadcast back from rank 1
    dist.broadcast(log_probs, src=1)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
