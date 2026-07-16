"""NCCL communication protocol for trainer <-> logprob server.

Two-process group (torch.distributed):
    rank 0 = trainer   (GPU_TRAINER)
    rank 1 = server    (GPU_LOGPROB_SERVER)

Commands:
    CMD_TEACHER_LOGPROBS  — get full log-softmax at completion positions
    CMD_SYNC_WEIGHTS      — push trainer weights to server
    CMD_SHUTDOWN          — exit

Identical protocol to the HF reference implementation. Works with any model
that exposes .parameters() — Megatron-Core GPTModel and HF models both do.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

CMD_TEACHER_LOGPROBS = 1.0
CMD_SYNC_WEIGHTS = 2.0
CMD_SHUTDOWN = -1.0


def send_command(cmd: float, device: torch.device) -> None:
    signal = torch.tensor([cmd], device=device)
    dist.broadcast(signal, src=0)


def recv_command(device: torch.device) -> float:
    signal = torch.zeros(1, device=device)
    dist.broadcast(signal, src=0)
    return signal.item()


# ---------------------------------------------------------------------------
# CMD_TEACHER_LOGPROBS
# ---------------------------------------------------------------------------

def request_teacher_log_probs(
    token_ids: list[int],
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Rank 0 (trainer): send sequence, receive teacher log-probs."""
    send_command(CMD_TEACHER_LOGPROBS, device)

    seq_len = len(token_ids)
    completion_len = seq_len - prompt_len

    meta = torch.tensor([seq_len, prompt_len], device=device, dtype=torch.long)
    dist.broadcast(meta, src=0)

    ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    dist.broadcast(ids, src=0)

    log_probs = torch.zeros(
        completion_len, vocab_size, device=device, dtype=torch.bfloat16
    )
    dist.broadcast(log_probs, src=1)

    return log_probs


def handle_teacher_log_probs(model: torch.nn.Module, device: torch.device) -> None:
    """Rank 1 (server): compute teacher log-probs and broadcast back.

    Works with MCore GPTModel: calls model(input_ids, position_ids, attention_mask=None)
    which returns (1, S, V) logits when labels=None.
    """
    meta = torch.tensor([0, 0], device=device, dtype=torch.long)
    dist.broadcast(meta, src=0)
    seq_len = int(meta[0].item())
    prompt_len = int(meta[1].item())
    completion_len = seq_len - prompt_len

    ids = torch.zeros(seq_len, device=device, dtype=torch.long)
    dist.broadcast(ids, src=0)

    with torch.no_grad():
        input_ids = ids.unsqueeze(0)
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits = logits[0]  # (S, V)

    completion_logits = logits[prompt_len - 1 : prompt_len + completion_len - 1, :]
    log_probs = F.log_softmax(completion_logits.float(), dim=-1).bfloat16()

    dist.broadcast(log_probs, src=1)


# ---------------------------------------------------------------------------
# CMD_SYNC_WEIGHTS (EMA)
# ---------------------------------------------------------------------------

def broadcast_weights_ema(
    model: torch.nn.Module, alpha: float = 0.01, src: int = 0
) -> None:
    """EMA weight sync: phi = alpha * theta_received + (1 - alpha) * phi_local.

    On the src rank, broadcasts own weights.
    On the dst rank, receives and EMA-blends.
    Works with any model exposing .parameters().
    """
    rank = dist.get_rank()
    for param in model.parameters():
        if rank == src:
            dist.broadcast(param.data, src=src)
        else:
            incoming = torch.empty_like(param.data)
            dist.broadcast(incoming, src=src)
            param.data.lerp_(incoming, alpha)
