"""KL divergence loss computation for SDFT."""

import torch
import torch.nn.functional as F


KL_CHUNK = 512  # tokens per chunk to avoid OOM on large models


def compute_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    completion_ids: list[int],
    eos_token_id: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Chunked reverse KL(student || teacher) averaged over token positions.

    Processes KL_CHUNK tokens at a time to avoid materializing full (C, V)
    intermediates. Gradient flows through slice assignment into per_token_kl.

    Args:
        student_logits:    (C, V) bfloat16, with gradient
        teacher_log_probs: (C, V) bfloat16, detached (log-softmax)
        completion_ids:    token IDs of the completion (for signal metrics)
        eos_token_id:      EOS token ID

    Returns:
        (loss, metrics_dict)
    """
    C = student_logits.size(0)
    device = student_logits.device
    token_ids = torch.tensor(completion_ids, device=device, dtype=torch.long)

    # Pre-allocate outputs
    per_token_kl = torch.zeros(C, device=device, dtype=torch.float32)
    policy_logp = torch.zeros(C, device=device, dtype=torch.float32)
    critic_logp = torch.zeros(C, device=device, dtype=torch.float32)

    for i in range(0, C, KL_CHUNK):
        j = min(i + KL_CHUNK, C)
        s_chunk = student_logits[i:j].float()       # (chunk, V) — has grad
        t_chunk = teacher_log_probs[i:j].float()     # (chunk, V) — detached

        s_log = F.log_softmax(s_chunk, dim=-1)
        s_prob = s_log.exp()
        # KL(p_s || p_t) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
        per_token_kl[i:j] = (s_prob * (s_log - t_chunk)).sum(dim=-1)

        # Signal metrics at generated tokens (detached)
        chunk_ids = token_ids[i:j]
        idx = torch.arange(j - i, device=device)
        policy_logp[i:j] = s_log[idx, chunk_ids].detach()
        critic_logp[i:j] = t_chunk[idx, chunk_ids]

    loss = per_token_kl.mean()

    # --- SDPO-style signal metrics ---
    with torch.no_grad():
        signal = critic_logp - policy_logp
        metrics = {
            "sdpo/signal_mean": signal.mean().item(),
            "sdpo/signal_std": signal.std().item(),
            "sdpo/len_signal_mean": signal.sum().item() / C,
            "sdpo/policy_logp": policy_logp.mean().item(),
            "sdpo/critic_logp": critic_logp.mean().item(),
        }

        if eos_token_id is not None:
            eos_mask = token_ids == eos_token_id
            if eos_mask.any():
                metrics["sdpo/eos_signal_mean"] = signal[eos_mask].mean().item()
                metrics["sdpo/eos_logp_mean"] = policy_logp[eos_mask].mean().item()
                metrics["sdpo/eos_logratio_mean"] = signal[eos_mask].mean().item()

    return loss, metrics
