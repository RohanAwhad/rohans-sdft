"""Chunked SDFT and GRPO losses via Megatron-Core's output_processor hook.

Each rank computes its own full-vocab reverse-KL loss (its data is its own —
cross-rank vocab coupling is impossible with per-rank data slices). The LM
head is called ONCE per microstep on the completion hidden states (the
module's single-call path — probe-validated, no deadlock, FSDP handles the
head gradients natively), and the loss math is chunked over vocab columns with
an analytic backward.

Memory: the old path retained (C, V) fp32 log-softmax tensors through the
backward (~4.9 GB at C=6144) — the analytic backward retains only per-row
scalars, so the peak student-side memory drops from ~13 GB to ~10 GB, and the
crash-moment allocation (the fp32 grad) now has ~9 GB of headroom instead of
~1 GB.

The loss backward is analytic: grad_z = p * (A - K_row) * (1/C), A = log p - t,
K_row = the per-row KL (total derivative — a naive detached-denominator
softmax would leave a spurious +p error term). The 1/C is the row-mean, which
is inside this Function.
"""

import torch

KL_CHUNK = 2048
ROW_CHUNK = 128


def compute_is_weight(
    policy_logp: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    cap: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-sequence importance-sampling weight (TIS, TRL DistilTrainer style).

    w = mean over valid tokens of clamp(exp(policy_logp - rollout_logp), max=cap).

    - policy_logp: (C,) detached log-probs of the sampled tokens under the
      current policy (from ChunkedRowKL)
    - rollout_log_probs: (C,) log-probs of the sampled tokens under the
      rollout (vLLM) proposal; NaN marks never-sampled tokens (masked out)
    - cap: truncated importance sampling cap (upper bound on the ratio)

    Returns (weight, metrics). The weight is detached — no gradient flows
    through it; it only rescales the (already computed) KL loss.
    """
    valid = ~torch.isnan(rollout_log_probs)
    ratio = torch.exp(policy_logp - rollout_log_probs).clamp(max=cap)
    ratio_v = ratio[valid]
    if ratio_v.numel() == 0:
        weight = torch.ones((), dtype=torch.float32, device=policy_logp.device)
        metrics = {
            "is/ratio_mean": 1.0,
            "is/ratio_min": 1.0,
            "is/ratio_max": 1.0,
            "is/logp_diff_mean": 0.0,
            "is/clip_rate": 0.0,
        }
        return weight, metrics
    diff = (policy_logp - rollout_log_probs)[valid]
    metrics = {
        "is/ratio_mean": ratio_v.mean().item(),
        "is/ratio_min": ratio_v.min().item(),
        "is/ratio_max": ratio_v.max().item(),
        "is/logp_diff_mean": diff.abs().mean().item(),
        "is/clip_rate": (ratio_v >= cap).sum().item() / ratio_v.numel(),
    }
    return ratio_v.mean(), metrics


class ChunkedRowKL(torch.autograd.Function):
    """Chunked reverse-KL over one row chunk of the (gathered) head output.

    forward(z, teacher, token_ids) -> (kl_sum, policy_logp, critic_logp)
    - z: (R, V) bf16 head logits for R rows, gradient attached
    - teacher: (R, V) bf16 teacher log-probs for the same rows
    - kl_sum: scalar sum of per-row KL (the row-mean is applied outside)
    - policy_logp / critic_logp: (R,) detached per-token log-probs

    The forward mirrors the old compute_kl math exactly (log_softmax-style
    full-vocab logsumexp, single tree-reduced sum over V) so losses are
    bit-close; the backward is the analytic total derivative.
    """

    @staticmethod
    def forward(ctx, z, teacher, token_ids):
        R, V = z.shape
        zf = z.float()
        d = torch.logsumexp(zf, dim=-1)  # (R,) — same as log_softmax's denom
        logp = zf - d.unsqueeze(1)
        p = logp.exp()
        tc = teacher.float()
        kl_row = (p * (logp - tc)).sum(dim=-1)  # (R,)

        sel = torch.clamp(token_ids, 0, V - 1)
        row_idx = torch.arange(R, device=z.device)
        policy_logp = logp[row_idx, sel].detach()
        critic_logp = tc[row_idx, sel]

        ctx.save_for_backward(z, teacher, d, kl_row)
        return kl_row.sum(), policy_logp, critic_logp

    @staticmethod
    def backward(ctx, grad_kl, _grad_policy, _grad_critic):
        z, teacher, d, kl_row = ctx.saved_tensors
        zf = z.float()
        logp = zf - d.unsqueeze(1)
        p = logp.exp()
        A = logp - teacher.float()
        g = grad_kl * p * (A - kl_row.unsqueeze(1))
        return g, None, None


class ChunkedSelectedLogp(torch.autograd.Function):
    """Selected-token log-probs without retaining a full fp32 softmax graph."""

    @staticmethod
    def forward(ctx, z, token_ids):
        _, vocab_size = z.shape
        selected_ids = torch.clamp(token_ids, 0, vocab_size - 1)
        zf = z.float()
        denominator = torch.logsumexp(zf, dim=-1)
        row_ids = torch.arange(z.size(0), device=z.device)
        selected_logp = zf[row_ids, selected_ids] - denominator
        ctx.save_for_backward(z, selected_ids, denominator)
        return selected_logp

    @staticmethod
    def backward(ctx, grad_selected):
        z, selected_ids, denominator = ctx.saved_tensors
        probabilities = torch.exp(z.float() - denominator.unsqueeze(1))
        gradient = -probabilities * grad_selected.unsqueeze(1)
        row_ids = torch.arange(z.size(0), device=z.device)
        gradient[row_ids, selected_ids] += grad_selected
        return gradient, None


def make_kl_processor(
    prompt_len: int,
    completion_ids: list[int],
    teacher_log_probs: torch.Tensor,
    eos_token_id: int | None,
    device: torch.device,
    row_chunk: int = 128,
    rollout_log_probs: torch.Tensor | None = None,
    is_weighting: bool = True,
    is_cap: float = 2.0,
):
    """Build an MCore output_processor hook computing the chunked reverse-KL loss.

    Runs the LM head ONCE on the completion hidden states (the module's
    single-call path), computes the reverse-KL loss chunked over vocab columns
    with an analytic backward, and returns (loss, metrics_dict) — the model(...)
    call then returns that tuple.

    When rollout_log_probs is provided (and is_weighting), the per-sequence
    importance-sampling weight (see compute_is_weight) rescales the loss to
    correct for the rollout policy differing from the current policy.
    """
    C = teacher_log_probs.size(0)
    token_ids = torch.tensor(completion_ids, device=device, dtype=torch.long)

    def processor(
        hidden_states,
        output_layer,
        output_weight=None,
        labels=None,
        loss_mask=None,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
        runtime_gather_output=None,
        context=None,
        compute_language_model_loss=None,
        scale_logits=None,
        config=None,
    ):
        hidden_c = hidden_states[prompt_len - 1 : prompt_len + C - 1, 0]  # (C, H)
        z, _ = output_layer(hidden_c, weight=output_weight)  # (C, V) bf16

        kl_sum = torch.zeros((), dtype=torch.float32, device=z.device)
        policy_logp = torch.zeros(C, dtype=torch.float32, device=z.device)
        critic_logp = torch.zeros(C, dtype=torch.float32, device=z.device)
        for r in range(0, C, row_chunk):
            s, pol, crit = ChunkedRowKL.apply(
                z[r : r + row_chunk], teacher_log_probs[r : r + row_chunk],
                token_ids[r : r + row_chunk],
            )
            kl_sum = kl_sum + s
            policy_logp[r : r + row_chunk] = pol
            critic_logp[r : r + row_chunk] = crit
        loss = kl_sum / C

        if rollout_log_probs is not None and is_weighting:
            if rollout_log_probs.size(0) != C:
                raise ValueError(
                    f"rollout_log_probs length {rollout_log_probs.size(0)} != "
                    f"completion length {C}"
                )
            is_weight, is_metrics = compute_is_weight(
                policy_logp, rollout_log_probs, is_cap
            )
            loss = loss * is_weight

        with torch.no_grad():
            signal = critic_logp - policy_logp
            metrics = {
                "sdpo/signal_mean": signal.mean().item(),
                "sdpo/signal_std": signal.std().item(),
                "sdpo/len_signal_mean": signal.sum().item() / C,
                "sdpo/policy_logp": policy_logp.mean().item(),
                "sdpo/critic_logp": critic_logp.mean().item(),
            }
            if rollout_log_probs is not None and is_weighting:
                metrics.update(is_metrics)
            if eos_token_id is not None:
                eos_mask = token_ids == eos_token_id
                if eos_mask.any():
                    metrics["sdpo/eos_signal_mean"] = signal[eos_mask].mean().item()
                    metrics["sdpo/eos_logp_mean"] = policy_logp[eos_mask].mean().item()
                    metrics["sdpo/eos_logratio_mean"] = signal[eos_mask].mean().item()

        return loss, metrics

    return processor


def compute_grpo_loss_from_logps(
    policy_logp: torch.Tensor,
    old_logp: torch.Tensor,
    *,
    advantage: float,
    clip_low: float,
    clip_high: float,
    is_c_max: float,
    is_mode: str,
    ref_logp: torch.Tensor | None = None,
    kl_coef: float = 0.0,
    special_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DAPO per-token surrogate before group/token normalization.

    NaN old log-probs mark deterministic inserted tokens. Their old policy is
    the detached current policy, so their PPO ratio is one without dropping
    the formatting token from the policy-gradient objective.
    """
    if policy_logp.shape != old_logp.shape:
        raise ValueError(
            f"policy_logp shape {policy_logp.shape} != old_logp shape {old_logp.shape}"
        )
    valid_old = ~torch.isnan(old_logp)
    effective_old = torch.where(valid_old, old_logp, policy_logp.detach())
    log_ratio = policy_logp - effective_old
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - clip_low, max=1.0 + clip_high)
    advantage_t = torch.as_tensor(
        advantage, dtype=torch.float32, device=policy_logp.device,
    )
    unclipped = ratio * advantage_t
    clipped = clipped_ratio * advantage_t
    pg_loss = -torch.minimum(unclipped, clipped)
    clipped_region = unclipped > clipped

    valid_diff = log_ratio.detach()[valid_old]
    if valid_diff.numel() == 0:
        sequence_ratio = torch.ones(
            (), dtype=torch.float32, device=policy_logp.device,
        )
        sampling_logp_diff = torch.zeros_like(sequence_ratio)
    else:
        sequence_ratio = torch.exp(valid_diff.mean())
        sampling_logp_diff = valid_diff.abs().mean()
    if is_mode == "truncate":
        is_weight = sequence_ratio.clamp(max=is_c_max)
    elif is_mode == "mask":
        is_weight = torch.where(
            sequence_ratio > is_c_max,
            torch.zeros_like(sequence_ratio),
            sequence_ratio,
        )
    else:
        raise ValueError(f"Unknown GRPO IS mode: {is_mode!r}")

    per_token_loss = pg_loss
    if kl_coef > 0:
        if ref_logp is None:
            raise ValueError("GRPO KL requires reference token log-probs")
        if ref_logp.shape != policy_logp.shape:
            raise ValueError(
                f"ref_logp shape {ref_logp.shape} != policy_logp shape "
                f"{policy_logp.shape}"
            )
        ref_ratio_log = (ref_logp - policy_logp).clamp(min=-20.0, max=20.0)
        k3 = torch.exp(ref_ratio_log) - ref_ratio_log - 1.0
        k3_pp = k3 * ratio
        if special_token_mask is not None:
            k3_pp = k3_pp.masked_fill(special_token_mask, 0.0)
        per_token_loss = per_token_loss + kl_coef * k3_pp
        if special_token_mask is None:
            active_kl = k3_pp
        else:
            active_kl = k3_pp[~special_token_mask]
        mean_kl = (
            active_kl.mean()
            if active_kl.numel()
            else torch.zeros((), dtype=torch.float32, device=policy_logp.device)
        )
    else:
        mean_kl = torch.zeros((), dtype=torch.float32, device=policy_logp.device)

    per_token_loss = per_token_loss * is_weight
    metrics = {
        "grpo/clip_frac": clipped_region.float().mean().detach(),
        "grpo/entropy": (-policy_logp.mean()).detach(),
        "grpo/sampling_logp_diff": sampling_logp_diff.detach(),
        "grpo/is_ratio": sequence_ratio.detach(),
        "grpo/is_clip_frac": (sequence_ratio > is_c_max).float().detach(),
        "grpo/kl": mean_kl.detach(),
    }
    return per_token_loss.mean(), metrics


def make_grpo_processor(
    prompt_len: int,
    completion_ids: list[int],
    *,
    advantage: float,
    loss_scale: float,
    rollout_log_probs: torch.Tensor | None,
    old_logps_type: str,
    clip_low: float,
    clip_high: float,
    is_c_max: float,
    is_mode: str,
    reference_log_probs: torch.Tensor | None,
    kl_coef: float,
    special_token_ids: set[int],
    device: torch.device,
    row_chunk: int = ROW_CHUNK,
):
    """Build the one-head-call DAPO GRPO output processor for one rollout."""
    completion_count = len(completion_ids)
    token_ids = torch.tensor(completion_ids, device=device, dtype=torch.long)

    if old_logps_type == "vllm":
        if rollout_log_probs is None:
            raise ValueError("GRPO_OLD_LOGPS=vllm requires rollout token log-probs")
        if rollout_log_probs.size(0) != completion_count:
            raise ValueError(
                f"rollout_log_probs length {rollout_log_probs.size(0)} != "
                f"completion length {completion_count}"
            )
    elif old_logps_type != "detached":
        raise ValueError(f"Unknown GRPO old-logp source: {old_logps_type!r}")

    if reference_log_probs is not None and reference_log_probs.size(0) != completion_count:
        raise ValueError(
            f"reference_log_probs rows {reference_log_probs.size(0)} != "
            f"completion length {completion_count}"
        )

    def processor(
        hidden_states,
        output_layer,
        output_weight=None,
        labels=None,
        loss_mask=None,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
        runtime_gather_output=None,
        context=None,
        compute_language_model_loss=None,
        scale_logits=None,
        config=None,
    ):
        hidden_c = hidden_states[
            prompt_len - 1 : prompt_len + completion_count - 1, 0
        ]
        logits, _ = output_layer(hidden_c, weight=output_weight)

        policy_parts: list[torch.Tensor] = []
        ref_parts: list[torch.Tensor] = []
        for row in range(0, completion_count, row_chunk):
            row_logits = logits[row : row + row_chunk]
            row_ids = token_ids[row : row + row_chunk]
            policy_parts.append(ChunkedSelectedLogp.apply(row_logits, row_ids))
            if reference_log_probs is not None:
                ref_rows = reference_log_probs[row : row + row_chunk]
                ref_parts.append(
                    ref_rows.gather(1, row_ids.unsqueeze(1)).squeeze(1).float()
                )
        policy_logp = torch.cat(policy_parts)
        if old_logps_type == "detached":
            old_logp = policy_logp.detach()
        else:
            old_logp = rollout_log_probs
        ref_logp = torch.cat(ref_parts) if ref_parts else None
        special_mask = torch.zeros(
            completion_count, dtype=torch.bool, device=token_ids.device,
        )
        for special_id in special_token_ids:
            special_mask |= token_ids == special_id

        mean_loss, tensor_metrics = compute_grpo_loss_from_logps(
            policy_logp,
            old_logp,
            advantage=advantage,
            clip_low=clip_low,
            clip_high=clip_high,
            is_c_max=is_c_max,
            is_mode=is_mode,
            ref_logp=ref_logp,
            kl_coef=kl_coef,
            special_token_mask=special_mask,
        )
        loss = mean_loss * loss_scale
        metrics = {key: value.item() for key, value in tensor_metrics.items()}
        metrics.update(
            {
                "grpo/advantage": advantage,
                "grpo/policy_loss": mean_loss.detach().item(),
            }
        )
        return loss, metrics

    return processor
