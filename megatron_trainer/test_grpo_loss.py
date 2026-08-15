"""Unit tests for the GRPO loss math (docs/megatron_trainer/grpo.md
verification plan §1) — toy tensors, no MCore/GPU needed. Run with:

    python -m megatron_trainer.test_grpo_loss
"""

import torch

from megatron_trainer.chunked_head import grpo_loss_from_logp


def _toy_logp(n: int = 6, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (-torch.rand(n, generator=g) * 2).requires_grad_(True)  # negative log-probs


def test_gradient_identity_detached_matches_plain_pg():
    """GRPO_OLD_LOGPS=detached => ratio == 1 identically => the clipped
    surrogate collapses to plain policy gradient: grad = -A/N * d(logp)/d(theta).
    docs/megatron_trainer/grpo.md verification plan (b)."""
    logp = _toy_logp()
    C = logp.size(0)
    rollout_lp = logp.detach().clone()  # irrelevant in detached mode except for validity mask
    valid = torch.ones(C, dtype=torch.bool)
    advantage = 0.75
    group_total_tokens = C

    loss, _ = grpo_loss_from_logp(
        logp_theta=logp, rollout_log_probs=rollout_lp, valid=valid,
        advantage=advantage, group_total_tokens=group_total_tokens, gpg_rescale=1.0,
        old_logps_mode="detached",
    )
    loss.backward()
    grad_grpo = logp.grad.clone()

    logp2 = logp.detach().clone().requires_grad_(True)
    plain_pg = -(logp2.sum() / group_total_tokens) * advantage
    plain_pg.backward()
    grad_plain = logp2.grad

    max_diff = (grad_grpo - grad_plain).abs().max().item()
    assert max_diff < 1e-6, f"gradient-identity check failed: max|grad_grpo - grad_plainPG| = {max_diff}"
    print(f"[PASS] gradient-identity (detached old logps): max_diff={max_diff:.2e}")


def test_advantage_and_clip_engage_with_vllm_old_logps():
    """GRPO_OLD_LOGPS=vllm: ratio != 1 when old logp differs from current
    policy logp -> clip band + IS clamp should engage as documented.
    docs/megatron_trainer/grpo.md verification plan (a) + (c)."""
    logp = _toy_logp(seed=1)
    C = logp.size(0)
    # Old (rollout) logps deliberately far from current policy -> big ratio.
    # log_ratio = +2.0 everywhere -> ratio = e^2 ~ 7.39 (comfortably above both
    # the PG clip band [0.8, 1.28] and the IS clamp cap=3.0).
    rollout_lp = logp.detach() - 2.0
    valid = torch.ones(C, dtype=torch.bool)
    advantage = 1.0
    group_total_tokens = C

    loss, metrics = grpo_loss_from_logp(
        logp_theta=logp, rollout_log_probs=rollout_lp, valid=valid,
        advantage=advantage, group_total_tokens=group_total_tokens, gpg_rescale=1.0,
        clip_low=0.2, clip_high=0.28, is_c_max=3.0, old_logps_mode="vllm",
    )
    assert metrics["grpo/clip_frac"] == 1.0, "expected every token to be clip-engaged at ratio=e^2"
    assert metrics["is/clip_rate"] > 0.0, "expected the sequence-level TIS clamp to engage too"
    # ratio=e^2~7.39 clamped to cap=3.0 -> is_weight must equal the cap exactly
    assert abs(metrics["is/ratio_mean"] - 3.0) < 1e-5, metrics["is/ratio_mean"]
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()
    print(f"[PASS] clip/IS engagement: clip_frac={metrics['grpo/clip_frac']} is/ratio_mean={metrics['is/ratio_mean']:.4f}")


def test_degenerate_group_zero_advantage_gpg_rescale():
    """k-of-G binary rewards -> advantage = r_i - k/G. All-same reward
    (degenerate group) -> advantage 0 for every member; GPG rescale should be
    the mechanism that compensates (tested separately, at the caller level —
    here we just confirm advantage math and that zero-advantage gives zero
    loss/grad for that sample)."""
    rewards = [1.0, 1.0, 0.0, 1.0]  # k=3 of G=4
    mean_r = sum(rewards) / len(rewards)
    advantages = [r - mean_r for r in rewards]
    assert advantages == [0.25, 0.25, -0.75, 0.25]

    degenerate_rewards = [1.0, 1.0, 1.0, 1.0]  # all-pass
    mean_deg = sum(degenerate_rewards) / len(degenerate_rewards)
    adv_deg = [r - mean_deg for r in degenerate_rewards]
    assert all(a == 0.0 for a in adv_deg)

    logp = _toy_logp(seed=2)
    C = logp.size(0)
    rollout_lp = logp.detach().clone()
    valid = torch.ones(C, dtype=torch.bool)
    loss, _ = grpo_loss_from_logp(
        logp_theta=logp, rollout_log_probs=rollout_lp, valid=valid,
        advantage=0.0, group_total_tokens=C, gpg_rescale=1.0, old_logps_mode="detached",
    )
    assert loss.item() == 0.0, "zero advantage must produce exactly zero loss"
    print("[PASS] degenerate-group advantage math + zero-advantage => zero loss")


def test_mask_all_truncated_gives_zero_loss():
    """DAPO Overlong Filtering: a length-truncated completion is masked
    entirely, never punished (never a negative reward via the loss)."""
    logp = _toy_logp(seed=3)
    C = logp.size(0)
    rollout_lp = logp.detach().clone()
    valid = torch.zeros(C, dtype=torch.bool)  # mask_all upstream sets valid all-False
    loss, metrics = grpo_loss_from_logp(
        logp_theta=logp, rollout_log_probs=rollout_lp, valid=valid,
        advantage=-1.0, group_total_tokens=C, gpg_rescale=1.0,
        old_logps_mode="detached", mask_all=True,
    )
    assert loss.item() == 0.0
    assert metrics["grpo/masked"] == 1.0
    print("[PASS] mask_all truncated completion => zero loss")


if __name__ == "__main__":
    test_gradient_identity_detached_matches_plain_pg()
    test_advantage_and_clip_engage_with_vllm_old_logps()
    test_degenerate_group_zero_advantage_gpg_rescale()
    test_mask_all_truncated_gives_zero_loss()
    print("\nAll GRPO loss unit tests passed.")
