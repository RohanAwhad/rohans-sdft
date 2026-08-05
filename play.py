"""Parity test: old compute_kl (in-graph log_softmax) vs ChunkedRowKL.

Checks loss value AND exact gradients for single-rank semantics (each rank
computes its own full-vocab loss — there is no cross-rank coupling in this
design).
"""

import sys

import torch

sys.path.insert(0, "megatron_trainer")

from chunked_head import ChunkedRowKL, ROW_CHUNK


def compute_kl_old(student_logits, teacher_log_probs, completion_ids, eos_token_id, k=128):
    C = student_logits.size(0)
    device = student_logits.device
    token_ids = torch.tensor(completion_ids, device=device, dtype=torch.long)

    per_token_kl = torch.zeros(C, device=device, dtype=torch.float32)
    policy_logp = torch.zeros(C, device=device, dtype=torch.float32)
    critic_logp = torch.zeros(C, device=device, dtype=torch.float32)

    for i in range(0, C, k):
        j = min(i + k, C)
        s_chunk = student_logits[i:j].float()
        t_chunk = teacher_log_probs[i:j].float()

        s_log = torch.log_softmax(s_chunk, dim=-1)
        s_prob = s_log.exp()
        per_token_kl[i:j] = (s_prob * (s_log - t_chunk)).sum(dim=-1)

        chunk_ids = token_ids[i:j]
        idx = torch.arange(j - i, device=device)
        policy_logp[i:j] = s_log[idx, chunk_ids].detach()
        critic_logp[i:j] = t_chunk[idx, chunk_ids]

    loss = per_token_kl.mean()
    return loss, policy_logp, critic_logp


def run_case(name, z, teacher, ids, eos, row_chunk=ROW_CHUNK, tol=None):
    C = z.size(0)
    # Reference: old loss math on the SAME dtype (a bf16 input makes autograd
    # bf16-round the gradient at the .float() cast boundary — that rounding
    # sets the tolerance floor for bf16 cases).
    z_ref = z.clone().requires_grad_(True)
    z_new = z.clone().requires_grad_(True)

    loss_old, pol_old, crit_old = compute_kl_old(z_ref, teacher, ids, eos)
    grad_old = torch.autograd.grad(loss_old, z_ref)[0]

    token_ids = torch.tensor(ids, dtype=torch.long, device=z.device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=z.device)
    pol_new = torch.zeros(C, dtype=torch.float32, device=z.device)
    crit_new = torch.zeros(C, dtype=torch.float32, device=z.device)
    for r in range(0, C, row_chunk):
        s, p, c = ChunkedRowKL.apply(
            z_new[r : r + row_chunk], teacher[r : r + row_chunk],
            token_ids[r : r + row_chunk],
        )
        kl_sum = kl_sum + s
        pol_new[r : r + row_chunk] = p
        crit_new[r : r + row_chunk] = c
    loss_new = kl_sum / C
    grad_new = torch.autograd.grad(loss_new, z_new)[0]
    if tol is None:
        # bf16 inputs: the engine returns bf16-rounded grads for BOTH paths
        # (identical rounding), plus the analytic formula's fp32 summation
        # noise floor ~1.5e-3 at V=201088. fp32: pure fp32 noise.
        tol = 5e-3 if z.dtype == torch.bfloat16 else 1e-4

    loss_diff = (loss_old.item() - loss_new.item())
    grad_diff = (grad_old - grad_new).abs().max().item()
    grad_rel = grad_diff / grad_old.abs().max().item()
    pol_diff = (pol_old - pol_new).abs().max().item()
    crit_diff = (crit_old - crit_new).abs().max().item()

    ok = (
        abs(loss_diff) < 1e-4
        and grad_rel < tol
        and pol_diff < 1e-4
        and crit_diff < 1e-5
    )
    print(f"{name:55s} loss_diff={loss_diff:+.2e} grad_rel={grad_rel:.2e} "
          f"pol_diff={pol_diff:.2e} crit_diff={crit_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    eos = 7
    all_ok = True

    # 1. bf16, single chunk (V < KL_CHUNK)
    C, V = 128, 1000
    ids = torch.randint(0, V, (C,)).tolist()
    z = (torch.randn(C, V) * 3).bfloat16()
    teacher = torch.randn(C, V).bfloat16()
    all_ok &= run_case(f"bf16 single-chunk (C={C}, V={V})", z, teacher, ids, eos)

    # 2. bf16, multi-chunk (V >> KL_CHUNK)
    C, V = 512, 10000
    ids = torch.randint(0, V, (C,)).tolist()
    z = (torch.randn(C, V) * 3).bfloat16()
    teacher = torch.randn(C, V).bfloat16()
    all_ok &= run_case(f"bf16 multi-chunk (C={C}, V={V})", z, teacher, ids, eos)

    # 3. fp32, odd row-chunk to exercise boundaries (row_chunk=100)
    C, V = 777, 10000
    ids = torch.randint(0, V, (C,)).tolist()
    z = torch.randn(C, V) * 3
    teacher = torch.randn(C, V)
    all_ok &= run_case("fp32 row_chunk=100 (C=777, V=10000)", z, teacher, ids, eos,
                       row_chunk=100)

    # 4. production-ish shape: C=2048 (multi row-chunk), V=201088 (full vocab)
    C, V = 2048, 201088
    ids = torch.randint(0, V, (C,)).tolist()
    z = (torch.randn(C, V) * 3).bfloat16()
    teacher = torch.randn(C, V).bfloat16()
    all_ok &= run_case("bf16 production shape (C=2048, V=201088)",
                       z, teacher, ids, eos, tol=3e-3)

    print("\nALL PASS" if all_ok else "\nFAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
