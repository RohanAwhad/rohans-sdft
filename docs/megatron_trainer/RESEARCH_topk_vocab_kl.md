# Top-K Vocab KL for SDFT — Deep Research Report

> Generated: 2026-08-12 | Sources: 12 web + 6 code
> Topic: issue #5 — "KL on Top-K vocab ids": does restricting reverse KL to top-K vocab ids change convergence speed/final score vs full-vocab KL in self-distillation fine-tuning?

## TL;DR

- The issue comment's premise ("DeepSeek report instability") is **inverted**: DeepSeek-V4 (arXiv:2606.19348 §5.1.2) rejects *token-level* (sampled-token) KL as unstable and adopts **full-vocab** logit distillation as the *stable* fix. The paper contains **no top-K-vocab KL claim at all**.
- The Tinker Cookbook's "Top-K=20 matches full-vocab KL (68.04% vs 68.04%)" is real but narrow: **forward KL** CE, one task (tooluse), one model (Qwen2.5-7B), motivated by an API limitation (no full-vocab logits from Tinker).
- The SDFT paper's loss claims are unreliable as a guide: the paper says reverse KL + full-vocab analytic estimator, but the authors later admitted (GitHub issue #5, then arXiv v2) that all published results used **per-token forward KL**.
- The exact A/B the issue asks for — renormalized top-K reverse KL vs full-vocab reverse KL, on-policy, ~8B — **does not exist in the literature**. Closest direct evidence: teacher-top-K renormalized reverse KL beats sampled-token OPD (+19.8% math, 2603.25562); shared top-16 token sets carry 97–99% of probability mass (2604.13016). Neither paper trained a full-vocab baseline.
- **Math verdict**: of 4 derived top-K reverse-KL variants, only the *renormalized over the subset* variant (teacher top-K) preserves our analytic `p·(A − const)` backward intact with `const = KL_b`, zero gradient outside the support, and backward working from gathered `(R,K)` tensors only — a real memory win for `ChunkedRowKL`. Truncated-unrenormalized variants keep full-vocab memory *and* reintroduce denominator-coupling bias (the "spurious +p" error class our backward was built to avoid).
- **Recommendation**: no literature basis to claim top-K KL is "insignificant" for reverse-KL on-policy SDFT. Implement the renormalized teacher-top-K variant behind a flag (`KL_TOP_K`, 0 = full vocab) and run the head-to-head ourselves at K≈32 with a small K sweep. If we want savings, the same flag should eventually gate the logprob server to ship `(C,K)` teacher tensors instead of `(C,V)`.

## Overview

The trainer computes a full-vocab reverse-KL loss per completion row via `ChunkedRowKL` (megatron_trainer/chunked_head.py:84-110): `logp = z − logsumexp(z)`, `kl_row = Σ_v p_v(logp_v − t_v)`, with an analytic backward `g = grad_kl · p · (A − kl_row)` — the total derivative; a naive detached-denominator softmax would leave a spurious `+p` term.

Issue #5 asks whether restricting this KL to the top-K vocab ids (which distribution's top-K? renormalized or not?) would be equivalent for finetuning. "Top-K KL" is **not one algorithm**: two renormalization semantics exist in the wild, and 4+ variants once you fix the selection distribution (teacher vs student top-K). The report derives all variants, audits the evidence for each, and states what our code would need.

## Key Findings

### 1. The "DeepSeek instability" citation is inverted (and about a different axis)

DeepSeek-V4 (arXiv:2606.19348, §5.1.2) uses multi-teacher on-policy reverse-KL distillation:

> "In handling the above OPD objective, prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate at each token position ... Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability. **Therefore, we adopt full-vocabulary logit distillation in our OPD.** Preserving the complete logit distribution in calculating reverse KL loss yields more stable gradient estimates and ensures faithful distillation of the teachers' knowledge."

- The paper's dichotomy is **sampled-token KL vs full-vocab KL** — not top-K-vocab.
- Every "top-k" occurrence in the paper refers to CSA sparse-attention KV selection, not vocab/KL.
- Their remedy for full-vocab cost at |V|>100k is engineering (teacher hidden-state caching + TileLang kernel), not truncation.
- DeepSeek-V3.2 (arXiv:2512.02556) is the actual source for KL *estimator* instability (biased k3 estimator in GRPO) — again sampled-token axis, not top-K.

### 2. Tinker Cookbook: the one real "top-K ≈ full-vocab" number — but forward KL, API-motivated

- Tinker's SDFT recipe (`tinker_cookbook/distillation/sdft.py`) uses teacher top-K=20 **forward-KL cross-entropy** (renormalized via `logprobs -= torch.logsumexp(logprobs, dim=0)`, sdft.py:426-428) because "The Tinker API does not expose full-vocabulary logits."
- Validation (README): "Top-K=20 matches full-vocab KL (68.04% vs 68.04% with EMA, 69.07% vs 67.01% without)" on tooluse, Qwen2.5-7B; independent rerun on Qwen3-4B: 56.70% = 56.70%.
- This is **forward** KL. Forward KL truncation is benign by construction: the teacher entropy term is constant w.r.t. the student, so dropping non-top-K terms only truncates a cross-entropy sum. **Reverse KL truncation is a different object** (see §4) — the equivalence does not transfer.

### 3. The SDFT paper's own loss story is muddled

- Paper (arXiv:2601.19897 v2, Eq 1): reverse KL `L(θ) = D_KL(π_θ ∥ π(·|c))`; Eq 2 + App. A.1: full-vocab analytic per-token gradient estimator, claimed "most stable optimization and best downstream performance" vs token-level estimator ("higher variance and weaker KL control") — but **the ablation is qualitative, zero numbers/tables**.
- v2 added: "Although the theory points to Reverse KL as a suitable loss, we found in practice that Forward KL yields the best performance."
- Author admission (GitHub idanshen/Self-Distillation issue #5, 2026-04-07): "all the results in our paper were produced using on-policy sampling, but **per-token forward KL loss** (similar to the GKD paper)."
- Repo `distil_trainer.py:1656-1676`: `alpha=0` default = forward KL, full-vocab `log_softmax`, `kl_loss.sum(-1)`. No top-K anywhere in paper or repo.
- **Implication**: "the SDFT paper proves full-vocab reverse KL is needed" is not supported — the paper's results came from full-vocab *forward* KL, which is exactly the loss that truncates benignly.

### 4. The four top-K reverse-KL variants — derivation (verified by hand + finite differences)

Notation per row: student logits `z`, teacher log-probs `t = log q`, `d = logsumexp(z)`, `logp = z − d`, `p = exp(logp)`, `A = logp − t`. `S = TopK_q` (teacher support), `P_S = Σ_{v∈S} p_v`, `Q_S = Σ_{v∈S} q_v`. Full-vocab gradient: `g_u = p_u(A_u − KL_row)`.

**Input**: `grad_kl` = upstream `1/C` row-mean gradient.

| Variant | Loss | Backward | Cost |
|---|---|---|---|
| Full (current) | `Σ_v p_v A_v` | `grad_kl · p_u(A_u − KL_row)` | full-V logsumexp; backward needs full (R,V) `p` |
| (a) Truncated, teacher-K, **no renorm** | `Σ_{v∈S} p_v A_v` | `u∈S: grad_kl·p_u[A_u − (L_a + P_S − 1)]`; `u∉S: grad_kl·(−p_u(L_a + P_S))` | full-V logsumexp still needed; full (R,V) `p`; out-of-set logits get a **spurious** mass-coupling pull |
| **(b) Renormalized, teacher-K** | `KL(softmax_S(z) ∥ q̃)`, `q̃ = q/Q_S` | `u∈S: grad_kl · p̃_u(b_u − KL_b)` where `b = z − t`, `KL_b = Σ_{v∈S} p̃_v b_v − log Σ_S e^z + log Q_S`; `u∉S: 0` | **no full-V logsumexp** (the d terms cancel); backward from **(R,K) only** |
| (c) Truncated, student-K | same as (a) with `S = TopK_z` frozen per step | fixed-set (a); a.e. subgradient of a nondifferentiable objective; support can drift (mass-shift degeneracy) | full-V; full (R,V) `p`; topk pass |
| (d) Renormalized, student-K | same as (b), `S = TopK_z` frozen | same as (b); zero gradient at the K-boundary → no pressure maintaining the split | K-only; topk pass |
| (e) Ghost-token renormalization | `Σ_{v∈S} p_v A_v + (1−P_S)log[(1−P_S)/(1−Q_S)]` | `grad_kl · p_u(A′_u − L′)` over ALL u — exact `p·(A − const)` shape, single per-row constant | full-V; full (R,V) `p`; mathematically clean, **no memory savings** |
| EMA-PG (head+tail) | unrenorm head (a) + `s·1[y∉S]·(p_y/sg(p_y))·sg(A_y)` | **unbiased** estimator of full reverse-KL gradient (OPD App. G.3.1, Zhang & Ba EMA-PG) | full-V + sampled token |

Cross-checks: (b) is exactly OPD's LSM objective (arXiv:2603.25562 Eq 7–8: `π̂ = π/Σ_S π`, `q̂ = q/Σ_S q`), which the paper calls essential — "removing it leads to rapid collapse" (Table 3). (a)'s spurious tail pull is the reverse-KL analog of the Sparse Logit Sampling forward-KL bias (arXiv:2503.16870 Eq 2: `∂L/∂x_i = (Σ_{j∈K} t_j)·p_i − t_i`). Megatron-LM's `topk_kl_div` and verl's `compute_forward_kl_topk` both keep full-vocab normalization + optional ghost token (forward KL direction).

### 5. Evidence audit — the exact A/B we want is untested

| Paper | Setting actually tested | Transfer |
|---|---|---|
| 2604.13016 (Rethinking OPD) | On-policy reverse-KL OPD, top-k renorm (student support, k=1/4/16/64) vs **sampled-token**; 1.5B–7B | **Direct** family, but never trains full-vocab baseline. k≥4–16 ≈ sampled-token; Top-1 fails (mode-flipping). Shared top-16 sets carry 97–99% of mass (App. B.1). |
| 2603.25562 (Revisiting OPD) | On-policy reverse-KL, teacher top-32 renorm vs **sampled-token** OPD; 7B; math/agentic | **Direct** family, no full-vocab baseline. +19.8% multi-task math avg. Own caveat (App. A): "Relative to full-vocabulary reverse-KL, this introduces bias... a property of the estimator, rather than a settled benefit or drawback." |
| 2606.19348 (DeepSeek-V4) | Full-vocab multi-teacher reverse-KL OPD at 1.6T MoE; token-level rejected | Supports full-vocab being trainable at scale. No top-K claim. |
| 2608.03796 (Efficient KD) | Offline **forward**-KL KD, top-100 unrenorm vs online full dense; 8B→3.2B; loss-curves only | Indirect. "Near-identical training loss", 29% faster/iter, ~40% throughput — speedup is from offline caching, not top-K per se. No downstream eval, no K sweep, no renormalization. |
| 2503.16870 (Sparse Logit Sampling, ACL'25 Oral) | Offline pretraining KD, **forward** KL; bias theory; 300M–3B | Indirect but load-bearing: naive top-K caching is provably biased (student converges to up-scaled teacher `p_i = t_i/M`); renormalization does NOT remove the bias (App. A.3); calibration degrades at small K. Proof is forward-KL-specific; no reverse-KL theorem exists. |
| 2406.13555 (BiLD) | Task-specific offline distillation; BiLD is KL over top-8 **pairwise logit differences**, not plain top-K KL | Not transferable. Beats full KL on small classification-heavy tasks; authors admit tail knowledge loss. |

**Net**: literature supports (i) top-K reverse KL ≫ sampled-token OPD, (ii) small K (~16–32) covers most teacher mass in on-policy reverse-KL, (iii) full-vocab reverse KL is stable and is the largest-scale practitioner's choice. It does **not** support the claim that top-K reverse KL equals full-vocab reverse KL downstream — that comparison is absent everywhere, and the only rigorous bias analysis (forward-KL-specific) says truncation is never free.

### 6. What our codebase would need (and what it would save)

- **Loss side** (chunked_head.py): variant (b) drops the full-V logsumexp from the loss path and the backward's full (R,V) fp32 recompute (`logp`, `p` — the module docstring's cited ~7.5 GB transient at C=6144, V=151936) to ~50 MB of (R,K) tensors at K=2048. The head GEMM `(C,H)@(H,V)` and the (C,V) bf16 `z` are untouched either way.
- **`policy_logp` for IS weighting must stay full-vocab normalized** (`compute_is_weight` semantics): keep a detached `d` row-reduction for the sampled completion token only, or augment `S` with the sampled token (OPD Variant 3).
- **Teacher transfer** (logprob_server.py:162-186 + logprob_client.py:63): today the server ships full (C,V) teacher log-probs over TCP (~1.87 GB bf16 at C=6144). The real end-to-end win comes from a server-side top-K mode shipping `(C,K)` ids+logps+`Q_S` — separate change, keep the full tensor until the loss-side flag is validated.
- **Config**: `KL_TOP_K = int(os.environ.get("KL_TOP_K", "0"))` (0 = off), thread through `make_kl_processor` (chunked_head.py:113) and trainer.py:402-411.
- K choice: literature uses K=16/32/64 (on-policy) and 100 (offline); a fixed 32 or teacher-top-p (Σ_S q ≥ 0.95) adaptive support are both defensible; sweep {16, 32, 64} in the A/B run.

## Practical Guide

Proposed A/B experiment (on-policy, Qwen3-8B, same as current runs):

1. Implement variant (b) behind `KL_TOP_K` flag; default 0 preserves current behavior bit-for-bit (parity test: `KL_TOP_K=0` run == current loss).
2. Full-vocab vs top-K∈{16, 32, 64} on the same seed/data/rollout budget (vLLM gumbel nondeterminism note: fix seeds or compare across K in one run family).
3. Metrics: step-time (loss-side O(R·V)→O(R·K) should show up in TIMING), training loss trajectory (note: truncated KL loss values are not comparable across K — compare on eval/downstream), signal metrics (sdpo/*), calibration check (IS `is/ratio_mean`, plus ECE if cheap), downstream eval accuracy vs steps.
4. Only after the loss-side result: switch the logprob server to ship top-K teacher tensors, re-validate end-to-end, measure NCCL time drop.

## Gotchas & Pitfalls

1. **Truncated ≠ renormalized** — (a)/(c) look innocuous but keep full backward memory AND add a spurious `−p_u(L_a + P_S)` pull on out-of-support logits (the exact error class the analytic backward was written to avoid). Do not implement the unrenormalized form.
2. **Student-top-K (d) has a support-drift pathology** — zero gradient at the K-boundary means nothing maintains the split; OPD reports student-top-K variants degrade in multi-task. Prefer teacher-top-K.
3. **Loss values across K are not comparable** — truncated/renormalized KL is a different scalar; only compare downstream metrics.
4. **IS-weighting coupling** — `policy_logp` feeds `compute_is_weight` (chunked_head.py:28-67) with full-vocab log-probs of the sampled token; a naive top-K logp would silently bias the TIS weight.
5. **Forward-KL evidence does not transfer to reverse KL** — Tinker's 68.04% = 68.04% and 2608.03796's loss curves are forward-KL results; forward-KL truncation is benign by algebra, reverse-KL truncation is not (tail carries gradient mass through `p_v` in `Σ_v p_v A_v` differently).
6. **The DeepSeek-V4 citation in issue #5 is backwards** — do not cite it as "full-vocab KL is unstable." It says the opposite: sampled-token KL is unstable, full-vocab is the fix. Any top-K change should be justified by memory/efficiency, not by an instability claim.
7. **SDFT paper's "most stable estimator" claim is qualitative** — no tables/numbers in App. A.1, and the results it reports came from forward KL anyway (author admission).

## Hardware / Environment Considerations

- V = 151,936 (Qwen3-8B; padded vocab at runtime may exceed this — trainer.py:118-119).
- Current per-microstep teacher tensor ≈1.87 GB bf16 (C=6144); recv buffer C×V×2 bytes (logprob_client.py:63).
- Loss-side fp32 transients are the module's stated pressure point; variant (b) removes them without touching the head GEMM.
- MCore torch-FSDP: loss change is per-rank, no collective implications; the single-call output_processor path (trainer.py:402-417) must remain single-call (devlog: deadlock at multiple calls).

## Sources

1. DeepSeek-V4 — arXiv:2606.19348, §5.1.2, §5.2.2 — https://arxiv.org/abs/2606.19348, https://arxiv.org/html/2606.19348v1
2. SDFT (Self-Distillation Enables Continual Learning) — arXiv:2601.19897 v2, Eq 1-2, App. A.1, §3 — https://arxiv.org/html/2601.19897v2
3. SDFT official repo — https://github.com/idanshen/Self-Distillation — distil_trainer.py:801, 1656-1676; issue #5 (author admission) comment 4201917481
4. Tinker Cookbook SDFT — https://github.com/thinking-machines-lab/tinker-cookbook — tinker_cookbook/distillation/sdft.py:9-14, 306-316, 426-428, 497, 583-584, 653-659; recipes/sdft/README.md
5. Revisiting On-Policy Distillation — arXiv:2603.25562 v2, Eq 4-8, Table 3-4, App. A, App. G.3.1 — https://arxiv.org/html/2603.25562v2
6. Rethinking On-Policy Distillation — arXiv:2604.13016 v2, Eq 4-5, App. B.1, §6 — https://arxiv.org/html/2604.13016v2
7. Efficient KD (Offline Top-K Logits) — arXiv:2608.03796 v1 — https://arxiv.org/html/2608.03796v1
8. Sparse Logit Sampling (ACL 2025 Oral) — arXiv:2503.16870 v2, Eq 2, App. A.3-A.7 — https://arxiv.org/html/2503.16870v2
9. BiLD — arXiv:2406.13555 v3 — https://arxiv.org/html/2406.13555v3
10. Megatron-LM topk_kl_div — https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/distillation/cached_logits_loss.py (topk_kl_div, ghost token)
11. verl forward_kl_topk — https://github.com/volcengine/verl/blob/main/verl/trainer/distillation/fsdp/losses.py (compute_forward_kl_topk)
12. TRL experimental SDFT — trl/experimental/sdft/sdft_trainer.py (topk_logits mode, distillation_add_tail)
13. This repo — megatron_trainer/chunked_head.py:28-67, 84-110, 113-203; trainer.py:118-119, 132-140, 402-417; logprob_server.py:162-186; logprob_client.py:63; config.py:72-73
