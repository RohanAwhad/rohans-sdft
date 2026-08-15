# GRPO and Its Variants — Deep Research Report

> Generated: 2026-08-15 | Sources: 18 papers (arXiv, full-text) + 4 framework codebases (TRL, verl, OpenRLHF, prime-rl) + community reports (GitHub issues/PRs, HN, practitioner blogs)
> Tags: #grpo #rlvr #reinforcement-learning #sdft #design
> Question: which GRPO variant is research-proven best, for a binary-reward, long-completion (4k–8k tokens), small-group (G=4–16), 8B-scale reasoning-RL setting — and how to design it into the SDFT Megatron trainer.

## TL;DR

- **There is no single "best GRPO variant" — the evidence converges on a composite**: Dr. GRPO's mean-only advantage (`r_i − mean(R_group)`, no std division) + DAPO's token-level loss aggregation + DAPO-style dynamic sampling for degenerate groups + clip-higher (ε 0.2/0.28) left at defaults. This is one config line in TRL (`loss_type="dapo"`, `scale_rewards="none"`) and verl (`norm_adv_by_std_in_grpo=false`, `loss_agg_mode="token-mean"`).
- **Std-normalized z-score advantage is the worst-documented choice for binary rewards**: it is unbounded (low-variance amplification), carries an exact question-difficulty bias (GRPO ascends `2·arcsin√p`, Dr. GRPO ascends `p`), and three papers (Dr. GRPO, MDP-GRPO, "three operations on one number") independently identify it as the pathology source.
- **Clipping is provably inert at 1 gradient iteration** (verified numerically in TRL PR #6681: `max|grad_GRPO − grad_plainPG| = 0.0`). Every paper's clip numbers are about the multi-iteration regime. Don't spend design effort on clip variants.
- **Reference-free GRPO (β=0) is the modern default** for verifiable/binary rewards (DAPO, Dr.GRPO, Open-Reasoner-Zero, GRPO-Zero, TRL default). The only controlled evidence that a KL anchor *helps* is EMA-PG (2026): use the **EMA of the policy as anchor** with small β — which maps ~free onto this project's existing EMA teacher logprob server.
- **"Log-signal GRPO" does not exist** in any accessible source (arXiv, GitHub, TRL/verl/OpenRLHF source). Verified exhaustively; closest real things: GSPO's sequence-level log-ratio averaging, kalomaze's power-scaled rewards. Documented as eliminated (see Eliminated Options).

## Overview

GRPO (DeepSeekMath, arXiv:2402.03300) replaced PPO's value network with a group-relative baseline: sample G responses per prompt, define advantage `Â_i = (r_i − mean(R))/std(R)`, and optimize a clipped surrogate with per-token importance ratios plus a KL term to a reference. It became the default RLVR algorithm (DeepSeek-R1, Qwen3, DAPO, DeepScaleR). 2025–2026 produced a large variant family that mostly *removes or reweights parts of the original objective*: DAPO (ByteDance) fixes four engineering pathologies, Dr. GRPO (Sea AI Lab) removes two statistical biases, GSPO (Qwen) replaces token ratios with sequence ratios, and a series of theory papers (impossibility theorem, "one dial" analysis) now explain *why* the original had problems.

For the SDFT trainer the relevant regime is: binary pass/fail verdict rewards (already computed by the envs), long completions where length itself is the hack dimension, small group sizes forced by compute, one forward/backward per rollout, and an existing EMA-teacher logprob server that can double as a KL anchor. This report compiles the evidence per decision axis and ends with a concrete recommended configuration and its evidence gaps.

## Key Findings

### 1. Advantage: mean-only (Dr. GRPO) beats z-score for binary rewards

Exact identities (arXiv:2607.00152, "GRPO, Dr. GRPO, DAPO Are Three Operations on One Number"): for binary rewards with k correct of G, `σ = √(k(G−k))/G`; the per-prompt update is `g = σ·(s̄₊ − s̄₋)` regardless of baseline — **the update size is the group's disagreement**. The three "competing" variants are the same dial:

| Method | Operation on σ | Implicit objective |
|---|---|---|
| GRPO | divide by σ | ascends `2·arcsin(√p)` — difficulty bias: 5%-or-95% prompts get ~2.3× the weight of coin-flip prompts |
| Dr. GRPO | remove division | ascends raw `p` — flat difficulty weight |
| DAPO | drop σ=0 groups | discards groups with no right-vs-wrong contrast |

- Dr. GRPO (arXiv:2503.20783): 43.3% AIME24 with Qwen2.5-Math-7B in 27h/8×A100; matches GRPO accuracy while eliminating the "double-increase" of response length (incorrect responses inflate under GRPO).
- MDP-GRPO (arXiv:2606.06058) formalizes z-score pathologies for discrete rewards: **low-variance amplification** — at k=1, G=16, a correct sample gets `A = 3.87`, unbounded advantages on rare-correct groups.
- Mean-only is RLOO's leave-one-out up to constant `G/(G−1)` (one-dial Prop 1) — the baseline choice is not the interesting axis.
- MC-GRPO (arXiv:2601.22582): at G=2–4 the *mean* baseline itself flips advantage signs; median/MAD centering with G+1 rollouts (drop the median) recovers +4.6 pts at G=2, +2.7 at G=4, gap to G=8 ≤1%. Gains vanish at G=8 (+0.07%).

### 2. Loss aggregation: DAPO token-level is the only length-neutral option

Impossibility theorem (arXiv:2607.23364): with outcome rewards, **no length-only weighting can be both gradient-unbiased and length-invariant**; the Pareto family is `f_α(L) = L^{α−1}` — α=0 → GRPO (per-sample 1/|y|, biased gradient), α=1 → Dr. GRPO (token-sum, length-biased: with R1-Zero's real length ratio 4,965 vs 8,206 tokens, incorrect trajectories would capture 62.3% of gradient).

- Per-sample 1/|y_i| (vanilla GRPO): rewards brevity in correct answers, tolerates verbosity in wrong ones — Dr. GRPO's documented length-inflation mechanism.
- **DAPO token-level 1/Σ|y_i|** (arXiv:2503.14476): per-token length-neutral; sidesteps the theorem (it's a per-token, not per-length, weighting); +1 AIME pt and "enhances training stability and makes the length increase more healthily". TRL warns its `"grpo"` loss type is length-biased; its default `loss_type="dapo"`.
- Dr. GRPO constant-denominator (÷ B·MAX_TOKENS): token-sum scaled by a constant — inherits length dominance, but incidentally cures degenerate-group denominator dilution (GPG concern).
- GSPO (arXiv:2507.18071): sequence ratio `s_i = (π_θ/π_old)^{1/|y_i|}` clipped at ±3e-4/4e-4; evidence is about **MoE routing stability** (Qwen3-30B-A3B), not length control; token-level ratios argued "ill-posed" (single-sample per-token weights, variance accumulating with length).

### 3. Degenerate groups (all-pass/all-wrong) are the dominant silent failure

Silent-group rate = `p^G + (1−p)^G`: **59% at G=4, 44% at G=8** (p=0.5, closed form + validated by subsampling 215k Big-Math prompts). At p≈0.5, G=8–16, negligible (0.78%/0.003%) — but explodes as the pass rate rises (p=0.9, G=8 → 43%) — i.e., **a late-training problem**. Fixes, evidence-ranked:
1. **DAPO dynamic sampling** (filter `0 < k < G`, resample, cap ~10 gen batches): +8 AIME pts (30→50 ladder, the single largest contribution).
2. **GPG denominator rescale** (TRL #6681): `loss /= non-degenerate fraction` — free, no resampling; fixes the silent dilution of the token denominator.
3. OpenRLHF variance filter (`dynamic_filtering_std_threshold`) — same keep-rule as DAPO, implemented as a filter.
4. Keep-with-zero-grad: universally agreed worst (wasted compute + denominator dilution).

### 4. Clipping is inert at μ=1; clip-higher matters only in the multi-iteration regime

- Verified: TRL #6681 shows with `num_iterations=1` and trainer-computed old logps, ratio ≡ 1 and the clipped surrogate is gradient-identical to plain policy gradient (`torch.equal(grad_GRPO, grad_plainPG) = True`). Same structure in verl.
- Clipping becomes active when (a) μ>1, or (b) **old logps come from the inference engine (vLLM)** — which is TRL's default with `use_vllm=True` and relevant to this project.
- DAPO clip-higher (ε_low=0.2, ε_high=0.28): the upper clip throttles low-probability token increases → entropy collapse; raising ε_high is the load-bearing anti-collapse tool (+2 AIME pts). Mechanistic support: "Clip-Low Increases Entropy and Clip-High Decreases Entropy in RL of LLMs" (arXiv:2509.26114).
- SAPO (arXiv:2511.20347, Qwen3-VL) replaces hard clip with a soft gate `σ(τ(x−1))·4/τ` (τ_pos=1.0, τ_neg=1.05) — evidence is from multi-minibatch MoE/dense runs; a knob, not a default, for our regime.

### 5. Reference/KL handling: β=0 default; EMA-of-policy anchor if any

| Method | KL handling | Evidence |
|---|---|---|
| DeepSeek-R1 | β=0.001, ref refreshed every 400 steps, ε=10 stage 1 | AIME 79.8 — the moving reference is a step-function EMA |
| DAPO / Dr.GRPO / ORZ / GRPO-Zero | β=0 | AIME 50 / 43.3 / 1/10 steps / memory savings — the regime default |
| **EMA-PG** (arXiv:2602.04417) | **EMA of the policy as anchor**, small β>0, token-level KL | OlympiadBench 50.8→53.9; agentic +33.3%; *only* controlled study that keeps KL — finds it necessary to prevent full-finetune collapse and Pass@N collapse |
| TRL current default | β=0.0 (ref model not even loaded) | — |

- KL estimator matters: **K3 (R1's) has the wrong gradient** — it descends forward-KL. K3⁺⁺ = K3 × π_θ/π_old (TRL `use_bias_correction_kl`, default ON, attributed to DeepSeek-V3.2) fixes value AND gradient. K3's `exp(ref−θ)` overflows as the policy drifts → the famous "!" token-collapse + NaN (verl #751/#747; TRL #3015 → `kl_log_ratio_clip` fix; clamp ±20).
- Special/format tokens dominate per-token KL by 1–2 orders of magnitude (TRL #2933) — mask them in the KL term and metric only.
- Reference-free is fine for rule rewards; the "keep the policy near the reward model" argument doesn't apply to binary verdicts.

### 6. Group size: G=8; the mini-batch (prompts/step), not G, drives variance

- 2-GRPO (arXiv:2510.00977): "GRPO is Secretly DPO" — contrastive N-vs-M learner; G=2 achieves **97.6% of G=16** with 12.5% rollouts / 21% time (with DAPO-style resampling it ties/beats G=16); group-size ablation G∈{2,4,8,16}: "consistently small", non-monotonic. Binary-reward derivation transfers directly to pass/fail.
- MC-GRPO closes the G=2-vs-G=8 gap to ≤1% (median centering).
- Recommendation: **G=8 with 4–8 prompts/step (32–64 rollouts)** — flat part of the curve; G=4 is a compute-saving option (~0.5–1.5 pts); G=2 needs resampling or median-centering; G=16 wastes 4× rollout cost at our budget.

### 7. Stability recipe (consensus across DAPO / MC-GRPO / TRL / verl-recipe / kalomaze / OpenPipe)

- **LR 1e-6 constant + warmup (~10–20 steps)** — hard consensus; "cosine or linear decaying LRs do not work well with GRPO" (kalomaze); current project default 5e-5 is SFT-scale, must be overridden.
- **Grad clip 0.2** (kalomaze's controlled finding; DAPO's script uses 1.0 — disagreement, no controlled ablation; 0.2 is cheap insurance for small batches).
- Temperature 1.0, top_p 1.0 (universal). Entropy coefficient 0 (DAPO/verl) — clip-higher is the anti-collapse tool; adaptive entropy (Skywork-OR1) and top-entropy-quantile 0.2 masking (Beyond the 80/20 Rule — validated on Qwen3-8B exactly) are experiments.
- **Mask truncated completions** (DAPO Overlong Filtering: +6 AIME; punitive rewards on truncation inject noise). DAPO soft overlong punishment (+3 more).
- μ=1 default (TRL, GPG, our trainer's structure); μ=2–4 (DAPO μ=16, OpenPipe) only as compute-gated experiment.
- Monitoring is load-bearing: entropy, mean token prob, response length, `frac_reward_zero_std`, clip/up-clipped fraction, pass rate — train reward can climb while val collapses (OpenPipe length-collapse regression).
- Train/inference mismatch (vLLM vs trainer) produces biased gradients: TRL ships TIS/MIS correction ON by default (`sequence_mask`, C_max=3.0); aligns with the project's existing `IS_WEIGHTING`/`IS_CAP` (2.0–5.0) machinery.

## Current vs. Recommended (SDFT trainer)

| Axis | Current (SDFT reverse-KL) | Recommended (GRPO mode) | Evidence | Impact |
|---|---|---|---|---|
| Objective | reverse KL(student ‖ teacher) per token, full vocab | sample-level `Σ_t logπ_θ(o_t)·Â_i`, token-level aggregation | DAPO token-level loss; impossibility theorem | different signal source: verdicts vs teacher distribution |
| Advantage | n/a (no rewards in loss) | `A_i = r_i − mean(R_group)` (no std) | Dr. GRPO; one-dial; MDP-GRPO | bounded, no difficulty bias |
| Reward source | verdicts only logged (rank 0) | binary pass_value per rollout, broadcast with payload | DAPO rule rewards; existing `_sample_meta` | small plumbing change |
| Reference KL | teacher logprobs (EMA or frozen) | β=0 default; optional β=0.001 vs **EMA teacher as anchor** | DAPO/Dr.GRPO/ORZ β=0; EMA-PG anchor | −1 full-vocab forward per sample when β=0 |
| IS correction | per-sequence TIS cap 2–5 (IS_WEIGHTING) | sequence-level TIS C_max=3.0 on the PG term | TRL default; existing machinery | keeps vLLM mismatch bounded |
| LR / scheduler | 5e-5, constant (no warmup) | **1e-6, constant + warmup ~15 steps** | consensus (DAPO, TRL, kalomaze) | stability at 8B long-CoT |
| Grad clip | 1.0 hardcoded | 0.2 in GRPO mode | kalomaze | small-batch insurance |
| Truncation | IS NaN-masking only | mask truncated completions from loss | DAPO Overlong Filtering (+6) | kills truncation reward noise |
| Degenerate groups | n/a | GPG rescale (always) + dynamic-sampling flag | DAPO (+8); trl #6681 | 44% silent groups at G=8 fixed |
| Completions per prompt | exactly 1 | G=8 per prompt (rank-local groups) | 2-GRPO; MC-GRPO | − |
| Teacher forward | every sample, (C,V) | only when GRPO_KL_COEF>0 | − | −1 full-vocab softmax/sample by default |

## Design Decisions (decision matrix)

| Decision | Option A | Option B | Option C | Verdict |
|---|---|---|---|---|
| Advantage | z-score (GRPO) — unbounded, arcsine bias | **mean-only (Dr.GRPO)** — bounded, unbiased REINFORCE baseline | median/MAD (MC-GRPO) — best at G≤4 | **B default; C knob for G≤4** |
| Loss aggregation | per-sample 1/\|y\| — length bias α=0 | **token-level 1/Σ\|y\|** — length-neutral, sidesteps theorem | constant denominator (Dr.GRPO) — length dominance α=1 | **B default; C fallback** |
| Degenerate groups | keep-with-zero-grad — worst | **GPG rescale** — free, must | DAPO dynamic sampling — +8 pts, extra decode | **B always; C flag, auto-ON frac>0.2** |
| Clip | symmetric ε=0.2 | **clip-higher 0.2/0.28** — anti-entropy-collapse | soft gate (SAPO) — multi-iteration evidence only | **B defaults (inert at μ=1 anyway)** |
| KL reference | **none (β=0)** — regime default | EMA anchor β=0.001 — EMA-PG, ~free here | frozen ref — drifts, k3 blows up | **A default; B first-class knob** |
| KL estimator | k2 — wrong value, right grad | **K3⁺⁺** — right value AND grad (bias-corrected) | k3 raw — wrong grad, exp overflow | **K3⁺⁺ + log-ratio clamp ±20 + special-token mask** |
| IS correction | none — biased grads as policy drifts | **sequence-level TIS, C_max=3.0** | MIS sequence_mask (TRL default) | **B default; C escalation** |
| Group size | G=4 — 12.5% degenerate, ~1 pt cost | **G=8** — flat curve, TRL default | G=16 — 4× rollout cost, no reliable gain | **G=8 default; G=4 knob** |
| μ (iterations) | **μ=1** — TRL default, clip inert, matches trainer | μ=2–4 — OpenPipe/DAPO evidence | μ=16 (DAPO) | **μ=1 default; μ∈{2,4} compute-gated knob** |

## Recommended configuration (evidence-backed default)

```
G=8 (4–8 prompts/step), advantage = r − mean(group) [no std],
loss = DAPO token-level clipped surrogate, ε_low=0.2 / ε_high=0.28,
old_logps = vLLM rollout logprobs (+ sequence-level TIS C_max=3.0),
KL β=0 (knob: 0.001 vs EMA teacher anchor, K3⁺⁺, special-token-masked),
GPG degenerate-group rescale ON, dynamic sampling flag OFF (auto-ON when
frac_reward_zero_std > 0.2), mask truncated completions,
LR 1e-6 constant + ~15-step warmup, grad clip 0.2, temperature 1.0, μ=1.
```

## Gotchas & Pitfalls

1. **z-score advantage with binary rewards is unbounded** — rare-correct groups get A≈3.9 (MDP-GRPO); the "classic GRPO" is the risky choice here.
2. **K3 KL exp-overflow → "!" collapse**: `exp(ref_logp − logp)` → inf → NaN gradients → single-token output (verl #751/#747, TRL #3015). Clamp log-ratio ±20; use bias-corrected K3⁺⁺; or β=0.
3. **`logging_nan_inf_filter` hides NaN steps** (TRL #6702): a dying run looks healthy in logs — own NaN/entropy/length monitors.
4. **Special tokens dominate per-token KL** by 1–2 orders of magnitude (TRL #2933) — mask them or the KL metric is meaningless.
5. **Length collapse after peak is irreversible-ish** (OpenPipe): train reward keeps climbing while val accuracy collapses; checkpoint-fork at peak val, don't trust train reward.
6. **Clipping inert at μ=1** — don't hyperparameter-tune what provably does nothing in your regime; it activates only with μ>1 or engine-supplied old logps.
7. **vLLM-vs-trainer logprob mismatch** is a real biased-gradient source (TRL docs; TIS/MIS default ON) — this project's vLLM rollouts + Megatron recompute is exactly the setup.
8. **Frameworks disagree on "default GRPO"**: TRL default loss is now `dapo`, verl/OpenRLHF default advantage estimator is `gae`, prime-rl default advantage is Dr.GRPO-without-std — cite the config, not the name.
9. **"Log-signal GRPO" and "GSPO-flen" are not real** (verified: 0 arXiv hits, 0 GitHub hits, 0 source-tree hits) — treat any recipe using them with suspicion (see Eliminated Options).
10. **DAPO's paper numbers (50) are below verl's community repro (52)** — infra/config deltas matter as much as the algorithm; budget for local tuning.

## Eliminated Options

- **Log-signal GRPO** (Wu et al. 2025 as hypothesized): **does not exist.** Verified 2026-08-15: arXiv API `all:"log-signal" AND all:"GRPO"` → 0 results; GitHub repo search → 0; GitHub issue search → 120 false positives (CI logs, unrelated PRs); grep of TRL/verl/OpenRLHF source → 0 hits. Closest real mechanisms: GSPO sequence-level log-ratio averaging (arXiv:2507.18071), kalomaze's power-scaled rewards (correct-verdict rewards rescaled as acc⁴), and the GRPO gradient itself (advantage × ∇log π, a log-probability signal by construction). If a "log-signal" objective is wanted, it would be a new variant (e.g., A → sign(A)·log(1+|A|)) — no pre-packaged implementation or evidence exists.
- **GSPO-flen (Duke, 2025)**: not found on arXiv (only Qwen's Group *Sequence* Policy Optimization, arXiv:2507.18071, matches "GSPO"). Unverifiable; dropped.
- **SPPO self-play** (arXiv:2405.00675): Nash-equilibrium preference self-play — needs a preference oracle (PairRM) and is for RLHF, not RLVR; ruled out for our binary-verdict setting.
- **ReMax greedy baseline** (arXiv:2310.10505): needs a second greedy rollout per sample (~2× decode); RLOO-equivalent family; superseded by mean-only at G≥8 evidence.
- **Frozen initial-checkpoint KL reference**: policy drift makes the K3 term blow up (verl #751 family); R1 itself refreshed every 400 steps; EMA anchor (EMA-PG) is the only evidence-backed non-frozen choice.
- **LLaMA-Factory**: ships no GRPO in main — not a candidate implementation reference.

## Sources

Papers (arXiv, full-text HTML fetched): 2402.03300 (GRPO), 2503.14476 (DAPO), 2503.20783 (Dr. GRPO / Understanding R1-Zero-Like Training), 2507.18071 (GSPO), 2607.00152 (three operations on one number), 2607.23364 (impossibility), 2606.06058 (MDP-GRPO/GRPO-CARE), 2511.04256 (SSPO), 2511.20347 (SAPO), 2510.00977 (2-GRPO), 2601.22582 (MC-GRPO), 2504.02546 (GPG), 2402.14740 (RLOO), 2310.10505 (ReMax), 2405.00675 (SPPO), 2501.12948 (DeepSeek-R1), 2503.24290 (Open-Reasoner-Zero), 2602.04417 (EMA-PG), 2508.03772 (GTPO), 2505.22312 (Skywork-OR1), 2506.01939 (Beyond the 80/20 Rule), 2509.26114 (clip entropy), 2606.18487 (rank inversion / entropy collapse), 1606.02647 (Retrace), 1806.09055 (IMPALA/TIS), 2512.02556 (DeepSeek-V3.2, via TRL), 2407.16216 (RL post-training survey).

Code: TRL `grpo_trainer.py`/`grpo_config.py` (main) — loss types, `scale_rewards`, `use_bias_correction_kl`, vLLM IS correction defaults, `frac_reward_zero_std`; TRL PR #6681 (GPG, clip-inert proof), PRs #6637/#6667, issues #2933/#3015/#6166/#6702; verl `core_algos.py` (advantage registry, `loss_agg_mode`, `kl_penalty`), verl-recipe DAPO script, issues #747/#751/#3025; OpenRLHF (advantage estimators, PR #1272); prime-rl (algorithms layer, Dr.GRPO-without-std default); open-r1 (TRL-based recipes, issues #407/#587/#593).

Community: kalomaze GRPO judge experiments; Neel Somani "Is GRPO broken?"; OpenPipe temporal-clue via HN 43284420/43288042; GRPO-Zero; TRL/verl docs.
