# GRPO Training Mode — design (`trainer.py`, `chunked_head.py`)

> New `LOSS_TYPE=grpo` mode: on-policy RL against the env verdicts, replacing
> the reverse-KL-to-teacher objective. Research basis (variants, evidence,
> formulas): `docs/research/RESEARCH_grpo_variants.md`. This spec is design
> only — no implementation.

## Baseline (today)

The only objective is **reverse KL(student ‖ teacher)** computed by
`make_kl_processor` (`chunked_head.py:113`) as an MCore `output_processor`
hook: one LM-head call on completion rows, `ChunkedRowKL` analytic backward,
per-sequence TIS weight rescale (`chunked_head.py:28-67`). Rewards (binary
verdicts) are computed by the envs (`_sample_meta`, `trainer.py:144-172`)
but only logged — they never enter the loss. Each dataset example produces
exactly **one** completion (`vllm_generate`, no `n`-sampling), and rollouts
are sharded to ranks as disjoint slices (`trainer.py:931`).

GRPO needs: G completions per prompt (a group), a reward per completion,
group-relative advantages, and a sample-level policy-gradient loss. None of
that exists — this spec defines it.

## Research summary (what we're building and why)

Evidence-backed composite for our regime (binary verdict rewards, 4k–8k
completions, G=4–16, one fwd/bwd per rollout — see research report §TL;DR):

- **Advantage: mean-only** `Â_i = r_i − mean(R_group)` — no std division.
  z-score is unbounded with binary rewards (low-variance amplification,
  MDP-GRPO) and carries an exact question-difficulty bias (one-dial paper).
  Mean-only is RLOO-equivalent up to a constant.
- **Loss: DAPO token-level** clipped surrogate, `1/Σ|y_i|` denominator —
  the only per-token length-neutral aggregation; per-sample `1/|y|` and
  token-sum both have documented length biases (impossibility theorem).
- **Clip: 0.2 / 0.28** (clip-higher) — provably inert at μ=1 with
  trainer-computed old logps, but the only evidence-backed anti-entropy-
  collapse tool once old logps come from vLLM or μ>1.
- **Degenerate groups** (all-pass/all-wrong, 44% of groups at G=8, p=0.5):
  GPG-style denominator rescale always on; DAPO-style dynamic sampling as a
  flag (auto-ON when degenerate fraction > 0.2).
- **KL: β=0 default** (regime default per DAPO/Dr.GRPO/TRL). Knob:
  β=0.001 against the **EMA teacher as anchor** (EMA-PG; ~free here — the
  teacher server already runs), K3⁺⁺ estimator, log-ratio clamp ±20,
  special-token masking.
- **IS correction: sequence-level TIS, C_max=3.0** on the PG term —
  vLLM-vs-trainer mismatch (TRL ships this ON by default; matches existing
  `IS_WEIGHTING`/`IS_CAP` machinery).
- **Stability: LR 1e-6 constant + ~15-step warmup, grad clip 0.2,
  temperature 1.0, G=8, μ=1**, mask truncated completions.

## Design

### 1. Config contract (`config.py` — env var + default + import-time validation)

| Var | Default | Meaning |
|---|---|---|
| `LOSS_TYPE` | `sdft` | `sdft` = existing reverse-KL path (byte-identical); `grpo` = this mode. Independent of `TRAIN_MODE` (full/lora). |
| `GRPO_GROUPS` (G) | `8` | rollouts per prompt. Invariant: `GRAD_ACCUM_STEPS % (world_size × G) == 0`. |
| `GRPO_ADV` | `mean` | `mean` = Dr.GRPO `r − μ`; `zscore` = vanilla `(r − μ)/σ` (experiment knob); `median` = MC-GRPO (G+1 rollouts, drop median, MAD scale — knob). |
| `GRPO_CLIP_LOW` / `GRPO_CLIP_HIGH` | `0.2` / `0.28` | DAPO clip-higher band. |
| `GRPO_OLD_LOGPS` | `vllm` | `vllm` = ratio vs rollout logprobs from vLLM payload (clip active, needs IS); `detached` = ratio vs detached trainer logps (clip inert, GPG-style plain PG). |
| `GRPO_IS_C_MAX` | `3.0` | sequence-level TIS clamp on ρ (sweep 2–5; reuses `IS_CAP` semantics) |
| `GRPO_IS_MODE` | `truncate` | `truncate` = clamp ρ; `mask` = zero ρ outside [C_min, C_max] (escalation) |
| `GRPO_KL_COEF` (β) | `0.0` | reference-KL coefficient. 0 = no reference forward at all (biggest compute win). |
| `GRPO_KL_REF` | `ema_teacher` | reference source when β>0: the existing logprob server (EMA teacher). Frozen `TEACHER_MODEL_PATH` also works (R1-style moving ref = the per-step EMA sync already in place). |
| `GRPO_FILTER_GROUPS` | `0` | DAPO dynamic sampling: keep groups iff `0 < #pass < G`; resample with cap (`GRPO_MAX_GEN_BATCHES`=10, verl convention). Default off; docs recommend ON when `frac_reward_zero_std` > 0.2. |
| `GRPO_LR` | `1e-6` | overrides `LEARNING_RATE` in grpo mode (SFT-scale 5e-5 is unsafe here). |
| `GRPO_LR_WARMUP_STEPS` | `15` | linear warmup to `GRPO_LR`, then constant (no decay — kalomaze). |
| `GRPO_GRAD_CLIP` | `0.2` | overrides `MAX_GRAD_NORM` in grpo mode. |
| `GRPO_MASK_TRUNCATED` | `1` | zero the loss (and reward) of completions that hit `GEN_MAX_NEW_TOKENS` (DAPO Overlong Filtering; never punitive). |

Import-time asserts: `LOSS_TYPE ∈ {sdft, grpo}`; when `grpo`:
`GRAD_ACCUM_STEPS % (world_size × G) == 0`; `GRPO_GROUPS ≥ 2`;
`GRPO_ADV=median` ⇒ G+1 rollouts required (see §2).

### 2. Rollout: G completions per prompt, rank-local groups

- `_build_env`/`produce` (`trainer.py:118-141, 190-227`): unchanged env
  construction, but `vllm_generate` is called **G times per example**
  (`ThreadPoolExecutor` already exists; G requests in flight per prompt).
  Group index `g ∈ [0, G)` stamps each rollout.
- **Reward plumbing**: `pass_value` (already computed in `_sample_meta`,
  `trainer.py:144-172` — rag: reflector verdict; api_adapter: episode
  verdict) is **embedded in the rollout payload** for every rollout
  (promote the async-mode `_meta` pattern, `trainer.py:244`, to both modes
  when `LOSS_TYPE=grpo`).
- **Grouping is rank-local**: broadcast the full batch as today; each rank
  slices **group-aligned** — `rollout_data[r*L : (r+1)*L]` where `L =
  local_accum_steps` now counts *groups* (i.e., `L × G` rollout records per
  rank). Advantages are computed per rank from that rank's own groups — no
  new collectives. (2-GRPO evidence: #prompts/step, not cross-rank group
  size, drives variance.)
- Async mode (`_pull_microbatch`): pop `world_size × G` records, broadcast,
  `G` per rank — mechanical extension of `trainer.py:417-442`.
- Effective per-step budget: `GRAD_ACCUM_STEPS/G` prompts × G = 32–64
  rollouts at defaults (G=8, GA=32). `policy_version` staleness machinery
  (`trainer.py:104`) is reused unchanged.

### 3. Loss math (`chunked_head.py` — new `make_grpo_processor`)

Same hook mechanism as `make_kl_processor` (`chunked_head.py:113`): one LM
head call on completion rows (`hidden_c`, `chunked_head.py:156-157`),
chunked over rows of 128 (same memory profile as today). Per completion:

```
logp_θ[t]     = log_softmax(z[t], token_t)          # with grad, selected tokens
old_logp[t]   = rollout_log_probs[t]                # vllm mode (payload), or
                logp_θ[t].detach()                  # detached mode
r[t]          = exp(logp_θ[t] − old_logp[t])
A_i           = r_i − mean(R_group)                 # per group, precomputed at rollout time
ℓ[t]          = −min(r[t]·A_i, clip(r[t], 1−ε_low, 1+ε_high)·A_i)
loss_group    = (Σ_{i,t} ℓ[i,t]·mask[i,t]) / Σ_{i,t} mask[i,t]    # DAPO token-level
                × ρ_seq · GPG_rescale · mask_truncated
```

- **Advantages are precomputed** at rollout time (rewards known before any
  training forward) and carried per-rollout — the loss hook only gathers.
- **`GRPO_ADV` variants**: `zscore` adds `/ (σ_group + 1e-4)`; `median`
  samples G+1, centers on median, scales by MAD, drops the median rollout
  from backward (MC-GRPO).
- **Selected-token gather**: unlike `ChunkedRowKL` (which returns detached
  `policy_logp`, `chunked_head.py:96`), the GRPO processor keeps
  `logp_θ[t]` attached — the loss is a scalar over gathered tokens, so
  backward is a plain (not analytic) autograd path. Memory profile is
  dominated by the same `(C, V)` head output as today; chunked processing
  keeps the fp32 conversion bounded per chunk.
- **GPG rescale (always on)**: `loss *= num_groups / max(num_nondegenerate_groups, eps)`
  where degenerate = all-same reward (TRL #6681 `inverse_alpha`); log
  `frac_reward_zero_std` per step.
- **Dynamic sampling** (`GRPO_FILTER_GROUPS=1`): before training, rank 0
  drops groups with `#pass ∈ {0, G}` and re-generates (cap
  `GRPO_MAX_GEN_BATCHES=10`); broadcast payload only contains kept groups.
- **Truncation mask**: `mask[i,t] = 0` for completions where
  `finish_reason == "length"` (DAPO Overlong Filtering; never a negative
  reward).
- **IS correction** (`GRPO_OLD_LOGPS=vllm`): `ρ_seq = clamp(exp(mean_t
  (logp_θ − old_logp)), C_min, C_max)` — sequence-level TIS; applied to
  `ℓ[i,t]` (and to the KL term when β>0). With `GRPO_OLD_LOGPS=detached`,
  ρ ≡ 1 and the clip is inert (plain PG, GPG-style).

### 4. Reference KL (only when `GRPO_KL_COEF > 0`)

- Reference logprobs come from the **existing logprob server** — same TCP
  client (`logprob_client.py:84`), but requested with the **plain student
  prompt** (`prompt_ids + completion_ids`, `prompt_len = len(prompt_ids)`),
  NOT the privileged teacher prompt. `GRPO_KL_REF=ema_teacher` (default) =
  EMA-PG's policy-anchor recipe: the EMA teacher is exactly an
  EMA-of-policy anchor (server-side `lerp_` with `EMA_ALPHA`,
  `logprob_server.py:315`), refreshed per optimizer step by the existing
  weight sync (`trainer.py:558-570`). Frozen reference = set
  `TEACHER_MODEL_PATH` (R1-style moving ref ≈ per-step refresh).
- Estimator — K3⁺⁺ (bias-corrected K3, TRL `use_bias_correction_kl` /
  DeepSeek-V3.2), per token, attached to the policy logps:

```
log_ratio  = clamp(ref_logp − logp_θ, −20, 20)        # safety: k3 exp-overflow
k3         = exp(log_ratio) − log_ratio − 1
k3_pp      = k3 · exp(logp_θ − old_logp)              # bias correction (K3++)
k3_pp      = mask_special_tokens(k3_pp)               # chat/think/EOS tokens
loss      += β · mean(k3_pp over active tokens)
```

- Special-token masking: mask in the KL term and the logged KL metric only
  (format tokens dominate per-token KL 1–2 orders of magnitude, TRL #2933);
  the policy-gradient term is untouched.

### 5. Trainer flow changes (`trainer.py`)

`_train_sample` (`trainer.py:298-414`) branches on `LOSS_TYPE`:
- `sdft`: current path, untouched (byte-identical).
- `grpo`: skip the teacher-logprob request (`trainer.py:346-357`) entirely
  when `GRPO_KL_COEF == 0` — removes one `(C,V)` full-vocab forward per
  sample; build `make_grpo_processor(...)` instead of
  `make_kl_processor(...)`; same `no_sync`/`scaled_loss` accumulation
  (`trainer.py:400-405`).

`_step_tail` (`trainer.py:452-597`): GRPO-mode metrics added to the wandb
log: `grpo/frac_reward_zero_std`, `grpo/pass_rate`, `grpo/entropy`
(mean −logp_θ over completions), `grpo/mean_length`, `grpo/clip_frac`
(fraction of tokens up-clipped), `grpo/adv_mean_std`, `grpo/sampling_logp_diff`
(mean |logp_θ − old_logp| — the train/inference mismatch monitor).
Loss all-reduce (`trainer.py:504-508`) unchanged.

## Hyperparameter contract (grpo mode defaults)

| Var | Default | Critical | Notes |
|---|---|---|---|
| `LOSS_TYPE` | `sdft` | **yes** | `grpo` switches the objective |
| `GRPO_GROUPS` | `8` | **yes** | must satisfy `GRAD_ACCUM_STEPS % (world_size × G) == 0` |
| `GRPO_ADV` | `mean` | — | `zscore`/`median` are experiment knobs |
| `GRPO_CLIP_LOW`/`HIGH` | `0.2`/`0.28` | — | inert at μ=1 + detached old logps; active with vllm old logps |
| `GRPO_OLD_LOGPS` | `vllm` | — | `vllm` = TRL use_vllm path (clip + IS active); `detached` = GPG plain-PG path |
| `GRPO_IS_C_MAX` / `GRPO_IS_MODE` | `3.0` / `truncate` | — | sequence-level TIS; `mask` escalation |
| `GRPO_KL_COEF` | `0.0` | — | 0 skips the teacher forward entirely |
| `GRPO_KL_REF` | `ema_teacher` | — | EMA anchor (EMA-PG); frozen via `TEACHER_MODEL_PATH` |
| `GRPO_FILTER_GROUPS` | `0` | — | auto-recommend ON when `frac_reward_zero_std` > 0.2 |
| `GRPO_MAX_GEN_BATCHES` | `10` | — | dynamic-sampling resample cap |
| `GRPO_LR` / `GRPO_LR_WARMUP_STEPS` | `1e-6` / `15` | **yes** | constant-after-warmup; SFT-scale LR unsafe |
| `GRPO_GRAD_CLIP` | `0.2` | — | small-batch insurance |
| `GRPO_MASK_TRUNCATED` | `1` | — | never a punitive reward |

## Hard invariants

- `LOSS_TYPE=grpo` ⇒ `GRAD_ACCUM_STEPS % (world_size × GRPO_GROUPS) == 0`
  (group-aligned rank slicing; assert at startup like `trainer.py:631-634`).
- `GRPO_ADV=median` ⇒ G+1 rollouts per group (median rollout dropped from
  backward).
- Weights are synced to vLLM every optimizer step as today — rollouts for
  step n+1 are generated by the post-step policy (`policy_version` tracks
  it; staleness bounded by the sync cadence, same as the SDFT path).
- `GRPO_KL_COEF > 0` ⇒ the logprob weight sync runs every step (already the
  default) so the EMA anchor tracks the policy; β=0 may skip the logprob
  server weight sync only if no other consumer needs it.

## Verification plan

1. **Loss numerics (unit, no GPU cluster)**: `make_grpo_processor` on toy
   logits vs hand-computed policy gradient: (a) advantage correctness
   (k-of-G binary rewards → `r − k/G`; degenerate group → 0 + GPG rescale);
   (b) **gradient-identity check** with `GRPO_OLD_LOGPS=detached`:
   `max|grad_grpo − grad_plainPG| == 0` (GPG/TRL #6681 style); (c) with
   `vllm` old logps: ratio ≠ 1 ⇒ clip band and IS clamp engaged, ρ_seq
   matches hand-computed `exp(mean(logp − old))`.
2. **KL path numerics** (β>0): k3⁺⁺ vs reference implementation; special-
   token mask zeroes exactly `all_special_ids` positions; log-ratio clamp
   ±20 bounds `exp` (no inf).
3. **Replay determinism**: reuse `RECORD_ROLLOUT_PATH`/`REPLAY` machinery —
   a grpo run on fixed rollouts trains deterministically; group alignment
   per rank verified via `DEBUG_ROLLOUT_HASH`-style checks.
4. **Cluster smoke** (2 trainers, G=8, GA=16): entropy/length/reward curves
   sane over 50–100 steps; `frac_reward_zero_std` logged; no NaN (own
   NaN detector — don't trust `logging_nan_inf_filter`); weight-sync + IS
   metrics present; pass rate improves on the eval split.
5. **A/B (optional)**: verl DAPO recipe on the same data as a reference
   point — the composite config above is the TRL `loss_type="dapo" +
   scale_rewards="none"` equivalent.

## Known gotchas / open questions

- `GRPO_OLD_LOGPS=vllm` makes clipping active — then `GRPO_CLIP_HIGH=0.28`
  matters; with `detached` it's provably inert. Document in the config, not
  in code comments.
- The TCP logprob recv buffer is sized by `GEN_MAX_NEW_TOKENS`
  (`logprob_client.py:63`) — only relevant when β>0; unchanged sizing rules.
- vLLM weight sync happens every optimizer step; G=8 rollouts per prompt
  means 4–8× the generation load per step — the async-rollout producer
  (`ASYNC_ROLLOUT`) composes with grpo mode (group-aligned pops), which is
  the first thing to enable if generation becomes the bottleneck.
- Reflector/api-adapter verdicts are binary; continuous rewards (e.g.,
  reflector score, partial credit) would need new env output — out of scope.
- `median` advantage with G=8: evidence says gains vanish at G≥8 — knob
  exists for the G=4 compute-saving configuration.
- Open question: whether the z-score-vs-mean difference is material at
  G=8–16 with binary rewards (theory says the dial, empirics say modest at
  small G) — the `GRPO_ADV` knob is the experiment.
