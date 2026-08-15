# GRPO Live Campaign — Plan

Node: `rh-h100-05`, repo `/home/rohan/1_Projects/rohans_sdft_api_adapter`.
Design spec (source of truth for math/config): `docs/megatron_trainer/grpo.md` +
`docs/research/RESEARCH_grpo_variants.md` (this repo, `ra/grpo`, merged via PR #19).
I am running this campaign autonomously from here on (`autoresearch` skill).

## Goal

Beat baseline accuracy **0.7415** (with thinking) on the analyze_deepresearch
eval for the gpt-oss-20B student — via **GRPO**, not self-distillation.
Reverse-KL self-distillation is retired for this phase; **no more distillation
loss**. Reward comes from the **reflector verdict that already exists**
(`megatron_trainer/reflector.py::run` → `{"verdict": "PASS"|"FAIL", ...}`,
already computed per-rollout in `trainer.py::_sample_meta` but currently only
logged, never used in the loss) — no new grading infra needed.

## Objective

- Primary: eval accuracy > 0.7415. Stretch/significant: ≥ **0.7715**
  (+3pp — the McNemar significance threshold established in the prior phase).
- Mechanism: `LOSS_TYPE=grpo` (on-policy, group-relative policy gradient)
  replacing `LOSS_TYPE=sdft` (reverse-KL) entirely as the training objective.
- Reward: binary reflector verdict, `PASS→1.0 / FAIL→0.0`, one per rollout.

## Verification (how we know it worked)

- Same harness as the prior phase, unchanged: `poll_eval.sh <run> <epoch>` →
  reads `{maas-knowledge-eval}/eval_results/analyze_deepresearch/{run}/{epoch}/run_1.json`,
  field `summary.accuracy`. 557 questions (~500 unique), Claude-judged
  majority vote.
- Noise floor: paired SE ≈ 1.5pp (McNemar). `0.7415–0.7715` = "crossed but
  not significant" (this is exactly where the best SDFT run, E045, landed at
  0.7433). Only claim success at **≥ 0.7715**.
- Track GRPO health metrics every step (new, don't exist yet — part of this
  implementation): `grpo/pass_rate`, `grpo/frac_reward_zero_std` (degenerate
  groups), `grpo/entropy`, `grpo/clip_frac`, `grpo/adv_mean_std`,
  `grpo/sampling_logp_diff` (train/inference mismatch). A run with flat/rising
  accuracy but collapsing entropy or `clip_frac→0` is failing silently —
  watch these, not just eval accuracy.

## Why pivot off self-distillation (context)

- Prior phase: 47 experiments (E001–E047), all reverse-KL + IS variants,
  plateaued at 0.69–0.73. Only lever that ever crossed baseline was **context
  injection** (E045: 0.7433, +0.18pp, not significant) — an input-signal
  change, not a loss change.
- SFT-anchor hybrid (direct NLL gradient mixed into KL) was tried and
  **refuted twice** (E038 λ=0.1 destabilized KL; E039 λ=0.01 degraded
  accuracy by 4.3pp) — the belief on record is "NLL gradient competes with
  and degrades reverse-KL."
- Standing hypothesis in `STATE.md`: reverse-KL optimizes distribution
  matching to the teacher, not correctness — "sharper but not smarter."
  GRPO's reward is literally correctness (reflector verdict), so it targets
  the eval criterion directly instead of a proxy.
- Decision: run GRPO **on top of the context-injected datasets**
  (`subset_k400_subset_with_context.jsonl`, `combined_dataset_train_sdft_with_context.jsonl`)
  — the one proven lever — rather than plain data, so a negative result isn't
  confounded by "model doesn't have the facts."

## Design surface (full detail in `docs/megatron_trainer/grpo.md`)

| Knob | v1 default | Why |
|---|---|---|
| `LOSS_TYPE` | `grpo` | retires `sdft` path for this phase |
| `GRPO_GROUPS` (G) | `8` | rollouts/prompt; `GRAD_ACCUM_STEPS % (world_size×G) == 0` |
| `GRPO_ADV` | `mean` | `r_i − mean(group)`, no /σ (binary reward, low-variance amplification risk with zscore) |
| `GRPO_CLIP_LOW/HIGH` | `0.2` / `0.28` | DAPO clip-higher, active since old logps come from vLLM |
| `GRPO_OLD_LOGPS` | `vllm` | ratio vs rollout logprobs (clip + IS active) |
| `GRPO_IS_C_MAX` | `3.0` | sequence-level TIS clamp, reuses `IS_CAP` semantics |
| `GRPO_KL_COEF` (β) | `0.0` | **no reference forward at all** — drops the 120b teacher from the critical path entirely |
| `GRPO_LR` / warmup | `1e-6` / 15 steps | constant after warmup; SFT-scale LR is unsafe here |
| `GRPO_GRAD_CLIP` | `0.2` | small-batch insurance |
| `GRPO_MASK_TRUNCATED` | `1` | never punish length-truncated completions |
| temperature | `1.0` | literature default for GRPO exploration |

Reward wiring (the actual new plumbing, everything else in the table is
already-designed math): `_sample_meta`'s `pass_value` is computed once per
rollout today and only logged. Under `LOSS_TYPE=grpo` it becomes the reward
`r_i` feeding the group advantage — same reflector call, new consumer.

## Implementation plan (minimal diff, in order)

1. On node05: commit/stash the outstanding loss-function work already sitting
   uncommitted (`GOAL.md`, `EXPERIMENTS.log`, `STATE.md`, `analysis/*.md`,
   `add_retrieval_context.py`, new dataset, `new_dataset_format.png`) — don't
   start branching on a dirty tree.
2. Branch `ra/grpo-live` off `ra/autoresearch-loop` tip — inherits async
   rollout + context-injection dataset support, drops nothing.
3. `config.py`: add the `GRPO_*` env vars above + import-time asserts
   (`LOSS_TYPE ∈ {sdft, grpo}`, group-alignment invariant).
4. `trainer.py` `_build_env`/`produce`: call `vllm_generate` **G times per
   prompt** when `LOSS_TYPE=grpo` (existing `ThreadPoolExecutor`, just more
   requests in flight); stamp group index `g`; embed `pass_value` in the
   rollout payload for every rollout (promote from log-only to reward).
5. `trainer.py` rollout slicing: group-aligned rank slices
   (`rollout_data[r*L:(r+1)*L]`, `L` now counts **groups**, not samples).
6. `chunked_head.py`: new `make_grpo_processor` — attached (non-detached)
   logp gather, ratio vs old logp, DAPO token-level clipped surrogate ×
   precomputed advantage, GPG degenerate-group rescale, truncation mask.
7. `trainer.py` `_train_sample`: branch on `LOSS_TYPE`; `grpo` path **skips
   the teacher-logprob request entirely** when `GRPO_KL_COEF==0` — removes
   one full-vocab forward per sample, frees the 120b teacher's GPU(s).
8. `trainer.py` `_step_tail`: add the six `grpo/*` wandb metrics.
9. Unit test first (cheapest bug catch, no cluster needed): toy logits,
   `GRPO_OLD_LOGPS=detached` ⇒ gradient must equal plain policy-gradient
   exactly (spec's gradient-identity check). Do this before touching GPUs.
10. Cluster smoke: 2 trainers, G=8, GA=16, ~50–100 steps on
    `subset_k400_subset_with_context.jsonl` — no NaN, sane
    reward/entropy/length curves, weight-sync + IS metrics present, pass
    rate moves off its initial value.

## Experiment plan (first entries, `autoresearch/EXPERIMENTS.log`)

Continuing the existing log (exact next number = last entry on node05 at
E04x, confirm before writing — the prior campaign's tail wasn't fully
re-verified as of writing this plan).

- **Smoke test** (not a science experiment — infra validation): defaults per
  table above, k400-with-context, short. Pass/fail = "no crash, metrics sane."
- **First real run**: same defaults, full length on
  `subset_k400_subset_with_context.jsonl`. `ASYNC_ROLLOUT=1` from the start
  (G=8 means 8× the generation load per step — async rollout is the standard
  mitigation per the existing design doc, and the prior campaign already
  adopted it).
- **Next**: if noisy but promising, try `GRPO_FILTER_GROUPS=1` (dynamic
  sampling for degenerate groups) or `GRPO_ADV=zscore` — both are explicit
  "knob, not default" experiment surfaces in the design spec.
- Comparison bar: must beat E045 (0.7433, best SDFT result) to be
  interesting at all, and beat 0.7715 to be a real win.

## Operational risks / notes

- **GPU reservation**: all 8 GPUs on rh-h100-05 show `IN_USE` under
  `ronny-romeo` (manual reservation, ~73h remaining) but 0% actual
  utilization observed. Need to confirm this is stale before reserving —
  will surface to you rather than silently launch on someone else's hold.
- **Compute cost**: G=8 → 8× rollouts/prompt vs the old 1×. Generation is the
  new bottleneck, not the teacher forward (which we're dropping via
  `GRPO_KL_COEF=0`). This roughly cancels — teacher GPU capacity can likely
  be reallocated to vLLM rollout capacity.
- **Reflector cost**: PASS/FAIL already costs one Claude/Vertex call per
  rollout; G× more rollouts means G× more reflector calls per prompt.
  Existing retry/backoff (tenacity) stays as-is.
- **Dirty working tree on node05** — must commit/stash before branching
  (item 1 above).

## Open questions carried into this phase

1. Does GRPO beat the 0.73 reverse-KL ceiling on **plain** data too, or does
   it need context injection as well (compounding vs. substitute lever)?
2. Reflector pass rate is currently ~0.65–0.74 (not 0.5) — how often are
   groups degenerate (all-pass/all-fail) at G=8 with this pass rate? Spec's
   44%-at-p=0.5 number doesn't directly apply; watch
   `grpo/frac_reward_zero_std` from the smoke test to calibrate whether
   `GRPO_FILTER_GROUPS` should be on by default here.
3. Literature defaults (LR=1e-6, clip 0.2/0.28, G=8) are generic — none of
   this has been validated on this task's long-CoT thinking-mode completions.
4. With `GRPO_KL_COEF=0` there's no distributional anchor at all — risk of
   entropy collapse or reward hacking against the reflector. `β>0` against
   the EMA teacher is the documented mitigation if that shows up.

## Next steps

1. You review this plan.
2. I formalize `autoresearch/GOAL.md` / `STATE.md` / `EXPERIMENTS.log` on
   node05 for this phase (per the `autoresearch` skill), confirming exact
   last E-number and current git/GPU state.
3. Implement steps 1–10 above.
4. Start the smoke test, then the first real GRPO run, then continue the
   loop autonomously.
