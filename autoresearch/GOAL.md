# Goal

Beat the baseline accuracy of **0.7415** (with thinking) on the analyze_deepresearch eval (557 questions, 500 unique, majority-vote judge via Claude Opus 4) on a 20B GPT-OSS student model — via **GRPO** (on-policy group-relative policy gradient), not self-distillation. Reverse-KL distillation is retired as of this phase.

# Problem Statement

47 self-distillation experiments (E001–E046, reverse-KL against a frozen 120B teacher, every loss/IS/schedule variant tried) plateaued at 0.69–0.73 accuracy. The only lever that ever crossed baseline was context injection (E045: 0.7433, +0.18pp, not significant) — an input-signal change, not a loss change. Standing hypothesis: reverse-KL optimizes distribution-matching to the teacher, not correctness ("sharper but not smarter" — devlogs Run 9). GRPO replaces the objective entirely: reward = the reflector's existing PASS/FAIL verdict (already computed every rollout, previously only logged). The core question is now whether reward-driven RL can push past the ceiling that 25+ loss-geometry variants couldn't.

# Baseline

- **With thinking:** 0.7415
- **Without thinking:** 0.7397
- Eval: 557 questions (~500 unique), majority-vote judge via Claude Opus 4
- Noise floor: paired SE ≈ 1.5pp (McNemar), 95% significance threshold ≈ 3pp
- **Best self-distillation result (retired phase):** E045, 0.7433 @ epoch 5, settled 0.72–0.73 through epoch 10. GRPO must beat this to be interesting, and beat 0.7715 to be a real win.

# What to Optimize

- **Primary metric:** accuracy on analyze_deepresearch eval
- **Target:** >0.7415 (beat baseline), ideally >0.7715 (statistically significant improvement)
- **Secondary signal:** GRPO health metrics per step — `grpo/pass_rate`, `grpo/frac_reward_zero_std` (degenerate groups), `grpo/entropy`, `grpo/clip_frac`, `grpo/sampling_logp_diff` (train/inference mismatch). Rising accuracy with collapsing entropy or `clip_frac→0` is failing silently — these matter as much as the eval number.

# How to Evaluate

- `auto_eval_poller.sh` (in `maas-knowledge-eval` repo, `scripts/rohan/`) was manually stopped when the SDFT campaign was paused — restart it (or run evals manually) once GRPO checkpoints exist.
- Per-checkpoint accuracy at `/home/rohan/1_Projects/maas-knowledge-eval/eval_results/analyze_deepresearch/{run_name}/{checkpoint}/run_1.json`
- Poll with: `bash poll_eval.sh <run_name> <epoch_or_step>`

# Constraints (do not change)

- Model: `unsloth/gpt-oss-20b-BF16`
- `STUDENT_THINKING=1`, `THINKING_BUDGET=512`
- `TRAIN_MODE=lora` — **required** for GRPO (reversed from the original "no LoRA" call: full FT's FSDP-sharded AdamW state doesn't fit this 20B model on <4 trainer GPUs — `grpo_smoke_test_2` OOM'd on the first `optimizer.step()` at ~79.15/79.17GB with only 2 trainers. LoRA sidesteps the ceiling at any trainer count. Issue #22, the gpt-oss MoE LoRA export bug that blocked this in E046, is fixed as of PR #23.) `LORA_DIM=32`/`LORA_ALPHA=32`.
- `ASYNC_ROLLOUT=1` — **required** for GRPO (v1 implementation asserts this; group-atomic streaming producer, not the old sync path)
- `LOSS_TYPE=grpo`, `GRPO_KL_COEF=0` — no teacher/reference model at all in v1 (biggest compute win; frees the GPU the old EMA/frozen-teacher logprob server used)
- Reward = reflector PASS/FAIL verdict (`megatron_trainer/reflector.py`), forced on for grpo regardless of `HINDSIGHT_FIELD`

# Finalized Knobs (v1 GRPO defaults, from `docs/megatron_trainer/grpo.md`)

- `GRPO_GROUPS=8`, `GRPO_ADV=mean` (r_i − group mean, no /σ)
- `GRPO_CLIP_LOW=0.2` / `GRPO_CLIP_HIGH=0.28` (DAPO clip-higher)
- `GRPO_OLD_LOGPS=vllm` (ratio vs rollout logprobs — clip + sequence-level TIS active)
- `GRPO_IS_C_MAX=3.0`
- `GRPO_LR`: `1e-6` for full FT (unused now), `1e-5` starting guess for LoRA (unvalidated — E046 used 3e-4 for SDFT-LoRA on a different model/loss, not directly transferable to GRPO); `GRPO_LR_WARMUP_STEPS=15` (linear warmup then constant, no decay)
- `GRPO_GRAD_CLIP=0.2`
- `GRPO_MASK_TRUNCATED=1` (never punish length-truncated completions)
- `GEN_TEMPERATURE=1.0`

# Still Tunable

## Active focus (this phase)
- `GRPO_FILTER_GROUPS` (dynamic sampling for degenerate groups) — not yet implemented, watch `grpo/frac_reward_zero_std` to see if it's needed
- `GRPO_ADV=zscore` — not yet implemented, documented as an experiment knob
- `GRPO_KL_COEF>0` (reference KL against EMA/frozen anchor) — not yet implemented; the mitigation if entropy collapse or reward hacking shows up with no distributional anchor
- `GRPO_GROUPS` (4 vs 8 vs 16), `GRAD_ACCUM_STEPS` (more unique prompts/step vs more groups/step)
- `LORA_DIM` (32 default, untested for GRPO) — rank bottleneck may limit exploration/quality vs full FT; fallback is full FT + 4+ trainer GPUs if LoRA underperforms
- `GRPO_LR` for LoRA mode specifically (`1e-5` starting guess, unvalidated)

## Parked (not active this phase)
- Everything from the SDFT loss-function surface (SFT anchor, per-token IS, JSD, self-normalized IS) — retired, see `EXPERIMENTS.log` E001–E046 for the full history if reviving distillation ever becomes relevant again.

# Datasets

- **k400 with context:** `subset_k400_subset_with_context.jsonl` (400 examples). Context is baked into the raw `prompt` field itself (retrieval block prepended to the question) — reaches the student's own rollout prompt, not just a teacher-side hint, so it matters for GRPO too (verified by diffing the plain vs with-context files: `prompt` differs, `enriched_user_response` does not).
- **Large with context:** `combined_dataset_train_sdft_with_context.jsonl` (5002 examples)
- Container path prefix: `/workspace/data/analyze_research/`
- Note: for GRPO, "one epoch" = one pass over **unique prompts** (`len(dataset)`), each expanded ×G — not one pass over `GRAD_ACCUM_STEPS`-sized batches of unique prompts like sdft. `steps_per_epoch = len(dataset) / (GRAD_ACCUM_STEPS/GRPO_GROUPS)`.

# Open Questions

1. Does GRPO beat the 0.73 reverse-KL ceiling on this data at all?
2. Reflector pass rate is ~0.65–0.74 historically (not 0.5) — how often are groups degenerate (all-pass/all-fail) at G=8? Calibrates whether `GRPO_FILTER_GROUPS` should default on.
3. Literature defaults (LR=1e-6, clip 0.2/0.28, G=8) are generic — none validated on this task's long-CoT thinking-mode completions yet.
4. With no distributional anchor (`GRPO_KL_COEF=0`), is there entropy collapse or reward hacking against the reflector over many steps?
5. Does the effect compound with context injection (already baked into the dataset) or is it redundant with it?
6. Does LoRA's rank-32 bottleneck limit GRPO's ability to shift behavior enough to beat baseline, vs full FT (untested — blocked by the FSDP memory ceiling at the trainer counts tried so far)?

# Operational Rules

- Branch: `ra/grpo-live` (off the old `ra/autoresearch-loop` tip, inherits SDFT history + context datasets).
- GPU layout for GRPO runs: same `train_full.sh GPU_START NUM_TRAINERS NUM_VLLM_GPUS` launcher; NUM_TRAINERS × GRPO_GROUPS must divide GRAD_ACCUM_STEPS. One GPU is still allocated to an (unused) logprob server in v1 — known inefficiency, not yet reclaimed.
- See root `GOAL.md` for launch scripts, polling, cleanup, and tmux commands (SDFT-phase content, still accurate for infra/ops).
