# Goal

Beat the baseline accuracy of **0.7415** (with thinking) on the analyze_deepresearch eval (557 questions, 500 unique, majority-vote judge via Claude Opus 4) using SDFT (Self-Distillation Fine-Tuning) with reverse-KL loss on a 20B GPT-OSS student model.

# Problem Statement

The 20B student model fine-tuned via on-policy reverse-KL distillation from a frozen 120B teacher plateaus at 0.69-0.73 accuracy over 10 epochs on a 400-example subset (k400). No configuration tested so far has beaten the baseline with statistical significance (≥3pp above 0.7415). The core question is what combination of training signal, loss shaping, and schedule can push accuracy past baseline.

# Baseline

- **With thinking:** 0.7415
- **Without thinking:** 0.7397
- Eval: 557 questions (~500 unique), majority-vote judge via Claude Opus 4
- Noise floor: paired SE ≈ 1.5pp (McNemar), 95% significance threshold ≈ 3pp

# What to Optimize

- **Primary metric:** accuracy on analyze_deepresearch eval
- **Target:** >0.7415 (beat baseline), ideally >0.7715 (statistically significant improvement)
- **Secondary signal:** epoch-over-epoch trajectory (sustained improvement vs zigzag plateau)

# How to Evaluate

- Eval runs automatically via `auto_eval_poller.sh` on every checkpoint
- Per-checkpoint accuracy at `/home/rohan/1_Projects/maas-knowledge-eval/eval_results/analyze_deepresearch/{run_name}/{checkpoint}/run_1.json`
- Poll with: `bash poll_eval.sh <run_name> <epoch_or_step>`
- On k400: 12 steps/epoch, eval per epoch. On large dataset: ~156 steps/epoch, eval per epoch + every SAVE_EVERY steps.

# Constraints (do not change)

- Model: `unsloth/gpt-oss-20b-BF16`
- `IS_WEIGHTING=1` (always on)
- `STUDENT_THINKING=1` (always on)
- `THINKING_BUDGET=512`
- `MAX_GRAD_NORM=1.0`
- `TRAINER_BACKEND=fsdp`
- GPU layout: `train_full.sh 0 4 2`
- Teacher: `openai/gpt-oss-120b` (frozen)
- vLLM logprobs mode: `--logprobs-mode processed_logprobs` only (raw_logprobs not allowed)
- `IS_CAP=5.0` (fixed)

# Finalized Knobs (locked by standup 2026-08-12)

- `LEARNING_RATE=2e-5`
- `LR_SCHEDULER=cosine`
- `IS_CAP=5.0`
- `HINDSIGHT_FIELD=online_feedback`
- `REFLECTOR_PROJECT_ID=itpc-gcp-ai-eng-claude`

# Still Tunable

## Active focus (this phase)
- **Loss function** (`megatron_trainer/chunked_head.py`) — primary experimental surface (reverse-KL base, no direction swap):
  - SFT anchor term: reverse-KL + λ·NLL on golden answer (hybrid, anti-collapse)
  - Per-token IS: apply ratio per-token, not per-sequence scalar (keeps IS_CAP=5.0 + processed_logprobs)
  - Self-normalized IS weights: normalize across batch (keeps IS_CAP=5.0 + processed_logprobs)
  - Alternative divergences: JSD / α-divergence (mass-covering, bounded; needs new backward)
  - Length normalization: per-completion mean (current) vs per-token batch mean vs unnormalized
- **Reflector prompt** (`megatron_trainer/reflector.py`) — system/user template, model

## Parked (tunable, not active this phase)
- `GEN_TEMPERATURE` (0.7, 1.0, 1.2)
- `GEN_TOP_P` (0.95, 1.0)
- Dataset (k400 subset, combined 5002 examples)
- `NUM_EPOCHS`
- `GRAD_ACCUM_STEPS` (32 is default)

# Datasets

- **k400:** `subset_k400_subset.jsonl` (400 examples, 12 steps/epoch at GA=32)
- **Large:** `combined_dataset_train_sdft.jsonl` (5002 examples, ~156 steps/epoch at GA=32)
- Container path prefix: `/workspace/data/analyze_research/`

# Open Questions

1. Why do all IS-enabled runs plateau at 0.69-0.73 while non-IS runs (3, 4, 13) beat baseline?
2. IS ratio_mean is systematically ~0.97 (<1.0) due to processed logprobs from vLLM — is this uniform down-weighting the real issue, or is it the clipping?
3. Can the large dataset (12.5x more examples) break through the plateau that k400 can't?
4. Would loss function modifications (per-token normalization, SFT anchor) address the zigzag pattern?
5. Does online_feedback's richer per-step signal help at scale (large dataset) even though it didn't on k400?

# Operational Rules

See root `GOAL.md` for launch scripts, polling, cleanup, and tmux commands.
