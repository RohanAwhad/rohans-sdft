# Phase 1: Hyperparameter Elimination via 3-Epoch k400 Runs

## Objective
Find hyperparameter settings that show promising scaling behavior (upward accuracy trajectory over epochs) by running fast 3-epoch experiments on the k400 subset (~400 examples, 12 steps/epoch). Eliminate settings that collapse, degrade, or go flat. Survivors advance to Phase 2 (longer runs, full dataset).

## Baseline
- With thinking: **0.7415**
- Without thinking: **0.7397**
- No run has beaten baseline with statistical significance yet.

## Noise Floor
- Paired SE ≈ 1.5pp (McNemar on 557 questions, ~500 unique)
- 95% significance threshold ≈ **3pp**
- Don't read signal into <3pp differences between runs or epochs

## Knobs to Sweep
| Knob | Range | Notes |
|---|---|---|
| `LEARNING_RATE` | 1e-5, 1.5e-5, 2e-5, 3e-5 | 2e-5 is the sweet spot for thinking mode |
| `LR_SCHEDULER` | constant, cosine | cosine warmup = min(10% total steps, 100) |
| `NUM_EPOCHS` | controls cosine schedule shape | 3 epochs = 36 total steps on k400 |
| `TEACHER_MODEL_PATH` | `""` (EMA self-distill) or `openai/gpt-oss-120b` (frozen) | frozen teacher disables weight sync |
| `EMA_ALPHA` | 0.01, 0.05, ... | only relevant when no frozen teacher |
| `IS_CAP` | 1.0, 2.0, 5.0 | importance sampling truncation cap |
| `GRAD_ACCUM_STEPS` | 16, 32 | 16 → 24 steps/epoch on k400 |
| `GEN_TEMPERATURE` | 0.7, 1.0 | 0.7 is better |
| `GEN_TOP_P` | 0.95, 1.0 | 0.95 paired with temp=0.7 |
| `HINDSIGHT_FIELD` | enriched_user_response, online_feedback | online_feedback = reflector LLM feedback + golden chunk + golden answer |
| Reflector prompt | code change in `megatron_trainer/reflector.py` | REFLECTOR_SYSTEM_PROMPT + REFLECTOR_USER_TEMPLATE. Model is fixed (claude-sonnet-4-6). Commit each prompt change before running. |

## Fixed Settings
- Model: `unsloth/gpt-oss-20b-BF16`
- Dataset: `subset_k400_subset.jsonl` (TRAIN_DATA_PATH=/workspace/data/analyze_research/subset_k400_subset.jsonl)
- `IS_WEIGHTING=1` (always on)
- `STUDENT_THINKING=1` (always on)
- `THINKING_BUDGET=512`
- `MAX_GRAD_NORM=1.0`
- `TRAINER_BACKEND=fsdp`
- GPU layout: `train_full.sh 0 4 2`

## Operational Rules

### How to launch a run
```bash
WANDB_MODE=online \
WANDB_BASE_URL="http://localhost:8080" \
WANDB_API_KEY="local-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" \
WANDB_PROJECT=analyze_deepresearch \
WANDB_NAME=sdft_gptoss_20b_run_N \
MODEL_NAME=unsloth/gpt-oss-20b-BF16 \
STUDENT_THINKING=1 \
TRAINER_BACKEND=fsdp \
HINDSIGHT_FIELD=<enriched_user_response|online_feedback> \
REFLECTOR_PROJECT_ID=itpc-gcp-ai-eng-claude \
NUM_EPOCHS=<10 for 3ep runs, more for validation> \
GRAD_ACCUM_STEPS=32 \
SAVE_EVERY=50 \
LEARNING_RATE=<lr> \
LR_SCHEDULER=<constant|cosine> \
IS_WEIGHTING=1 \
IS_CAP=2.0 \
GEN_TEMPERATURE=0.7 \
GEN_TOP_P=0.95 \
TEACHER_MODEL_PATH=<""|openai/gpt-oss-120b> \
MAX_TOTAL_LEN=8192 \
GEN_MAX_NEW_TOKENS=4096 \
TEACHER_MAX_PROMPT_LEN=4096 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/mnt/nvme5n1/rohan_patched_ckpts/hf-cache \
OUTPUT_DIR=/mnt/nvme5n1/rawhad/analyze_deepresearch_ckpts/sdft_gptoss_20b_run_N \
TRAIN_DATA_PATH=/workspace/data/analyze_research/subset_k400_subset.jsonl \
bash megatron_trainer/train_full.sh 0 4 2 2>&1 | tee logs/analyze_deepresearch_run_N.log
```
Write the launch script to `/tmp/launch_run_N.sh`, then: `tmux send-keys -t sdft_megatron_bridge:2.0 "bash /tmp/launch_run_N.sh" Enter`

### How to monitor a run
```bash
bash poll_eval.sh <run_name> <epoch>
# e.g. bash poll_eval.sh run_31 epoch_3
```
Polls every 30s for the eval result at `/home/rohan/1_Projects/maas-knowledge-eval/eval_results/analyze_deepresearch/{run_name}/{epoch_N}/run_1.json`. Once found, prints accuracy and exits.

### How to stop a run
```bash
podman stop sdft-megatron-train
```

### How to read eval results
```python
python3 -c "
import json
base = '/home/rohan/1_Projects/maas-knowledge-eval/eval_results/analyze_deepresearch/run_N'
for ep in ['epoch_1', 'epoch_2', 'epoch_3']:
    f = f'{base}/{ep}/run_1.json'
    try:
        d = json.load(open(f))
        print(f'{ep}: {d[\"summary\"][\"accuracy\"]:.4f}')
    except: print(f'{ep}: not found')
"
```

### After each run
1. Read eval results for all epochs
2. Clean up checkpoints: `rm -rf /mnt/nvme5n1/rawhad/analyze_deepresearch_ckpts/sdft_gptoss_20b_run_N/`
3. Update GOAL.md experiment log
4. Decide next experiment based on results
5. Launch next run

### Every 10 runs
Run a 10-epoch validation of the best config to check sustained scaling.

### Reflector prompt changes
When changing the reflector prompt in `megatron_trainer/reflector.py`, commit the change before launching the run so we have a git log of each prompt version.

## Elimination Criteria (per 3-epoch run)
- **Eliminate:** epoch 1 accuracy catastrophically low (<0.55), accuracy degrading ep1→ep3, or accuracy collapsing mid-run
- **Survives:** flat-or-improving trend, epoch 1 not blown up
- **Note:** can't reliably rank survivors within noise — Phase 1 is elimination, not ranking

## Key Findings (runs 18-30, HINDSIGHT=enriched_user_response)

1. **LR=2e-5 is the only LR that scales with thinking mode.** 1e-5 and 1.5e-5 plateau at ~0.69-0.71.
2. **120b frozen teacher is essential.** EMA self-distillation declines at 2e-5 (run_24: -1.4pp).
3. **temp=0.7 >> temp=1.0** (run_25: -2.9pp vs run_23: +8.8pp).
4. **IS_CAP=2.0 is the sweet spot.** 1.0 (+1.6pp) < 5.0 (+3.2pp) << 2.0 (+8.8pp).
5. **GRAD_ACCUM=32 >> 16** (run_28: +2.5pp vs run_23: +8.8pp).
6. **Cosine warmup just delays convergence** — same ep3 as constant but ep1 dragged down.
7. **Best config (enriched_user_response): LR=2e-5, constant, 120b teacher, IS cap=2.0, temp=0.7, GRAD_ACCUM=32** (run_23: +8.8pp over 3 epochs).
8. **10-epoch validation (run_30):** plateaued at 0.69-0.71 over 8 epochs (ep9-10 lost to disk full). Never reached baseline. Zigzag pattern suggests overfitting on k400.

## Experiment Log

| Run | LR | Sched | IS_CAP | Teacher | Temp | Hindsight | ep1 | ep2 | ep3 | Δ(1→3) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | — | — | — | — | — | — | **0.7415** | — | — | — |
| 18 | 3e-5 | cosine | 2.0 | EMA(0.01) | 1.0 | enriched | 0.6391 | 0.6194 | 0.6822 | +4.3pp |
| 19 | 1e-5 | cosine | 2.0 | 120b | 0.7 | enriched | 0.6391 | 0.6930 | 0.7056 | +6.7pp |
| 20 | 1e-5 | const | 2.0 | 120b | 0.7 | enriched | 0.7056 | 0.7127 | 0.7038 | -0.2pp |
| 21 | 1e-5 | const | 2.0 | EMA | 0.7 | enriched | 0.6912 | 0.6930 | 0.6912 | 0pp |
| 22 | 1e-5 | const | off | 120b | 0.7 | enriched | 0.6912 | 0.6822 | 0.6894 | -0.2pp |
| **23** | **2e-5** | **const** | **2.0** | **120b** | **0.7** | **enriched** | 0.6230 | 0.6966 | **0.7110** | **+8.8pp** |
| 24 | 2e-5 | const | 2.0 | EMA | 0.7 | enriched | 0.6786 | 0.6715 | 0.6643 | -1.4pp |
| 25 | 2e-5 | const | 2.0 | 120b | 1.0 | enriched | 0.6876 | 0.6930 | 0.6589 | -2.9pp |
| 26 | 2e-5 | const | 5.0 | 120b | 0.7 | enriched | 0.6230 | 0.6481 | 0.6553 | +3.2pp |
| 27 | 2e-5 | const | 1.0 | 120b | 0.7 | enriched | 0.6643 | 0.6858 | 0.6804 | +1.6pp |
| 28 | 2e-5 | const | 2.0 | 120b | 0.7 | enriched | 0.6697 | 0.6804 | 0.6948 | +2.5pp (GA=16) |
| 29 | 1.5e-5 | const | 2.0 | 120b | 0.7 | enriched | 0.6894 | 0.7092 | 0.6948 | +0.5pp |
| 30 | 2e-5 | const | 2.0 | 120b | 0.7 | enriched | 0.6697→0.6984(ep8) | — | — | 10ep validation, plateaued |
| 31 | 2e-5 | const | 2.0 | 120b | 0.7 | **online_fb** | 0.6338 | 0.6750 | 0.6517 | +1.8pp (worse than enriched run_23) |
| 32 | 2e-5 | cosine | 2.0 | 120b | 0.7 | **online_fb** | 0.6607 | 0.6535 | 0.6948 | +3.4pp (cosine > constant for online_fb) |
| 33 | 2e-5 | const | 2.0 | 120b | 0.7 | **online_fb** | 0.6984 | 0.7092 | 0.6804 | 10ep: peak 0.7253@ep9, zigzag 0.67-0.73, no baseline beat |
| 34 | 2e-5 | cosine | 2.0 | 120b | 0.7 | enriched | 0.6804 | 0.6589 | 0.6804 | 10ep: peak 0.7289@ep7, zigzag 0.66-0.73, no baseline beat |
| 35 | 2e-5 | const | **5.0** | 120b | **1.2** | enriched | 0.6194 | 0.6230 | 0.6589 | 10ep: peak 0.7235@ep6-7, collapsed to 0.69 by ep10. Smooth climb but late-epoch overfit |
| 36 | 2e-5 | **cosine** | **5.0** | 120b | **1.2** | **online_fb** | ? | ? | ? | 10ep. Locked config. Cosine to prevent late collapse |

## Key Insight: IS Clipping May Be The Plateau Cause
- Runs WITHOUT IS (run_3, run_4, run_13) beat baseline 0.7415. ALL IS runs plateau at 0.69-0.73.
- IS ratio_mean ~0.97 across all IS runs — systematically <1.0 due to processed logprobs (vLLM --logprobs-mode processed_logprobs returns post-top-p renormalized logprobs, inflated vs raw training logprobs). Upper clamp rarely fires; real effect is uniform ~3% down-weighting of all loss.
- clip_rate: cap=1.0 → ~20%, cap=2.0 → ~2%, cap=5.0 → ~0.3%
- signal_mean (sdpo/signal_mean): run_4 (no IS) trended from -1.0 toward -0.4 over 500 steps. IS runs stay stuck at -0.8 to -1.0 for 120 steps. But at 120 steps run_4 also looked flat — breakthrough came after step 200+.

## Finalized Knobs (locked, do not change)
- `LEARNING_RATE=2e-5`
- `LR_SCHEDULER=cosine`
- `IS_CAP=5.0`
- `HINDSIGHT_FIELD=online_feedback`
- `REFLECTOR_PROJECT_ID=itpc-gcp-ai-eng-claude`

## Still Tunable
- `GEN_TEMPERATURE` (0.7, 1.0, 1.2)
- `GEN_TOP_P` (0.95, 1.0)
- Reflector prompt (`megatron_trainer/reflector.py`)
- Dataset (k400, combined)
- `NUM_EPOCHS`

## Decision Gate (after run 35)
- **run_35 peak ≥ 0.74** → go to large dataset with locked config (cosine + online_fb + IS_CAP=5.0 + temp TBD)
- **run_35 peak < 0.74** → run 36: locked config with temp=1.2, 10ep k400. Keep iterating (temp, top_p, reflector prompt) until a run crosses 0.74, then large dataset.
- **Deadline:** large dataset run must start by 2026-08-13 ~18:00 UTC
- **Fallback:** if large dataset runs with enriched don't beat baseline 0.7415, switch to online_feedback

## Current Status
- **run_35** done: IS_CAP=5.0+temp=1.2, constant, enriched. Peak 0.7235@ep6-7, collapsed to 0.69. Below 0.74 gate.
- **run_36** in progress: locked config (cosine+online_fb+IS_CAP=5.0+temp=1.2), 10ep k400. Cosine may prevent late collapse.
- Large dataset: combined_dataset_train_sdft.jsonl (5002 examples), SAVE_EVERY=50, TRAIN_DATA_PATH=/workspace/data/analyze_research/combined_dataset_train_sdft.jsonl
- Next run number: 36

## Phase 2 (later)
Take surviving settings from Phase 1 and run longer (10+ epochs on k400, or full dataset ~5k examples). Goal: find a setting that beats baseline by >3pp with statistical significance.
