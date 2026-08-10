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
- Item-level flip rate ~14% even on identical weights — aggregate looks stable but individual questions are noisy

## Knobs to Sweep
| Knob | Range | Notes |
|---|---|---|
| `LEARNING_RATE` | 1e-5, 2e-5, 3e-5 | 1e-5 best in prior constant-LR sweep; never tested with thinking+IS |
| `LR_SCHEDULER` | constant, cosine | cosine warmup = min(10% total steps, 100) |
| `NUM_EPOCHS` | controls cosine schedule shape | 3 epochs = 36 total steps on k400 |
| `TEACHER_MODEL_PATH` | `""` (EMA self-distill) or `openai/gpt-oss-120b` (frozen) | frozen teacher disables weight sync |
| `EMA_ALPHA` | 0.01, 0.05, ... | only relevant when no frozen teacher |
| `IS_CAP` | 1.5, 2.0, 3.0 | importance sampling truncation cap |
| `GRAD_ACCUM_STEPS` | 16, 32 | 16 → 24 steps/epoch on k400 |
| `GEN_TEMPERATURE` | 0.7, 1.0 | 0.7 used in runs 1-15; 1.0 in runs 16-18 |
| `GEN_TOP_P` | 0.95, 1.0 | 0.95 used in runs 1-15; 1.0 in runs 16-18 |

## Fixed Settings
- Model: `unsloth/gpt-oss-20b-BF16`
- Dataset: `subset_k400_subset.jsonl`
- `IS_WEIGHTING=1` (always on)
- `STUDENT_THINKING=1` (always on)
- `THINKING_BUDGET=512`
- `MAX_GRAD_NORM=1.0`
- `TRAINER_BACKEND=fsdp`
- GPU layout: `train_full.sh 0 4 2`

## Elimination Criteria (per 3-epoch run)
- **Eliminate:** epoch 1 accuracy catastrophically low (<0.55), accuracy degrading ep1→ep3, or accuracy collapsing mid-run
- **Survives:** flat-or-improving trend, epoch 1 not blown up
- **Note:** can't reliably rank survivors within noise — Phase 1 is elimination, not ranking

## Prior Results Summary
| Run | LR | Schedule | IS | EMA | Thinking | Best Acc | Trend |
|---|---|---|---|---|---|---|---|
| run_3 | 1e-5 | constant | no | 0.05 | no | 0.7576 (ep10) | flat, within noise of baseline |
| run_12 | 1e-5 | cosine(180) | no | 0.05 | yes | 0.7289 (ep10) | flat |
| run_4 | 2e-5 | constant | no | 0.05 | no | 0.7540 (ep29) | slow climb, peaked late |
| run_5 | 2e-5 | constant | no | 0.05 | no | 0.7325 (ep6) | transient peak, fell back |
| run_8 | 3e-5 | constant | no | 0.05 | no | 0.6679 (ep11) | poor, never close to baseline |
| run_11 | 3e-5 | constant | no | 0.05 | no | 0.5673 (ep6) | collapsed |
| run_13 | 1e-5 | cosine | no | 0.05 | yes | 0.7415 (step100) | matched baseline, full dataset |
| run_14 | 3e-5 | cosine | no | 0.05 | yes | 0.7253 (ep3/step300) | slow recovery, full dataset |
| run_15 | 3e-5 | cosine(36) | no | 0.05 | yes | 0.6732 (ep1) | decayed, LR fully spent by ep3 |
| run_16 | 3e-5 | cosine(468) | no | 0.05 | yes | 0.6984 (ep3) | slight upward (still in warmup) |
| run_17 | 3e-5 | cosine(120) | yes | 0.05 | yes | 0.7056 (ep2) | up then down |
| run_18 | 3e-5 | cosine(120) | yes | 0.01 | yes | 0.6391 (ep1) | in progress |
| baseline | — | — | — | — | yes | 0.7415 | — |

## Monitoring a Run
```bash
bash poll_eval.sh <run_name> <epoch>
# e.g. bash poll_eval.sh run_18 epoch_3
```
Polls every 30s for the eval result file. Once found, prints accuracy and exits. Then kill the training container:
```bash
podman stop sdft-megatron-train
```

## Phase 2 (later)
Take surviving settings from Phase 1 and run longer (10+ epochs on k400, or full dataset ~5k examples). Goal: find a setting that beats baseline by >3pp with statistical significance.
