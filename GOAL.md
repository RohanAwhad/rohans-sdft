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

## Experiment Log (Phase 1)

### run_18 — LR=3e-5, cosine(120), IS on, EMA=0.01, self-distill, temp=1.0
- ep1=0.6391, ep2=0.6194, ep3=0.6822, Δ(1→3)=+4.3pp
- Upward trend but starts too low. EMA=0.01 + self-distillation + high temp not promising.

### run_19 — LR=1e-5, cosine(120), IS on, 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.6391, ep2=0.6930, ep3=0.7056, Δ(1→3)=+6.7pp
- Strongest 3-epoch climb so far. But ep1 suspiciously low — cosine warmup=12 means entire first epoch trained at ramping LR. The +6.7pp may partly be "getting past warmup" not "real improvement."

### run_20 — LR=1e-5, constant, IS on, 120b frozen teacher, temp=0.7, top_p=0.95
- **Hypothesis:** Same as run_19 but constant LR — isolates cosine warmup effect. If ep1 jumps to ~0.74 (like run_3), confirms warmup is the ep1 drag.
- Status: in progress

### run_20 — LR=1e-5, constant, IS on, 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.7056, ep2=0.7127, ep3=0.7038, Δ(1→3)=-0.2pp
- Flat. Confirmed cosine warmup was dragging run_19's ep1 — both converge to ~0.705 by ep3.
- But ep1=0.7056 is ~4pp below run_3's 0.7487 (1e-5 constant, no IS, no thinking, EMA). Something else is hurting.

### run_21 — LR=1e-5, constant, IS on, EMA(0.05) self-distill, temp=0.7, top_p=0.95
- ep1=0.6912, ep2=0.6930, ep3=0.6912, Δ(1→3)=0pp
- Dead flat, worse than run_20 (120b). EMA teacher worse than 120b at this config.
- run_3 (same LR/sched but no IS, no thinking) hit 0.7487 ep1. IS weighting is the remaining suspect.

### run_22 — LR=1e-5, constant, IS OFF, 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.6912, ep2=0.6822, ep3=0.6894, Δ(1→3)=-0.2pp
- Flat. IS on/off makes no difference at 1e-5. The ~4-5pp gap vs run_3 is entirely from STUDENT_THINKING=1 (the only remaining difference).
- **Key finding:** with thinking on, 1e-5 constant plateaus at ~0.69-0.71 regardless of IS or teacher type. Need higher LR to get signal with thinking-mode rollouts.

### run_23 — LR=2e-5, constant, IS on, 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.6230, ep2=0.6966, ep3=0.7110, Δ(1→3)=+8.8pp
- **Strongest 3-epoch climb yet.** Ep1 starts low but trajectory is steep. By ep3 already surpasses run_20's plateau. This is the scaling signal we want.
- Confirms: thinking mode needs higher LR (2e-5) to learn effectively. 1e-5 plateaus at ~0.70.

### run_24 — LR=2e-5, constant, IS on, EMA(0.05) self-distill, temp=0.7, top_p=0.95
- ep1=0.6786, ep2=0.6715, ep3=0.6643, Δ(1→3)=-1.4pp
- Declining. EMA teacher can't provide stable signal at 2e-5 — teacher drifts with student.
- **Key finding:** 120b frozen teacher is critical at 2e-5. EMA works at 1e-5 (flat) but fails at 2e-5 (declining).

### run_25 — LR=2e-5, constant, IS on, 120b frozen teacher, temp=1.0, top_p=1.0
- ep1=0.6876, ep2=0.6930, ep3=0.6589, Δ(1→3)=-2.9pp
- Declining. temp=1.0 hurts — noisier rollouts degrade learning at 2e-5.
- **Key finding:** temp=0.7 >> temp=1.0 at the winning config.

### run_26 — LR=2e-5, constant, IS on (cap=5.0), 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.6230, ep2=0.6481, ep3=0.6553, Δ(1→3)=+3.2pp
- Much weaker than run_23 (cap=2.0). Higher cap adds gradient variance, slows learning.
- **Key finding:** IS_CAP=2.0 >> IS_CAP=5.0.

### run_27 — LR=2e-5, constant, IS on (cap=1.0), 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.6643, ep2=0.6858, ep3=0.6804, Δ(1→3)=+1.6pp
- Weak, flattening. Cap=1.0 (effectively no correction) underperforms cap=2.0.
- **IS_CAP sweep: 1.0 (+1.6pp) < 5.0 (+3.2pp) << 2.0 (+8.8pp). Sweet spot confirmed at 2.0.**

### run_28 — LR=2e-5, constant, IS on (cap=2.0), 120b frozen teacher, temp=0.7, GRAD_ACCUM=16
- ep1=0.6697, ep2=0.6804, ep3=0.6948, Δ(1→3)=+2.5pp
- Weaker than run_23 (GRAD_ACCUM=32). Noisier gradients from smaller batch hurt more than doubled update frequency helps.
- **GRAD_ACCUM=32 >> 16.**

### run_29 — LR=1.5e-5, constant, IS on (cap=2.0), 120b frozen teacher, temp=0.7, top_p=0.95
- ep1=0.6894, ep2=0.7092, ep3=0.6948, Δ(1→3)=+0.5pp
- Flat. 1.5e-5 behaves like 1e-5 — plateaus immediately. Only 2e-5 has upward trajectory.
- **LR sweep: 1e-5 (flat) ≈ 1.5e-5 (flat) << 2e-5 (climbing). 2e-5 is the sweet spot for thinking mode.**

### run_30 — LR=2e-5, constant, IS on (cap=2.0), 120b frozen teacher, temp=0.7, top_p=0.95 — 10 EPOCH VALIDATION
- **Purpose:** Phase 2 preview. Exact same config as run_23 but let it run full 10 epochs. Does the +8.8pp/3ep trajectory continue? Does it reach/beat baseline (0.7415)?
- Status: in progress (polling epoch_10)

## Phase 2 (later)
Take surviving settings from Phase 1 and run longer (10+ epochs on k400, or full dataset ~5k examples). Goal: find a setting that beats baseline by >3pp with statistical significance.
