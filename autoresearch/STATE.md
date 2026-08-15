# Belief State -- updated after E045 (partial, ep1-5)

## Current Best

- **Experiment:** E045 (locked config + context-injected k400, async rollout)
- **Peak metric:** 0.7433 @ epoch 5 — **FIRST BASELINE CROSSING**
- **Baseline:** 0.7415
- **Gap:** +0.18pp above baseline (within noise floor of 3pp)

## Remaining Gap

- Baseline crossed (+0.18pp) but NOT statistically significant (need +3pp = 0.7715)
- Must sustain or improve through epochs 6-10

## What We Currently Believe

### Strong evidence
- LR=2e-5 is the only LR that scales with thinking mode
- 120b frozen teacher is essential (EMA self-distill fails at all LRs)
- Cosine schedule prevents late-epoch collapse
- Data scale alone does NOT break the plateau (E037 large dataset = same range)
- SFT anchor term (M5) REFUTED — NLL gradient competes with and degrades reverse-KL
- Per-token IS (M4) CONFIRMED — signal is real but not actionable via weighting
- All IS aggregation variants plateau at 0.70-0.73 on plain data
- **Context injection (M8) provides ~5-8pp uplift vs same config without context**
- **Context injection produced the first baseline crossing (0.7433) in 25+ experiments**

### Moderate evidence
- The 0.72-0.73 plateau on plain data is robust across ALL loss-function variants
- **The plateau is likely an input-signal problem, not a loss-geometry problem** — E045 uses the same loss as E036 but crosses baseline
- Context injection prevents early-epoch dips (ep1=0.7235 vs typical 0.63-0.68)

### Refuted
- EMA teacher (E021, E024)
- temp=1.0 (E025)
- IS_CAP=1.0 (E027)
- GRAD_ACCUM=16 (E028)
- LR=1.5e-5 (E029)
- Constant schedule for 10-epoch runs (E030, E035)
- Large dataset alone breaks plateau (E037)
- SFT anchor term (E038, E039)
- Per-token IS as a plateau breaker (E040, E041, E042)

## Current Failure Distribution

- **IS plateau on plain data (0.69-0.73):** 100% of IS-enabled runs without context injection
- **Context-injected data:** 1 run (E045), peak 0.7433, pending confirmation
- **Late-epoch collapse:** TBD for E045 (epochs 6-10 pending)
- **Zigzag oscillation:** E045 shows reduced oscillation (stays above 0.70)

## Active Candidate Mechanisms

- **M1: Loss/IS mechanism is the bottleneck:** WEAKENED. E045 crossed baseline with same loss (reverse-KL + IS). The loss may impose a secondary ceiling above 0.74, but the primary bottleneck was input signal.
- **M3: IS mechanism itself caps performance:** WEAKENED. E045 crossed 0.7415 with IS_WEIGHTING=1. IS doesn't hard-cap at 0.73.
- **M7: Reverse-KL mode-seeking causes plateau:** INCONCLUSIVE. E043 killed early. Could revisit with context injection.
- **M8: Context injection provides missing domain knowledge:** SUPPORTED. First baseline crossing. Pending epochs 6-10 confirmation.

## Highest-Value Unknowns

1. **Does E045 sustain ≥ 0.73 through epoch 10?** (epochs 6-10 pending, training in progress)
2. **Is context injection additive with E034 config?** (E034: enriched, IS_CAP=2.0, temp=0.7 — prior best peak 0.7289 without context)
3. **Can context + combined dataset push to 0.7715 (statistical significance)?**
4. **Does the student internalize context knowledge or just pattern-match on the format?**

## Next Experiments

- **E045:** Complete epochs 6-10 (in progress, step 84/120)
- **E046:** E034 config (cosine, enriched, IS_CAP=2.0, temp=0.7) + context-injected k400 (launch script: `/tmp/launch_run_46.sh`)
- **E047:** E023 config (constant, enriched, IS_CAP=2.0, temp=0.7) + context-injected k400 (launch script: `/tmp/launch_run_47.sh`)
