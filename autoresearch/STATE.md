# Belief State -- updated after E040

## Current Best

- **Experiment:** E034 (cosine, enriched, IS_CAP=2.0, temp=0.7, 10ep k400)
- **Peak metric:** 0.7289 @ epoch 7
- **Baseline:** 0.7415
- **Gap:** -1.3pp (within noise floor of 3pp)

## Remaining Gap

- Need +1.3pp to match baseline, +4.3pp for statistical significance
- No run has crossed 0.7415

## What We Currently Believe

### Strong evidence
- LR=2e-5 is the only LR that scales with thinking mode
- 120b frozen teacher is essential (EMA self-distill fails at all LRs)
- temp=0.7 >> temp=1.0 for rollout quality
- GRAD_ACCUM=32 >> 16
- Cosine schedule prevents late-epoch collapse
- Data scale alone does NOT break the plateau (E037 large dataset = same range)
- SFT anchor term (M5) REFUTED -- NLL gradient competes with and degrades reverse-KL distillation at both lambda=0.1 and lambda=0.01

### Moderate evidence
- Per-token IS (M4) CONFIRMED -- per-token ratios vary wildly (0.006-5.0, std 0.44-1.54) and per-sequence averaging was losing real signal. ratio_mean is ~1.1-1.5 (not 0.97 as previously believed -- the per-sequence mean was biased low)
- But raw per-token IS doesn't improve peak accuracy (0.7217 vs 0.7235) and causes late-epoch collapse (noisier gradients from 12-15% clip rate)
- online_feedback doesn't outperform enriched_user_response on k400

### Refuted
- EMA teacher (E021, E024)
- temp=1.0 (E025)
- IS_CAP=1.0 (E027)
- GRAD_ACCUM=16 (E028)
- LR=1.5e-5 (E029)
- Constant schedule for 10-epoch runs (E030, E035)
- Large dataset alone breaks plateau (E037)
- SFT anchor term at lambda=0.1 (E038 -- KL destabilized) and lambda=0.01 (E039 -- accuracy 4.3pp worse)

## Current Failure Distribution

- **IS plateau (0.69-0.73):** 100% of IS-enabled runs, both k400 and large dataset
- **Late-epoch collapse with per-token IS:** E040 collapsed from 0.7217 to 0.6930 after ep5
- **Zigzag oscillation:** Most runs show +/-2-3pp noise
- **SFT anchor interference:** E038, E039 -- SFT NLL degrades KL distillation

## Active Candidate Mechanisms

- **M1: Loss/IS mechanism is the bottleneck, not data scale:** SUPPORTED
- **M3: IS mechanism itself caps performance:** UNTESTED (non-IS runs beat baseline but confounded)
- **M4: Per-token IS ratio variation lost by per-sequence averaging:** SUPPORTED (ratio_std 0.44-1.54, per-token range 0.006-5.0). But raw per-token IS is too noisy.
- **M5: SFT anchor provides direct gradient toward correct tokens:** REFUTED (competes with KL, degrades accuracy)
- **M6 (new): Per-token IS needs stabilization:** UNTESTED. Lower IS_CAP or self-normalized IS might preserve the per-token signal while reducing noise.

## Highest-Value Unknowns

1. Would per-token IS with lower IS_CAP (e.g. 2.0) reduce clip rate and stabilize late training?
2. Would self-normalized IS weights smooth the per-token signal?
3. Is IS itself the problem? (Direct IS on/off at locked config -- IS_WEIGHTING=0 is locked, needs GOAL.md amendment)
4. Would combining per-token IS with a lower temp (0.7 instead of 1.2) reduce rollout noise?

## Next Experiment

- **E041:** Per-token IS with IS_CAP=2.0 (lower cap to reduce clip rate from 12-15% to ~2-4%, stabilizing late training while preserving per-token correction). Same locked config otherwise. Pre-registered: peak > 0.74, no late collapse (ep8-10 within 2pp of peak).
- **Alternative E042:** Per-token IS with temp=0.7 (reduce rollout noise to compensate for noisier per-token gradients).
