# Belief State -- updated after E042

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
- Cosine schedule prevents late-epoch collapse
- Data scale alone does NOT break the plateau (E037 large dataset = same range)
- SFT anchor term (M5) REFUTED -- NLL gradient competes with and degrades reverse-KL distillation at both lambda=0.1 and lambda=0.01
- Per-token IS (M4) CONFIRMED -- per-token ratios vary wildly (0.006-5.0, std 0.19-1.54) and per-sequence averaging was losing real signal
- But raw per-token IS causes late-epoch collapse (E040: -2.9pp, E041: -3.2pp)
- Self-normalized IS prevents collapse (E042: only -1.8pp) but doesn't improve peak

### Moderate evidence
- The 0.72-0.73 plateau is robust across ALL IS variants tested (per-sequence, per-token raw, per-token lower cap, per-token self-normalized)
- Per-sequence IS (E036) remains the best stable approach at 0.71-0.72
- Per-token IS accelerates early learning but peaks lower and/or collapses
- Self-normalized IS is the most stable per-token variant but peaks at 0.7056

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

- **IS plateau (0.69-0.73):** 100% of IS-enabled runs across all IS variants
- **Late-epoch collapse:** Only with raw per-token IS (E040, E041). Self-normalized (E042) and per-sequence (E036) are stable.
- **Zigzag oscillation:** Most runs show +/-2-3pp noise

## Active Candidate Mechanisms

- **M1: Loss/IS mechanism is the bottleneck:** SUPPORTED. No IS variant breaks the plateau.
- **M3: IS mechanism itself caps performance:** UNTESTED. Non-IS runs (3, 4, 13) beat baseline but confounded. IS_WEIGHTING=0 is locked.
- **M4: Per-token IS ratio variation lost by per-sequence averaging:** SUPPORTED (ratio_std confirmed). But exploiting it (per-token IS) doesn't improve peak.
- **M5: SFT anchor provides direct gradient toward correct tokens:** REFUTED.
- **M6: Self-normalized IS stabilizes per-token gradients:** SUPPORTED (stability confirmed) but doesn't improve peak.

## Highest-Value Unknowns

1. Is IS itself the problem? Direct IS on/off at locked config (needs GOAL.md amendment -- IS_WEIGHTING=0 is locked)
2. Would JSD or alpha-divergence (mass-covering, bounded) work better than reverse-KL (mode-seeking)?
3. Would length normalization change the gradient distribution?
4. Is the 0.73 ceiling a fundamental limit of reverse-KL distillation on this task?

## Next Experiment

- **E043:** Run the autoresearch loop to decide. Options:
  - (a) IS on/off at locked config (requires GOAL.md amendment)
  - (b) JSD/alpha-divergence (new backward, bigger implementation)
  - (c) Length normalization (simpler, untested)
  - (d) Accept the plateau and optimize for stability (E036 config is most stable at 0.71)
