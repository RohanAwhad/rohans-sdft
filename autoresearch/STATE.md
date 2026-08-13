# Belief State — updated after E036

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
- LR=2e-5 is the only LR that scales with thinking mode (1e-5, 1.5e-5, 3e-5 all fail)
- 120b frozen teacher is essential (EMA self-distill fails at all LRs)
- temp=0.7 >> temp=1.0 for rollout quality
- GRAD_ACCUM=32 >> 16
- Cosine schedule prevents late-epoch collapse and is better than constant for 10-epoch runs
- Reflector fallback patch works reliably (~0.5-1.3% fallback rate)

### Moderate evidence
- IS_CAP=2.0 is best for 3-epoch runs, but IS_CAP=5.0 may be better for longer runs (smoother climb in E035 until collapse)
- online_feedback doesn't outperform enriched_user_response on k400 (tested at 3ep and 10ep)
- k400 dataset (400 examples) is too small — all configs plateau at 0.69-0.73 after epoch 3

### Refuted
- EMA teacher (E021, E024)
- temp=1.0 (E025)
- IS_CAP=1.0 (E027 — clips too aggressively)
- GRAD_ACCUM=16 (E028)
- LR=1.5e-5 (E029 — doesn't scale)
- Constant schedule for 10-epoch runs (E030 — zigzags, E035 — collapses)

## Current Failure Distribution

- **Plateau at 0.69-0.73:** 100% of 10-epoch k400 runs (E030, E033, E034, E035, E036)
- **Zigzag oscillation:** Most runs show ±2-3pp noise between adjacent epochs
- **Late-epoch collapse:** Constant schedule + IS_CAP=5.0 (E035)
- **Early large dataset signal:** Same range as k400 (E037 partial)

## Active Candidate Mechanisms

- **IS ratio systematic bias (ratio_mean ~0.97):** UNTESTED. vLLM processed logprobs inflate student logprobs, making all IS weights <1.0. This uniformly down-weights all gradients by ~3%.
- **IS clipping as plateau cause:** UNTESTED at scale. Non-IS runs (3, 4, 13) beat baseline, IS runs don't. But those runs used different configs too.
- **Dataset diversity bottleneck:** UNTESTED. k400 has only 400 examples — model may memorize/oscillate. Large dataset has 12.5x more.
- **Loss function shape (reverse KL oscillation):** UNTESTED. Pure reverse KL is mode-seeking and may oscillate between modes, causing the zigzag.
- **Per-token vs per-example loss normalization:** UNTESTED. Longer completions may dominate gradients.

## Highest-Value Unknowns

1. Does the large dataset break the plateau? (E037 testing, but with wrong config)
2. Would the locked config (cosine + online_fb + IS_CAP=5.0) perform differently on large dataset?
3. Is the IS ratio bias (0.97 mean) the real issue, not the cap?
4. Would loss function modifications help?

## Next Experiment

- Stop E037 (wrong config), relaunch with locked knobs: cosine, online_feedback, IS_CAP=5.0, on large dataset
- OR: let E037 finish since it's already running, then run the locked config as E038
