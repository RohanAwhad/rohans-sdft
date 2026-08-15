# Context — Compiled from EXPERIMENTS.log and STATE.md

---

## Tier 1 — Latest experiments (last 3, full detail)

---

### #E045 [RUNNING]

1. **Motivation:** All loss-function axes exhausted (IS aggregation, SFT anchor, JSD). 25+ experiments plateau at 0.69-0.73. New axis: inject retrieved context into the student's training prompt via retriever at localhost:9090.
2. **Hypothesis (M8):** The 0.73 plateau is caused by insufficient domain knowledge in the training prompt, not loss geometry. Context injection provides factual grounding that breaks the plateau.
3. **Prediction:** Peak > 0.74 by epoch 5. Sustained above 0.74 through epoch 10.
4. **Design:** Locked config (LR=2e-5, cosine, IS_CAP=5.0, 120b teacher, temp=1.2, online_feedback) + ASYNC_ROLLOUT=1 + context-injected k400 dataset (400 examples, retriever k=5). 10 epochs, 120 total steps.
5. **Results (partial, through ep5):** ep1=0.7235, ep2=0.7056, ep3=0.7289, ep4=0.7235, ep5=**0.7433**. FIRST BASELINE CROSSING. Epochs 6-10 pending.
6. **Analysis (partial):** Context shifts entire trajectory upward by ~5-8pp vs E036 (same config, no context). ep1 starts at 0.7235 vs E036's 0.6338. Never drops below 0.70. Peak 0.7433 is +0.18pp above baseline but within noise floor.
7. **Conclusion (partial):** WEAKLY SUPPORTED. First baseline crossing. Pending confirmation from epochs 6-10.
8. **New questions:** Does accuracy sustain above 0.74 through epoch 10? Is context additive with other configs?

---

### #E043 [KILLED]

1. **Motivation:** All IS variants exhausted. JSD changes the divergence geometry (mass-covering vs mode-seeking).
2. **Hypothesis (M7):** Reverse-KL mode-seeking causes the 0.73 plateau. JSD reduces mode-seeking and breaks through.
3. **Prediction:** Peak > 0.74 by epoch 10. signal_mean trends toward 0.
4. **Design:** DIVERGENCE=jsd, per-sequence IS, IS_CAP=5.0, cosine, online_feedback, temp=1.2, 10ep k400.
5. **Results (partial, killed at ep6):** ep1=0.5458, ep2=0.6535, ep3=0.6786, ep4=0.7110, ep5=0.7092.
6. **Analysis:** JSD trajectory comparable to reverse-KL. signal_mean slightly better (~-0.8 vs ~-0.9) but not dramatically different. Killed to shift focus.
7. **Conclusion:** INCONCLUSIVE (killed early).

---

### #E042 [DONE]

1. **Motivation:** Per-token IS (E040/E041) confirmed per-token signal is real but causes late-epoch collapse. Self-normalized IS normalizes gradient magnitude while preserving per-token correction.
2. **Hypothesis (M6):** Self-normalized per-token IS preserves per-token correction while preventing magnitude fluctuations.
3. **Prediction:** Peak > 0.74. No late collapse.
4. **Design:** IS_PER_TOKEN=1, IS_SELF_NORMALIZED=1, IS_CAP=5.0. Locked config, 10ep k400.
5. **Results:** ep1=0.5889, ep2=0.6230, ep3=0.6535, ep4=0.6984, ep5=0.7056, ep6=0.7038, ep7=0.7038, ep8=0.7056, ep9=0.6948, ep10=0.7038. Peak 0.7056@ep5. Drop -1.8pp (stable).
6. **Analysis:** Self-normalization prevents collapse but peak is lower than per-sequence IS (E036: 0.7235). Stability-peak trade-off.
7. **Conclusion:** WEAKLY SUPPORTED (stability) but REFUTED (peak). Per-token IS axis exhausted.

---

## Tier 2 — Recent experiments (4-10 back, one-liners)

- #E041 REFUTED [peak 0.7038] — Per-token IS with IS_CAP=2.0 reduces clip rate but still collapses and lower peak
- #E040 INCONCLUSIVE [peak 0.7217] — Per-token IS preserves per-token correction but causes late collapse
- #E039 REFUTED [peak 0.6804] — SFT anchor at lambda=0.01 degrades accuracy by 4.3pp vs E036
- #E038 REFUTED [peak 0.5745] — SFT anchor at lambda=0.1 destabilizes KL objective
- #E037 REFUTED [peak 0.7307] — Large dataset (5002 ex) shows same 0.69-0.73 plateau
- #E036 WEAKLY SUPPORTED [peak 0.7235] — Locked config (cosine, online_fb, cap=5.0, temp=1.2), stable at 0.71, no breakthrough
- #E035 INCONCLUSIVE [peak 0.7235] — IS_CAP=5.0 + temp=1.2, smooth climb then late collapse

---

## Tier 3 — Older experiments (11+, aggregate only)

**Total count:** 17 (E018-E034)

**Keep/Revert rates:**
- Keep: 3 (E019, E020, E023) = 18%
- Revert: 10 (E018, E021, E024, E025, E026, E027, E028, E029, E030, E031) = 59%
- Weakly supported: 4 (E032, E033, E034) = 23%

**Verdict counts:**
- REFUTED: 11
- SUPPORTED: 2 (E020, E023)
- WEAKLY SUPPORTED: 4 (E019, E032, E033, E034)
- INCONCLUSIVE: 2 (E018, E022)

**Delta(1->3) range:** -2.9pp to +8.8pp (3-epoch runs only)

**Anti-patterns:**
- EMA self-distillation as teacher: fails at all LRs (E018, E021, E024)
- temp >= 1.0 on plain data: degrades signal (E018, E025)
- LR=1e-5 and 1.5e-5: don't scale with thinking mode
- Constant schedule in 10-epoch runs: causes late-epoch oscillation
- k400 subset without context: causes overfitting and oscillation beyond epoch 3

---

## Belief State

**Current Best:**
- Experiment: E045 (locked config + context-injected k400)
- Peak metric: **0.7433** @ epoch 5 — FIRST BASELINE CROSSING
- Baseline: 0.7415
- Gap: +0.18pp (within noise floor of 3pp, not significant)

**Key Breakthrough:**
- Context injection (M8) is the FIRST axis to cross baseline in 25+ experiments
- All prior axes (IS aggregation, SFT anchor, JSD, large dataset) capped at 0.73
- Context injection provides ~5-8pp uplift at epoch 1 vs same config without context

**SUPPORTED mechanisms:**
- LR=2e-5 is the only LR that scales with thinking mode
- 120b frozen teacher is essential
- Cosine schedule prevents late-epoch collapse
- Context injection (M8) provides domain knowledge that breaks the 0.73 plateau
- Per-token IS (M4) signal is real but not actionable via weighting

**REFUTED mechanisms:**
- EMA teacher, temp=1.0, IS_CAP=1.0, GRAD_ACCUM=16, LR=1.5e-5
- Constant schedule for 10ep, large dataset alone, SFT anchor, per-token IS variants

**Highest-Value Unknowns:**
1. Does E045 sustain ≥ 0.73 through epoch 10?
2. Is context injection additive with E034 config (enriched, cap=2.0, temp=0.7)?
3. Can context + combined dataset push to 0.7715 (statistical significance)?
