# SDFT Experiment Analysis Log

Analysis-only sessions on wandb runs + eval results for `analyze_deepresearch`.
No code changes were made during this session (all read-only: wandb API queries + eval JSON inspection). Contrast with `devlogs.md`, which tracks code/infra changes.

---

## 2026-08-10 — wandb archaeology, LR-sweep read, LR-schedule investigation

### Context
Project: `megatron_trainer/` SDFT training (gpt-oss-20b student, gpt-oss-120b teacher). Goal explored this session: read existing wandb runs + eval results to find patterns for fast hyperparameter/algorithm iteration on a small data subset before scaling to the full dataset.

### wandb access setup (gotcha)
- Self-hosted wandb server at `http://localhost:8080` (not cloud). Creds in `~/.netrc` (`machine localhost:8080`).
- **`WANDB_API_KEY` env var was set to a stale cloud key** and overrides `.netrc`, breaking auth against the local server (`AuthenticationError`). Fix: `unset WANDB_API_KEY` before using `wandb.Api()` so it falls back to `.netrc`.
- Entity: `rohanawhad`. Only project with runs: `analyze_deepresearch`.

### Run inventory (`rohanawhad/analyze_deepresearch`)
17 distinct run names (18 IDs — run_5 has two: one `failed` with no logged history, one `crashed` with history).

| run | id | state | created | dataset | steps/epoch |
|---|---|---|---|---|---|
| run_1 | 5zwepngo | crashed | 08-05 | ? (not queried) | 156 (2 ep: 156,312) |
| run_2 | 1oqxtgu1 | crashed | 08-05 | subset_k400_subset.jsonl | 12 (2 ep) |
| run_3 | gsdz7ze5 | finished | 08-05 | subset_k400_subset.jsonl | 12 (10 ep) |
| run_4 | pkm3wjch | crashed | 08-06 | subset_k400_subset.jsonl | 12 (41 ep, 492 steps) |
| run_5 | gogmsqbc | failed | 08-07 | ? | no epoch/number logged |
| run_5 | rtkepv7v | crashed | 08-07 | subset_k400_subset.jsonl | 12 (13 ep) |
| run_6 | dv17jqxe | crashed | 08-07 | subset_k400_subset.jsonl | 12 (9 ep) |
| run_7 | 6v7qg27s | crashed | 08-08 | subset_k400_subset.jsonl | 12 (4 ep) |
| run_8 | ewosdguh | failed | 08-08 | subset_k400_subset.jsonl | 12 (17 ep) |
| run_9 | 72r8cbbs | crashed | 08-08 | subset_k400_subset.jsonl | 12 (2 ep) |
| run_10 | 9bqmtl1l | crashed | 08-08 | ? | no epoch/number logged |
| run_11 | pvqme38x | crashed | 08-08 | subset_k400_subset.jsonl | 12 (13 ep) |
| run_12 | l4pc2pi4 | crashed | 08-08 | subset_k400_subset.jsonl | 12 (13 ep) |
| run_13 | y5o9cwje | finished | 08-09 | combined_dataset_train_sdft.jsonl | 156 (2 ep: 156,312) |
| run_14 | ntutv7qg | finished | 08-09 | combined_dataset_train_sdft.jsonl (assumed) | 156 (3 ep: 156,312,468) |
| run_15 | 662v261j | finished | 08-10 | subset_k400_subset.jsonl | 12 (3 ep: 12,24,36) |
| run_16 | 7ukm2wqg | finished | 08-10 | subset_k400_subset.jsonl | 12 (39-ep target, 468 steps) |
| run_17 | 0c5ygmll | finished | 08-10 | subset_k400_subset.jsonl | 12 (10-ep target, 120 steps) |

`subset_k400_subset.jsonl` ≈ 400 examples → `steps_per_epoch = len(dataset)//GRAD_ACCUM_STEPS` (`megatron_trainer/trainer.py:161`) = 12 with `GRAD_ACCUM_STEPS=32`. `combined_dataset_train_sdft.jsonl` → 156 steps/epoch (~5k examples).

There is no `MAX_STEPS` knob — only `NUM_EPOCHS` (`megatron_trainer/config.py:54`, default 10). Total steps = `steps_per_epoch * NUM_EPOCHS` (`trainer.py:165`).

### Eval harness (`maas-knowledge-eval/eval_results/analyze_deepresearch/`)
- Layout: `{run}/{epoch_N or step_N}/run_1.json`.
- JSON shape: `{config, results, summary}`.
  - `results`: list of 557 per-question dicts (`question`, `answer`, `metadata`, `retrieved_chunks`, `answer_hat`, `judge_results`, `majority_pass` [0/1]).
  - `summary`: `{total, passed, failed, accuracy, failure_modes, metadata_breakdown}`.
- 557 rows but only **500 unique question strings** (57 duplicates) — noted, not root-caused.
- Question order is **identical across all epoch files for a given run** (index-aligned) — enables paired testing (McNemar) without needing to match by question text.
- Eval dirs exist for: run_3, run_4, run_5(crashed one), run_8, run_9, run_11, run_12, run_13, run_14, run_15, run_16, run_17. **No eval dirs for run_1, run_2, run_6, run_7, run_10.**

### LR sweep (constant LR, k400 subset, 12 steps/epoch)
| run | lr | state |
|---|---|---|
| run_2 | 5e-06 | crashed |
| run_3 | 1e-05 | finished |
| run_4 | 2e-05 | crashed |
| run_5 (rtkepv7v) | 2e-05 | crashed |
| run_6 | 2e-05 | crashed |
| run_7 | 5e-05 | crashed |
| run_8 | 3e-05 | failed |
| run_9 | 3e-05 | crashed |
| run_11 | 3e-05 | crashed |
| run_12 | 1e-05 (cosine, total=180, warmup=18) | crashed |

Epoch 1-5 eval accuracy (constant-LR runs, where available):
| run | lr | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|---|
| run_3 | 1e-05 | 0.7487 | 0.7307 | 0.7397 | 0.7253 | — |
| run_4 | 2e-05 | 0.6912 | 0.6966 | 0.7145 | 0.7110 | 0.6984 |
| run_5 | 2e-05 | 0.6391 | 0.6661 | 0.6697 | 0.6948 | 0.6966 |
| run_8 | 3e-05 | 0.6086 | 0.6266 | 0.6553 | 0.6427 | 0.6391 |
| run_9 | 3e-05 | 0.5673 | 0.5781 | — | — | — |
| run_11 | 3e-05 | 0.4470 | 0.5530 | 0.5601 | 0.5655 | 0.5171 |
| run_12 | 1e-05 (cosine) | 0.7127 | 0.6912 | 0.7199 | 0.7092 | 0.7181 |

Raw takeaway: LR ranking (1e-05 > 2e-05 > 3e-05) visible by epoch 1, holds through epoch 4-5. 3e-05 band (run_8/9/11) clearly worse; run_11 shows visible late-run degradation (instability signature).

### Statistical significance methodology (how "is this signal or noise" was actually calculated)

**Naive / back-of-envelope (independent-samples approximation):**
- `SE_single = sqrt(p(1-p)/n)`, p≈0.7, n=557 → **1.94pp**
- `SE_diff = SE_single * sqrt(2)` (assumes 2 independent samples) → **2.75pp**
- 95% threshold ≈ `1.96 * SE_diff` → **~5.4pp**

**Correct version (McNemar's paired test)** — evals reuse the *same* 557/500 questions every checkpoint, so samples are correlated, not independent. Ran on run_3 epoch1 vs epoch2 (500 matched questions):
- pass→pass=335, fail→fail=108, pass→fail=32, fail→pass=25 (discordant n=57)
- `McNemar chi2 (corrected) = (|b-c|-1)^2/(b+c) = 0.63` → need >3.84 for p<0.05 → **not significant**
- `SE_paired = sqrt((b+c) - (b-c)^2/n) / n = 1.51pp`
- Real 95% threshold ≈ `1.96 * 1.51 ≈ 3.0pp`, **not 5pp** — the independent-samples formula overstates the true noise floor for this paired setup.

**Real-world confirmation (accidental repeat-eval on identical weights):** `run_15/epoch_3` and `run_15/step_36` are the *same checkpoint* (`trainer.py:530-531` epoch-end save and `trainer.py:501-503` `SAVE_EVERY`-triggered save both fire at optimizer_step=36), evaluated independently twice:
- accuracies: 0.6409 vs 0.6553 (**1.44pp gap on identical weights**) — matches the ~1.5pp paired-SE estimate almost exactly.
- Item-level noise is much larger than the aggregate suggests: **14.4% of individual questions flip pass/fail** between the two runs (85.6% agreement) — they mostly cancel out in the aggregate.
- Correct handling: **pool** repeat evals of the same checkpoint rather than treat as two trend points. Pooled: 722/1114 = **0.6481**. Redid ep1→ep3(pooled) test: diff=2.51pp, SE_diff=2.45pp, z=1.02 → still not significant (same conclusion, tighter evidence).

### Full epoch-over-epoch significance sweep (McNemar on every consecutive pair, all 7 constant/cosine sweep runs)
Caveat: consecutive-pair chi2 tests are **uncorrected for multiple comparisons** — e.g. run_4 has 40 such tests, so ~2 "significant" flags are expected by pure chance at α=0.05 even under the null. Cleaner test = first-vs-last / first-vs-best (fewer tests, more power):

| run | lr | #epochs | first | last | best (ep) | first→last | first→best |
|---|---|---|---|---|---|---|---|
| run_3 | 1e-05 | 5 (of 10) | 0.7487 | 0.7576 | 0.7576 (10) | +0.89pp, no | +0.89pp, no |
| run_4 | 2e-05 | 41 | 0.6912 | 0.7325 | 0.7540 (29) | +4.13pp, **YES** | +6.28pp, **YES** |
| run_5 | 2e-05 | 9 | 0.6391 | 0.6517 | 0.7325 (6) | +1.26pp, no | +9.34pp, **YES** |
| run_8 | 3e-05 | 16 | 0.6086 | 0.6266 | 0.6679 (11) | +1.80pp, no | +5.93pp, **YES** |
| run_9 | 3e-05 | 2 | 0.5673 | 0.5781 | 0.5781 (2) | +1.08pp, no | +1.08pp, no |
| run_11 | 3e-05 | 13 | 0.4470 | 0.4309 | 0.5673 (6) | -1.61pp, no | +12.03pp, **YES** |
| run_12 | 1e-05 (cosine) | 13 | 0.7127 | 0.7253 | 0.7289 (10) | +1.26pp, no | +1.62pp, no |

**Verdict:**
- **run_4 — only run with real, sustained improvement** (both tests significant). Genuine upward drift over 41 epochs despite noisy zigzag day-to-day.
- **run_5, run_8, run_11 — real transient peak, not a trend.** Statistically real high point mid-run, but doesn't hold: first→last not significant (run_5/8 fall back toward baseline); run_11 actively reverses and ends up worse than epoch 1 — clearest instability signature, consistent with lr=3e-05 being too high.
- **run_3, run_9, run_12 — flat, pure noise.** No significant change in either direction.

### Fast-iteration strategy discussion (small subset → scale up)

Constraints: full/large dataset max run so far = 468 steps (3 epochs, 156 steps/epoch, GRAD_ACCUM_STEPS=32, ~2-2.5k completion tokens backpropped/step). Small subset (k400) = 12 steps/epoch, same other stats.

**Q1: should small-scale runs set total steps to 468 (to cleanly capture the LR schedule's effect)?**
- No direct `MAX_STEPS` knob — hitting 468 steps on k400 (12 steps/epoch) means `NUM_EPOCHS=39` → 39 repeats of the same ~400 examples.
- Matching total steps matters only for **schedule shape** (cosine warmup/decay is a function of `total_train_steps`, `trainer.py:168-174`: `warmup_steps = min(int(0.1*total_train_steps), 100)`) — irrelevant to the constant-LR sweeps above (all had `lr_scheduler=None`).
- Structural tradeoff identified: matching **total steps** (39 epochs on 400 samples) trades in a **repetition confound** — real run does 3 epochs over ~5k *unique* samples; small run would do 39 passes over the *same* 400. Especially risky for on-policy SDFT (student generates its own rollouts each step) — repetition risks mode collapse / memorized completions, a dynamic absent from the real run.
- **You can't match both total-steps and epoch-count-regime simultaneously without growing the subset toward full size** (steps_per_epoch=156 needs ~5k samples — no longer "small").
- run_4 (2e-05, 41 epochs / 492 steps on k400) is effectively a prior version of this exact experiment: plateaus/oscillates 0.69-0.75 after ~epoch 10, no clear late-stage improvement from further repetition.
- **Recommendation:** don't force step-matching by over-repeating tiny data. Switch schedule-shape experiments to **warmup-stable-decay (WSD)** instead of cosine — decouples "how long will I train" from "what LR shape," letting peak-LR/warmup iteration happen at any step count; only fix decay/cooldown length once total budget is chosen for the real run.

**Q2: other ways to find fast-iteration patterns**
- LR-magnitude ranking stabilizes fast (visible by epoch 1, holds through epoch 4-5) — don't need >1-2 epochs to rank candidates by LR magnitude at constant LR.
- But early ranking ≠ stability — run_11 looked mid-pack at epoch 3-4 then degraded by epoch 13. Use short runs to eliminate bad candidates, but run the top candidate(s) longer (10+ epochs on small data) to catch late-stage collapse before committing big-run compute.
- Track eval accuracy **and** train loss together — on-policy reverse-KL can show decreasing loss while eval collapses (run_11 pattern); loss-only monitoring misses it.
- Respect the noise floor (see significance section above) — don't read tea leaves on 1-2pp moves; real threshold for this eval set (paired) is ~3pp, not smaller.
- Compare on step/samples-seen axis, not epoch index — "epoch 1" means 12 steps on the small subset vs 156 steps on the large set; not directly comparable.
- Consider a mid-size "bridge" subset (~2-3k samples) to confirm winners before full scale — fewer repeats needed to hit meaningful step counts, less repetition confound than the 400-sample set.
- Monitor generation diversity/repetition rate of student rollouts on small-set runs — early signal for mode collapse specific to on-policy self-distillation, ahead of eval-accuracy cratering.

### New runs: cosine-schedule total-step-budget sweep (run_15, run_16, run_17)
All three: same k400 subset, same model, **lr=3e-05 + cosine** (new axis vs. the constant-LR sweep above) — this is the team directly testing the Q1 schedule-matching question.

| run | state | total steps | warmup | design intent |
|---|---|---|---|---|
| run_15 | finished | 36 (3 ep) | 3 | epoch-matched to real run's epoch count (3) |
| run_16 | running | 468 (39 ep) | 46 | **step-matched to the real run's 468-step max** — literally the Q1 experiment |
| run_17 | running | 120 (10 ep) | 12 | third point on the budget spectrum |

Eval accuracy (epochs 1-3, all available so far):
| run | total steps | warmup | ep1 | ep2 | ep3 | 1→2 | 2→3 | 1→3 (net) |
|---|---|---|---|---|---|---|---|---|
| run_15 | 36 | 3 | 0.6732 | 0.6553 | 0.6409 (pooled w/ step_36: **0.6481**) | -1.79pp, no | -1.44pp, no | -3.23pp, no (pooled: -2.51pp, z=1.02, no) |
| run_17 | 120 | 12 | 0.6786 | 0.7056 | 0.6553 | +2.70pp, no | -5.03pp, **YES** | -2.33pp, no |
| run_16 | 468 | 46 | 0.6750 | 0.6858 | 0.6984 | +1.08pp, no | +1.26pp, no | +2.34pp, no |

Schedule position at epoch 3 (step 36):
- run_15: **100% decayed** (LR≈0) — fully past its own schedule.
- run_17: 30% through decay (past warmup=12).
- run_16: **still inside warmup** (36 < 46) — LR still ramping toward peak, hasn't hit peak yet.

**Observed pattern:** none of the three shows a statistically real net 1→3 trend individually (all within noise) — but there's a consistent cross-run ordering: the run with the most schedule-budget still ahead of it at step 36 (run_16, still warming up) has the highest ep3 accuracy (0.6984); the run with LR fully decayed by step 36 (run_15) has the lowest (0.6409-0.6481). run_17 sits in between.

**Caveat (not yet resolved):** this doesn't cleanly isolate "schedule shape" from "cumulative effective LR applied so far" — a run still in warmup has, by construction, been training at a *lower average LR* than one that already hit and decayed from peak. Can't separate those two effects with only 3 runs at 3 different total budgets. Would need matched-cumulative-LR comparisons to disentangle.

### Open items / next steps
- Keep polling run_16 (needs to clear warmup at step 46+ before its schedule-position tells us anything about decay behavior) and run_17 (3/10 epochs so far) as they progress; re-run McNemar/first-vs-last significance once more epochs land.
- Consider designing a run that holds cumulative LR×steps roughly constant while varying schedule shape, to separate "more effective LR so far" from "schedule shape" in the run_15/16/17 comparison.
- No eval data exists for run_2, run_6, run_7 (never evaluated) or run_1, run_10 (no training-step history logged) — would need to check training logs directly if those are needed later.
- 557-vs-500-unique-question duplication in the eval set is unexplained — worth a quick look at the eval dataset generation if it matters for future analysis.
- WSD (warmup-stable-decay) scheduler not yet implemented in `megatron_trainer` — currently only `constant` (`lr_scheduler=None`) and `cosine` (`trainer.py:168-178`) exist. Would need adding if the team wants to act on the Q1 recommendation.
