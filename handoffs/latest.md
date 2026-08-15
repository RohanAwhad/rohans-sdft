# Handoff — 2026-08-14 ~23:50 UTC (standup 08/14-8pm)

## ADD

- **Commit current codebase** before any other action. All loss-function work (SFT anchor, per-token IS, self-normalized IS, JSD divergence, reflector fallback) must be committed.
- **Rebase on v0.2.0** of the library. This brings async rollout support — vLLM generates the next batch's rollouts while the current batch trains, decoupling generation from training. Same wall-clock, more gradient updates per epoch.
- **Launch one long training run** from the promising runs {E033, E036, E039, E040, E041, E042}:
  - Measure time per epoch with async rollout enabled.
  - Calculate how many epochs fit in a 5-hour window.
  - Take min(epochs_in_5hrs, 100).
  - These runs showed promise with longer training but were capped at 10 epochs due to wall-clock. Async rollout should allow more epochs in the same time.
  - Pick the best candidate from that set — use the one with the best trajectory/stability profile. (E036 is the most stable at 0.71-0.72; E040 had the steepest early climb; E042 was the most stable per-token variant. Use your judgment.)

## CONTINUE

- E043 (JSD divergence) — record whatever results are available before killing, then kill. The JSD data point is valuable even if incomplete.

## REMOVE

- Kill E043 (JSD divergence run) after recording partial results. Reason: shifting focus to longer training runs with async rollout on the promising configs, rather than continuing to test new loss-function variants one at a time at 10 epochs.

---

Reminder: you are the autoresearcher. Execute the instructions above, then continue researching autonomously — do not stop when these tasks are done. Keep working through the agenda, and propose new experiments when it runs empty, and run those experiments.
