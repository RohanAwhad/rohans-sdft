# Streaming Rollouts — Implementation Plan

Goal: implement the streaming async rollout pipeline described in
`docs/megatron_trainer/async_rollouts.md` (Magistral-style streaming producer +
microbatch consumer, per-rank batch size stays 1, off-policy staleness
corrected by existing TIS).

Branch: `ra/async-rollout`.

## Locked decisions

- **Flags**: `ASYNC_ROLLOUT` (`"0"`/`"1"`) + separate `ASYNC_IN_ORDER`
  (`"0"`/`"1"`, only meaningful with async on; deterministic plumbing mode for
  Layer 1 verification).
- **Epoch semantics**: dataset-exhaustion epochs in streaming mode — producer
  drains `data_iter` then in-flight generations; epoch ends when exhausted +
  queue drained; `steps_per_epoch` dissolves.
- **Producer location**: lives in `trainer.py` for now; extraction into
  `rollout_stream.py` is a separate future PR (Phase 5 creates the issue).
- **policy_lag logging**: mean + max per step.

## Status (2026-08-14)

- Phase 0 ✅ config plumbing (8d1f996) — regression smoke ASYNC_ROLLOUT=0 passes.
- Phase 1 ✅ in-order restructure (40ed0e0) — Layer 1 PASS via replay procedure
  (see "Verification" note in async_rollouts.md): within-run assignment exact
  on both modes, cross-mode produced streams identical, steps 1-2 bit-identical
  (steps 3+ track at 3 decimals — kernel numerics, same as sync-vs-sync).
- Phase 2 ✅ streaming producer (9e60237) — smoke shows gen fully overlapped
  (producer_wait=0.0s, gen_overlap=11.4s), epoch drains cleanly, 16 steps in
  ~3 min vs sync ~15 min.
- Phase 3 ⬜ Layer 2/3 A/B campaign (200-step matched + 30-min wall-clock, evals).
- Phase 4 ⬜ N_ASYNC tuning + close-out.
- Phase 5 ⬜ create producer-refactor issue (rollout_stream.py).

## Phases

### Phase 0 — Config plumbing
- `config.py`: `ASYNC_ROLLOUT` (default `"0"`), `ASYNC_IN_ORDER` (default
  `"0"`), `N_ASYNC` (default `2 * GRAD_ACCUM_STEPS`), `IS_CAP` default
  `2.0 → 5.0`.
- `train_full.sh` + `.env.example` passthrough.
- wandb run config gains `async_rollout`, `n_async`.
- Verify: `py_compile` + `bash -n`; smoke with `ASYNC_ROLLOUT=0` unchanged
  (regression).

### Phase 1 — Plumbing restructure (in-order mode)
- Extract rank-0 rollout block (`trainer.py:249-316`) into a pure
  `produce(items) -> (rollout_data, meta)` function — same envs, same payload,
  plus `policy_version` field.
- Producer thread (rank 0, HTTP-only). In-order mode: generate the
  `GRAD_ACCUM_STEPS`-batch in dataset order, push samples column-major
  interleaved so rank `r` receives exactly the samples it gets today.
- Consumer: wait until `queue >= world_size` → pop `world_size` samples →
  `broadcast_object_list` → rank `r` takes index `r` → fwd/bwd. After
  `GRAD_ACCUM_STEPS` microbatches → optimizer → blocking weight sync → barrier.
- Move rank-0 state into producer: `data_iter`, `success_cache`,
  `reflector_fallback_count`. Ship meta per microbatch (pass_rate,
  adapter-history fields for the wandb table — extend the payload dict).
- Bounded queue (`maxsize`), backpressure both ways; producer exception →
  crash hard (no try/except, house policy).
- **Warmup**: pre-fill the queue with at least `world_size` samples (the first
  microbatch) before the consumer loop starts — else the first microbatch
  deadlocks on an empty queue.
- Verify: local `py_compile`/`bash -n`; **Layer 1 on cluster**:
  `ASYNC_ROLLOUT=1 ASYNC_IN_ORDER=1` vs baseline — per-step loss, grad_norm,
  IS metrics bit-identical. Layer 1 runs use `GEN_TEMPERATURE=0` (greedy) —
  vLLM's per-request seed is not cross-restart deterministic (verified
  empirically on 0.23), greedy is argmax-deterministic; `TRAINER_SEED` fixes
  the shuffle.

### Phase 2 — True streaming
- Producer keeps `N_ASYNC` envs in flight; per-sample push on completion
  (completion order — the reordering is the feature).
- Version-stamp each sample at generation start; log `policy_lag` (mean/max)
  + IS stats (already wired via `chunked_head.py` metrics) per step; TIMING
  line gains `producer_wait` / `gen_overlap`. Health metrics are logged
  rank-0 side only — never part of the all-reduce path.
- Epoch semantics: producer drains `data_iter`, then drains in-flight; epoch
  ends on exhaustion, not fixed counts. Epoch-end `epoch_{N}` checkpoint
  (`trainer.py:533-534`) still saves at drain-time — intentional.
- Verify: local checks; cluster smoke `ASYNC_ROLLOUT=1` — TIMING shows gen
  overlapped behind training.

### Phase 3 — Verification campaign (per design doc)
- Layer 2: sync vs async, equal optimizer steps (e.g. 200), same
  `Qwen/Qwen3-8B` starting checkpoint, checkpoints every 50 → eval each with
  `eval_maas_sdft.py` on `test_maas_sdft.jsonl` (per EVAL.md) → curves.
- **Config consistency**: both runs use identical env config, including the
  new `IS_CAP=5.0` default (the default change applies to sync too) — no
  other knobs differ.
- Layer 3: sync vs async, 30 min wall-clock each → payoff claim.
- Record: loss/grad_norm curves, IS clip-rate, policy_lag distribution.

### Phase 4 — Tune + close out
- `N_ASYNC` tuning from Layer 3 results.
- Drop-policy (AReaL η) only if IS clip-rate / policy_lag show problems.
- Update devlogs, README, PR description with test evidence.

### Phase 5 — Refactor issue
- Create GitHub issue: extract producer into `rollout_stream.py` (separate
  future PR).

## Execution notes (node rh-h100-12)

- Repo already at `/home/rawhad/1_Projects/rohans-sdft`:
  `git fetch && git checkout ra/async-rollout && git pull` before each test
  run.
- No GPU reservation — throwaway runs; if kicked off the node, just stop.
- Smoke via `smoke_all_in_container.sh` (NeMo container) with
  `VLLM_PORT=8007` as before.
- Layer 3 eval needs EVAL.md prerequisites on the node: driver-compatible
  eval venv (vLLM/torch pins per EVAL.md), `gcloud auth
  application-default login`, `CLOUD_ML_REGION` /
  `ANTHROPIC_VERTEX_PROJECT_ID`, and `test_maas_sdft.jsonl` from
  `rh-h100-01`.

## Verification layers (summary)

1. **Plumbing equivalence** — in-order async vs sync: bit-identical per-step
   loss, grad_norm, IS weights.
2. **Matched-step A/B** — equal optimizer steps, eval curves at every
   checkpoint.
3. **Matched wall-clock A/B** — 30 min each; async doing more steps is the
   point, not a bug.
