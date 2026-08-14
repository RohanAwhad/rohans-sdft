# Streaming Rollouts — design (`trainer.py`)

> Overlaps generation and training: Magistral-style streaming producer +
> microbatch consumer. Per-rank batch size stays 1 — no padding/masking
> changes. Off-policy staleness is accepted (bounded) and corrected by the
> existing TIS machinery.

## Baseline (today)

Serial step loop, `trainer.py:243-517`:

1. Rank 0 pulls `GRAD_ACCUM_STEPS` items from `data_iter`, builds envs
   (RagEnv/ApiAdapterEnv), runs them via `ThreadPoolExecutor` — vLLM
   generation completes for **all** samples before anything else happens.
2. Broadcast the whole batch → teacher log-probs (TCP) → student forward →
   reverse-KL → backward, 1 sequence per rank per micro-step
   (`BATCH_SIZE = 1`, `attention_mask=None`).
3. `GRAD_ACCUM_STEPS` micro-steps → optimizer → weight sync (vLLM + logprob
   server) → barrier.

Waste: trainer/teacher GPUs idle during generation; vLLM idle during
training; every step waits for the slowest rollout (skewness bubble).

## Design

### Producer (rank 0, HTTP-only thread)

- Continuously pulls from `data_iter` and keeps `N_ASYNC` envs in flight
  (vLLM HTTP calls).
- **Each completed generation is pushed to the queue immediately** — a
  per-sample stream, no batch barrier. A 20K-token straggler finishes whenever
  it finishes; the queue is already fed by the shorter completions.
- Each sample carries a `policy_version` stamp (optimizer-step counter when
  its generation started).
- Runs until `data_iter` is exhausted, then drains remaining in-flight
  generations into the queue.

### Consumer (main path, all ranks)

```
wait until queue ≥ MICROBATCH
pop MICROBATCH samples (world_size, one per rank) → broadcast → fwd/bwd
repeat until GRAD_ACCUM_STEPS microbatches done
→ optimizer step
→ weight sync (all ranks, collective; training paused until sync completes)
→ barrier
→ back to waiting for the queue
```

- **Microbatch = `world_size` samples, 1 per rank** — per-rank batch size
  stays 1, so the forward/backward math is byte-identical to today
  (`no_sync` on non-final micro-steps, `scaled_loss / local_accum_steps`).
  Only the pull/broadcast granularity changes: `GRAD_ACCUM_STEPS` broadcasts
  per optimizer step instead of one.
- Training starts as soon as `MICROBATCH` samples are ready — it never waits
  for the full `GRAD_ACCUM_STEPS` batch to be generated.

### Off-policy contract

- No sync-then-gen barrier. Generations in flight when θ updates finish under
  the old policy; the `policy_version` stamp records it.
- Staleness is bounded: Magistral's `N_ASYNC / step_batch ≤ 2` conservative
  limit (`step_batch = GRAD_ACCUM_STEPS`).
- Correction is the existing TIS: `completion_log_probs` from vLLM →
  `compute_is_weight` rescales the reverse-KL by
  `clamp(exp(policy_logp − rollout_logp), IS_CAP)` — no new math. Teacher is
  frozen, so teacher log-probs carry no staleness.
- `IS_CAP` default: **5** (was 2.0 — config change to be applied when
  implemented).

### Config

- `ASYNC_ROLLOUT` — flag, env-read in `config.py`, default off; `0` keeps the
  byte-identical baseline path.
- `N_ASYNC` — in-flight generations (default: `2 × GRAD_ACCUM_STEPS`).
- `IS_CAP` — default `5`.
- `train_full.sh` passthrough for all three.

## Verification plan

Three layers — plumbing correctness, per-step quality, then wall-clock payoff.
All training runs start from `Qwen/Qwen3-8B` (same starting checkpoint as the
smoke test), same data, same env config.

### Layer 1 — plumbing equivalence

Claim: the thread/queue/per-microbatch-broadcast plumbing changes nothing about
the math — async trains on the same per-rank microbatch slicing as sync, with
per-sample teacher log-probs, reverse-KL and IS weighting applied identically.

- **Result (verified during the campaign)**: batch-0 produced rollout hashes
  and step-1 loss/grad_norm/IS metrics were exactly identical across sync and
  async runs; per-microbatch consumed-stream hashes matched the produced
  stream under the column-major map `rank r, microbatch k ← sample r·L + k`;
  steps 2+ loss/grad_norm curves track each other (same numerics drift as
  sync-vs-sync).
- **Determinism reality (verified empirically)**: vLLM 0.23's per-request
  `seed` is *not* cross-restart deterministic (two fresh engines, same
  prompts, same seed → different completions). The campaign verified Layer 1
  with `GEN_TEMPERATURE=0` (greedy) + `TRAINER_SEED` (fixed shuffle), where
  batch 0 and step-1 metrics are exactly identical across runs; from batch 1
  onward training-kernel numerics (flash-attn/TE atomics, sub-1e-4) accumulate
  and flip greedy argmax ties — so cross-run equality beyond step 1 is
  statistical, not bit-exact.
- The verification machinery (deterministic in-order producer, rollout
  record/replay, hash stamping) was removed from the repo after the campaign
  concluded; the async path is the streaming producer only.

### Layer 2 — matched-step A/B

- Sync vs async, **equal optimizer steps** (e.g. 200 each), same starting
  checkpoint, same data. Checkpoints every 50 steps → eval each → curves.
- Isolates: does off-policy staleness + reordered data hurt quality *per step*?

### Layer 3 — matched wall-clock A/B

- Sync 30 min vs async 30 min, same starting checkpoint. Async does more
  steps — that is the point, not a bug: this run answers "what do I get for 30
  minutes", not "is it correct" (Layer 2 is the correctness gate).

### Metrics / health signals

- **Eval**: `eval_maas_sdft.py` on `test_maas_sdft.jsonl` (canonical copy per
  `EVAL.md` on `rh-h100-01`), Claude-judged no_context + with_context pass
  rate, at every checkpoint.
- **Training curves**: per-step train loss + grad_norm (sensitive, free).
- **Async health (new logging, part of implementation)**: per-step mean IS
  weight, **% IS weights clipped at `IS_CAP`**, and `policy_lag` distribution
  (steps between generation and training). Low clip-rate + lag ≪ `N_ASYNC`
  bound = TIS doing bounded work = the "async is safe" evidence.

### Pitfalls (accepted)

- Data order differs between runs (completion-order vs dataset-order) → some
  variance; compare curves (multiple checkpoints), not single final values.
- 30 min at 8B/6 GPUs ≈ tens of steps — likely too short for pass-rate eval
  signal; Layer 2 is step-matched (however long it takes), 30 min is Layer 3
  only.
- 100-question eval split → 1% noise floor; loss curves are the more sensitive
  correctness signal.

## Hard invariants

- **No collectives in the producer thread** — HTTP-only; `broadcast`,
  all-reduce, weight sync, barrier stay on the main path.
- **Weight sync after every optimizer step, all ranks, main path, blocking** —
  producer never generates while the vLLM server is mid-update.
- **Queue bounded** (`maxsize`), backpressure both ways: producer blocks on
  full queue; consumer blocks on empty.
- **Rank-0 state moves into the producer**: `data_iter`, `success_cache`,
  reflector fallback counter; main path only reads shipped meta
  (`full_pass_rate`, `reflector_fallback_count`, `success_cache` updates).
- **Warmup**: queue pre-filled before the loop starts, else the first
  microbatch deadlocks.

## Known gotchas

- Producer exceptions must propagate and crash hard (house policy) — a dead
  producer hangs the consumer on `get()` forever.
- loguru is thread-safe; wandb is not — wandb stays on the main path.
- `steps_per_epoch` semantics change: producer drains `data_iter`, then the
  queue; epochs end on data exhaustion, not fixed counts.
- Per-microbatch broadcasts add collectives per step (e.g. 32 vs 1) — small
  overhead at this scale, accepted.
- `success_cache` becomes single-writer (producer).

## Future work

- **Mid-generation weight sync** (pause → sync → resume, no KV recompute):
  tracked in https://github.com/RohanAwhad/rohans-sdft/issues/12. Not planned
  now; streaming + TIS covers the staleness it would fix.
- **Per-rank batch > 1**: padding + attention masks, AReaL-style dynamic
  token-balanced microbatching — explicitly out of scope for now
  (https://github.com/RohanAwhad/rohans-sdft/issues/13).
- **Drop/regenerate stale samples** (AReaL η) if IS-weight variance shows up
  in practice (https://github.com/RohanAwhad/rohans-sdft/issues/14).
- **Extract producer into `rollout_stream.py`**:
  https://github.com/RohanAwhad/rohans-sdft/issues/15.

## Campaign results (2026-08-14, rh-h100-12, Qwen3-8B)

Config: 400-sample maas sdft train set, `GRAD_ACCUM_STEPS=8` (50 steps/epoch,
exact in both modes), `IS_CAP=5.0`, `GEN_TEMPERATURE=1.0`, `TRAINER_SEED=1234`,
2 trainers + 1 vLLM + 1 logprob server per run, sync and async in parallel on
8×H100. Eval: `eval_maas_sdft.py` at steps 50/100/150/200 (3× Claude majority
judge, 100 test questions).

- **Layer 2 (matched 200 steps)**: async 3.6 s/step (~12 min total) vs sync
  10–18 s/step (~55 min) — ~4–5× wall-clock at equal steps. Async health:
  `producer_wait` 0.10 s mean (N_ASYNC=16 keeps the queue full — no tuning
  needed), `policy_lag` 3.8 mean / 4.5 max, IS clip-rate ~0.002.
- **Layer 3 (matched wall-clock, L3 set = 4000 lines)**: sync 150 steps in
  ~55 min; async 500 steps in ~25 min (3.3× steps in half the time).
  First async attempt with N_ASYNC=16 deadlocked at step 2 (rank 0 blocked in
  `rollout_queue.get()` while rank 1 waited at the broadcast collective →
  1800 s NCCL watchdog abort; logprob server went silent mid-request);
  rerun with N_ASYNC=8 completed cleanly — the bounded window prevents the
  producer from out-running the servers.
- **Eval pass rates** (no_context / with_context, 100-question split,
  3× Claude majority judge):

  | ckpt | sync | async |
  |---|---|---|
  | base | 3 / 84 | — |
  | step 50 | 12 / 83 | 8 / 81 |
  | step 100 | 24 / 78 | 11 / 66 |
  | step 150 | 21 / 72 | 15 / 51 |
  | step 200 | 27 / 67 | 12 / 50 |
  | L3 final (150 sync / 500 async) | 25 / 79 | 16 / 61 |

  Sync leads per-step quality (no off-policy staleness), async leads
  wall-clock step count (4–5× throughput). At matched wall-clock (~30–60 min)
  both improve on base (3/84); sync's final checkpoint is ahead in absolute
  pass rate, async offers ~3.3× more optimizer steps in less time — the
  trade-off Layer 3 was designed to expose.
