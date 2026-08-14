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

- `ASYNC_ROLLOUT=1` with a **deterministic in-order producer mode**: queue
  pre-filled in dataset order, no overlap (producer waits for consumption).
- Claim: per-step loss, grad_norm, and IS weights are **bit-identical** to
  `ASYNC_ROLLOUT=0`. Proves the thread/queue/per-microbatch-broadcast plumbing
  changes nothing about the math.
- **Verification runs use `GEN_TEMPERATURE=0` (greedy)** + `TRAINER_SEED`
  (fixed shuffle): empirically, vLLM 0.23's per-request `seed` is *not*
  cross-restart deterministic (two fresh engines, same prompts, same seed →
  different completions; in-session repeats do match). Greedy sampling is
  argmax over identical logits → identical completions across restarts,
  which is what bit-identity requires. The stochastic sampling path is
  exercised by Layers 2/3 at the default temperature.

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
