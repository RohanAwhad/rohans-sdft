# Async Rollouts — overlapped generation & training (`trainer.py`)

> Design follows PRIME-RL `max_async_level=1` semantics (sync-then-gen). Evidence and
> alternatives (N-ahead window, partial rollout, verl-style transfer queues) in
> `../research/async_rollouts_porting_analysis.md`.

## Role

Today the step loop is strictly serial (`trainer.py:243-517`): rank 0 runs the
whole rollout (vLLM generation, ~120s on gpt-oss) **before** any training work
starts, so the teacher (logprob server) and trainer GPUs sit idle during
generation — and vLLM sits idle during training. With async rollouts:

- A **producer thread** on rank 0 generates the rollout batch for step `n+1`
  while all ranks train step `n` (teacher log-probs → student forward →
  backward → optimizer).
- The main path never changes its collective structure: it still broadcasts the
  batch, accumulates gradients, syncs weights at the end of each optimizer step.
- The only reordering: **weight sync of step `n` completes before generation for
  step `n+1` starts**, so every rollout is generated under the latest synced
  policy `θ_n` (PRIME-RL `max_async_level=1`; NCCL sync cadence forces k=1 — see
  research doc, Hard constraints §1-2).

Net effect: `gen` hides behind `teacher + student + loss_bwd + optim`; the
teacher GPUs process batch `n` while vLLM generates batch `n+1`.

## Overlap model (step semantics)

Global optimizer step `n` (1-indexed):

- Rollout batch `(x_n, y_n)` was generated **during step `n-1`** under policy `θ_{n-1}`.
- Step `n` trains on it, then syncs `θ_n` to vLLM + logprob server.
- After the sync, the producer starts generating `(x_{n+1}, y_{n+1})` under `θ_n`.

Timeline:

```
step n (main path, all ranks):          producer thread (rank 0 only):
  get batch_n from queue (blocking)        (idle — batch_n already produced)
  broadcast → accum loop → optimizer
  sync θ_n → vLLM + logprob server
  signal producer ─────────────────────▶   generate batch_{n+1} under θ_n
  barrier                                  puts batch_{n+1} + meta into queue
step n+1: get batch_{n+1} ...              ...
```

- **Warmup**: before the loop, the producer synchronously generates batch 1
  (identical to today's first-step behavior) so step 1 has data.
- **Backpressure**: `queue.Queue(1)` — `get()` blocks when generation lags
  (trainer-side stall, prime-rl's `wait_for_batch` equivalent); `put()` blocks
  when training lags (vLLM-side backpressure).

## Communication contract (how the pieces talk)

```
# rank 0, producer thread — owns everything gen-related today's rank-0 block does
def produce_step(items: list) -> tuple[list[dict], dict]:
    envs = [RagEnv(...) | ApiAdapterEnv(...) for item in items]   # trainer.py:254-279
    ThreadPoolExecutor(...).map(lambda e: e.run(), envs)           # trainer.py:281-282
    rollout_data = [{prompt_text, completion_text,
                     completion_log_probs, privileged_information_prompt} ...]  # trainer.py:296-304
    meta = {full_pass_rate, reflector_fallback_count, success_cache_updates}
    return rollout_data, meta

gen_queue: queue.Queue[tuple[list[dict], dict]]   # maxsize=1
gen_ready: threading.Event                        # set by main path after sync
```

Main path (`trainer.py:243-517` reshaped):

1. `rollout_data, meta = gen_queue.get()` (replaces today's inline rollout block).
2. `dist.broadcast_object_list(rollout_data, src=0)` — **stays on the main path**
   (it is a collective; it can never run inside the thread).
3. Teacher log-probs / student forward / backward / optimizer — unchanged
   (`trainer.py:336-447`).
4. Weight sync to both servers — unchanged position, collective on all ranks
   (`trainer.py:494-500`).
5. `gen_ready.set()` → producer (woken) generates the next batch.
6. `dist.barrier()` — unchanged (`trainer.py:506-509`).

When `ASYNC_ROLLOUT=0` the code path is byte-identical to today (no thread, no
queue, inline rollout block).

## I/O shapes

- **Producer → queue**: the existing broadcast payload, unchanged — a list of
  `GRAD_ACCUM_STEPS` dicts with `prompt_text`, `completion_text`,
  `completion_log_probs`, `privileged_information_prompt` (`trainer.py:296-304`).
- **Producer → meta** (rank-0-only, consumed by the main-path logger):
  `full_pass_rate` (episode/reflector verdicts, `trainer.py:307-313`),
  `reflector_fallback_count` (`trainer.py:284-285`), and the api_adapter
  `success_cache` updates (`trainer.py:287-294` — cache object itself lives in
  the producer).
- **No shape changes** downstream: broadcast, teacher TCP request, `make_kl_processor`
  all consume the same dicts.

## Off-policy contract

- `y_n` is generated under `θ_{n-1}` → off-by-one policy gap. The existing
  TIS machinery already corrects exactly this: rollout (vLLM) per-token
  log-probs → `rollout_log_probs` tensor (`trainer.py:347-368`) →
  `compute_is_weight` (`chunked_head.py:28`, `chunked_head.py:172-181`) rescales
  the reverse-KL loss by `mean(clamp(exp(policy_logp − rollout_logp), IS_CAP))`.
  No new math (see research doc, Decision 3A).
- **Teacher is frozen** (`TEACHER_MODEL_PATH`) or EMA-synced — its log-probs
  carry no staleness; the teacher GPUs simply stop idling.
- Gap is bounded at 1 step by construction (sync-then-gen ordering). If gen
  becomes slower than training, the trainer blocks on `get()` — vLLM stays
  continuously busy; accepted for v1 (see Future work §A for the fix).

## Config

- `ASYNC_ROLLOUT` — feature flag, env-read in `config.py` at import:
  `ASYNC_ROLLOUT = os.environ.get("ASYNC_ROLLOUT", "0") == "1"`.
  **Default off**; both code paths retained for A/B comparison.
- `train_full.sh` passthrough: `-e ASYNC_ROLLOUT="${ASYNC_ROLLOUT:-0}"`.
- TIMING line (`trainer.py:511-517`) gains: `producer_wait` (main-path time
  blocked on `gen_queue.get()` past data readiness), `gen_overlap` (gen wall
  time hidden behind training). wandb `run config` gains `async_rollout`.

## Hard invariants

- **No collectives in the producer thread** — it is HTTP-only (vLLM calls). Any
  torch.distributed call from the thread deadlocks the step (two NCCL groups
  already coexist; the main path is the only collective context).
- **Weight sync stays collective on all ranks, on the main path, after the
  optimizer step and before `gen_ready.set()`** — the producer must never
  generate while the vLLM server is paused mid-update.
- **Queue depth = 1** — at most one rollout batch in flight; memory bounded.
- **Producer starts `gen(n+1)` only after the sync event** — guarantees `θ_n`
  is what vLLM serves (k=1 off-policy bound).
- **`broadcast_object_list` stays on the main path** (collective, cannot move
  into the thread).
- **Rank-0-only state moves into the producer**: `data_iter`, `success_cache`,
  reflector fallback counter. The main path only reads the shipped meta.
- **Warmup batch 1 is generated before the loop** (blocking), else step 1
  deadlocks on an empty queue.

## Known gotchas

- **Producer exception must propagate and crash hard** (no try/except — house
  policy). A dead producer would otherwise hang the main path on `get()` forever.
- **loguru is thread-safe; wandb is not** — all wandb calls stay on the main
  path (rank 0); the producer only logs via logger.
- **Event discipline**: producer clears `gen_ready` before waiting, and the main
  path sets it exactly once per step — a stale set would let gen run under an
  unsynced policy.
- **vLLM pause-during-sync** is avoided entirely by the sync-then-gen ordering —
  no mid-generation aborts, no resume logic needed (this is why ordering A beats
  verl's abort-resume for us; research doc Decision 2A).
- **`success_cache` becomes single-writer** (producer) after handoff — today's
  block at `trainer.py:287-294` mutates it inline; moving it avoids a data race.
- First-step latency is unchanged (warmup generation is blocking, same as
  today's step 1).

## Future work

- **A. N-ahead window + staleness drop** (only if gen-bound persists): let the
  producer run up to `max_off_policy_steps` batches ahead (prime-rl's inflight
  window + `off_policy_steps` culling, default 8) and drop/regenerate rollouts
  older than the cap (verl `ReplayBufferAsync` drop/wait strategies). Requires
  version-stamping batches and a bigger queue. Evidence: research doc, Decision 2B.
- **B. DPPO-Binary TV masks** (prime-rl loss.py:108-135) in place of the TIS cap
  if the off-policy gap grows with A.
- **C. Partial-rollout abort-resume** (verl `FullyAsyncLLMServerClient`) only if
  we ever want to sync mid-generation instead of sync-then-gen.
- **D. Separate orchestrator process** (prime-rl's CPU asyncio orchestrator +
  FS/ZMQ transport) if generation ever moves off the trainer node.
- **E. Teacher version stamps** if the teacher ever becomes trainable — today's
  frozen teacher makes staleness tracking unnecessary.
