# Async Rollouts for the SDFT Trainer — Porting Analysis

> Generated: 2026-08-12 | Companion to `RESEARCH_async_rollouts_prime_rl.md` + `RESEARCH_async_rollouts_verl.md`
> Sources: the two research docs (all code, no web) + `megatron_trainer/trainer.py` (our loop)
> Tags: #async-rollouts #porting #sdft #design-decisions

## TL;DR

- **PRIME-RL (new design)** is the closest architectural match to our SDFT trainer: 3 processes (vLLM pool + CPU orchestrator + torchrun trainer), `max_async_level=1` overlap, weight sync every step via NCCL broadcast, staleness cap (`max_off_policy_steps=8`) cancels stale inflight rollouts, token-level IS ratio in the loss.
- **VERL (v1)** is the wrong tool to port wholesale (Ray + TransferQueue + colocated worker groups), but has 3 stealable ideas: **partial-rollout abort-resume** (`FullyAsyncLLMServerClient`), **staleness-evicting replay buffer** with weight-version stamps, and **Decoupled PPO** (`parameter_sync_step` mini-updates per weight sync).
- Recommended port: **background producer thread on rank 0, 1-ahead pipeline, sync-then-gen ordering** — ~100-line diff in `trainer.py`, no new processes, no new dependencies. Reuses our existing `IS_WEIGHTING`/`IS_CAP` machinery for the induced off-policy gap.

## Current Approach vs. Recommended

| Dimension | Current (synchronous) | Recommended (1-ahead async) | Evidence | Expected impact |
|---|---|---|---|---|
| Generation | Rank 0, blocking, `ThreadPoolExecutor` over `GRAD_ACCUM_STEPS` envs at top of each step (trainer.py:249-282) | Same, but in a producer thread running **during** the previous step's training | prime-rl k=1 overlap (docs/async.md:3) | Hides ~120s gen time behind teacher/student/backprop/optimizer |
| Handoff | `dist.broadcast_object_list` after gen (trainer.py:320) | Unchanged — broadcast from the completed producer queue at step start | — | No transport rewrite |
| Weight sync | Collective on all ranks at step end, to vLLM + logprob server (trainer.py:496-500) | Unchanged position — **sync θ_n before gen(n+1) starts** (prime-rl ordering, NCCL mode forces k=1) | prime-rl `broadcast/nccl.py` + validator `orchestrator.py:1089-1094` | Gen always uses θ_n; off-by-one policy gap |
| Off-policy | Already computes vLLM rollout logprobs + `IS_WEIGHTING`/`IS_CAP` (trainer.py:347-368) | Keep as-is; it's exactly the ratio prime-rl uses (`exp(π−μ)`, loss.py:137-138) | prime-rl loss.py; verl rollout_corr_helper.py | Zero new math |
| Teacher logprobs | Per-rank TCP inside accum loop (trainer.py:376-381) | Unchanged — teacher is **frozen**, so teacher logprobs have no staleness; teacher GPUs now work on batch n while vLLM generates batch n+1 | — | Teacher GPUs stop idling |
| Backpressure | Barrier at step end (trainer.py:507-509) | Producer-queue `get()` blocks if gen lags (trainer-side stall = prime-rl `wait_for_batch`) | prime-rl data.py:174-183 | Fail-fast, no deadlock |

## PRIME-RL vs VERL: Design Comparison

| Dimension | PRIME-RL (new arch) | VERL v1 |
|---|---|---|
| Process model | 3 fixed processes: vLLM pool + CPU orchestrator (asyncio) + torchrun FSDP2 trainer | Ray actors everywhere: colocated worker groups + server replicas + agent-loop workers |
| Overlap mechanism | Orchestrator keeps `max_inflight_rollouts` requests in flight; k-step stale-policy bound (`max_async_level`, default 1) | Fire-and-forget agent-loop workers; trainer polls TransferQueue; warmup batches get gen ahead |
| Producer/consumer decoupling | Filesystem (msgpack per-step) or ZMQ; marker-file handshake (`STABLE`, `NCCL_READY`) | External TransferQueue KV store (force-enabled in v1) with status + staleness tags |
| Weight sync | Every trainer step; NCCL broadcast (trainer rank 0 = root) or FS export; inference pool paused during update | Colocated: naive in-process + sleep/resume. Disaggregated: NCCL/NIXL/Mooncake with abort → KV-cache release → resume-generation |
| Staleness policy | Cancel inflight rollouts with `off_policy_steps ≥ max_off_policy_steps` (8) | Evict finished trajectories with span > `max_off_policy_threshold` (8); `drop` or `wait` strategy + refill |
| Off-policy correction | Token ratio `exp(π−μ)` + DPPO-Binary TV masks (unclipped ratio × advantage) | Full toolbox: TIS clamp / IcePop / rejection-sampling masks / bypass mode; Decoupled PPO `parameter_sync_step` |
| Interrupted generation | No — pool is drained (paused) before update; no mid-gen aborts | **Partial rollout**: abort + resume from `prompt+tokens_so_far`, stitch outputs, retry loop |
| Step semantics | Global step n tags everything; π_n from (x_n, y_n); y_n from π_{max(0,n−k)} | Weight-version stamps `min/max_global_steps` on every trajectory |
| Closest to our SDFT trainer | **Yes** — same 3-process shape (vLLM server + logprob server + FSDP trainer) | No — Ray/TransferQueue would be a rewrite of our orchestration |

## Design Decisions (for the SDFT port)

### Decision 1: Where does the async producer live?

| Option | Pros | Cons | Best when | Evidence |
|---|---|---|---|---|
| **A. Producer thread on rank 0** (recommended) | ~100-line diff; gen is already rank-0-only; keeps `broadcast_object_list` handoff; no new processes/launch changes | Rank-0 CPU/threading contention; less process isolation | Single-node, current codebase size | Our loop trainer.py:249-320 |
| B. Separate orchestrator process (prime-rl port) | Clean separation; CPU-only process; can decouple from trainer node; matches prime-rl exactly | New process management; replace broadcast with FS/ZMQ transport; launch-script changes; bigger diff | Multi-node scale-out, elastic pools | prime-rl entrypoints/rl.py, transport/ |
| C. Ray + TransferQueue (verl port) | Battle-tested; staleness tooling built-in | Introduces Ray + external kv lib into a torchrun/MCore stack; colocated worker rewrite | Rebuilding the whole trainer | verl trainer_base.py, agent_loop_tq.py |

→ **Option A.** Our trainer is torchrun + MCore FSDP with two NCCL groups; it is not Ray-based, and the gen path is already rank-0-only + HTTP. A thread is the smallest diff that captures the overlap.

### Decision 2: Sync ordering / staleness model

| Option | Pros | Cons | Best when | Evidence |
|---|---|---|---|---|
| **A. 1-ahead, sync-then-gen** (recommended) | Gen(n+1) always uses θ_n (latest synced); off-by-one gap fully covered by existing IS; no mid-gen interruptions; matches our collective weight-sync constraint | Gen must finish within one train step to avoid trainer stall; vLLM idles while waiting for sync | Trainer-bound or balanced; our current sync-every-step cadence | prime-rl k=1 + NCCL validator (orchestrator.py:1089-1094) |
| B. N-ahead window + staleness drop | vLLM never starves even if gen > train; more utilization headroom | More off-policy; needs drop accounting + refill; complicates env multi-turn caching | Gen-bound (our 120s case!) or when sync is expensive | prime-rl max_off_policy_steps=8; verl ReplayBufferAsync drop/refill |
| C. Continuous inflight + abort-resume | Gen never interrupted; verl-style partial rollout | Needs vLLM `/pause`+resume with partial-generation resume support; most complex | Weight sync slower than gen turnaround | verl llm_server.py:404-456 (FullyAsync client) |

→ **Start with A**, instrument `gen_time` vs `train_time` (we already log TIMING, trainer.py:513-517). If gen-bound (120s >> ~40s train), move to **B**: increase the inflight window to `max_off_policy_steps` batches and let staleness drop handle the tail. Note our knob: `GRAD_ACCUM_STEPS` controls both batch size and train time per sync — the balance knob prime-rl assumes equals 1:1 (async.md:3).

### Decision 3: Off-policy correction

| Option | Pros | Cons | Best when | Evidence |
|---|---|---|---|---|
| **A. Existing IS_WEIGHTING/IS_CAP** (recommended) | Already implemented and smoke-tested; token-level ratio from vLLM logprobs; cap ≈ AIPO δ | Capping only, no hard trust region | Off-by-one gap (k=1) | trainer.py:347-368; prime-rl loss.py:137-138 |
| B. DPPO-Binary TV masks | Hard per-token masking instead of soft cap | Replaces current cap; needs validation vs our chunked-KL loss | Larger k, diverging policies | prime-rl loss.py:108-135 |
| C. Rejection sampling (verl) | Strongest guarantees; response_mask overwrite | Heavy; per-sample rejection wastes rollouts; interacts with our reverse-KL teacher loss | k large, on-policy-critical domains | verl rollout_corr_helper.py:197-413 |

→ **Option A.** Our teacher is frozen, so the IS ratio only corrects student-vs-rollout-policy drift — much smaller than GRPO advantage drift. Revisit B only if we go N-ahead (Decision 2B).

### Decision 4: What the logprob (teacher) server does

- Teacher is frozen → **no staleness, no version stamps, no sync changes**. It just gets to work on batch n while vLLM generates batch n+1. This is the direct win of the overlap: our 120s gen time currently idles the teacher GPUs too.
- Only change: none. (If we later train the teacher — SDFT with moving teacher — we'd need verl-style weight-version stamps; note as future work.)

## Hard Constraints (surface-level findings)

1. **Weight sync is collective on all trainer ranks** (FSDP export, trainer.py:496-500). Any overlap must not collide with it — the producer thread must be HTTP-only (no collectives). Gen is already HTTP → safe.
2. **NCCL sync cadence forces k=1** in prime-rl (validator oracle: `orchestrator.py:1089-1094`); same logic applies to our `sync_weights_to_vllm` + `sync_weights_to_logprob_server` — deeper overlap (N-ahead) only helps if we sync less often (see GRAD_ACCUM knob).
3. **Two independent NCCL groups already coexist** (trainer torch.distributed + vLLM weight engine) — a producer thread adds no NCCL surface.
4. **`dist.broadcast_object_list` must stay the handoff** (it's a collective — can't run inside the producer thread; must run on the main step path as today).
5. Gen-bound case: if `gen_time > train_time`, 1-ahead alone still leaves trainer idle part of the time (prime-rl's zero-idle claim requires balanced step times, async.md:3). Our lever = `GRAD_ACCUM_STEPS` (bigger batches per sync) or N-ahead window.

## Next Steps: Implementation

1. **`megatron_trainer/trainer.py`** — rank 0 only:
   - Producer thread owns `data_iter` + env construction + `ThreadPoolExecutor` + `rollout_data` build (extract today's lines 249-315 into `produce_step(items)`).
   - `queue.Queue(maxsize=1)` + `threading.Event`: after init, producer pre-generates step-1 data (warmup = verl's `num_warmup_batches=1`); after each weight sync, main loop signals producer to generate the next batch (uses θ_n).
   - Main loop: `rollout_data = gen_queue.get()` → broadcast → accum → optimizer → sync → signal.
   - Metrics: log `producer_wait_time` (trainer idle waiting for gen) and `gen_overlap_time`; keep TIMING line.
2. **Config**: optional `ASYNC_ROLLOUT=1` env flag (default on once validated) so sync mode remains runnable for comparison.
3. **No changes**: `logprob_server.py`, `vllm_utils.py`, weight-sync path, collator, IS machinery.
4. **Test plan**: run existing smoke (sync) → run async smoke → compare TIMING lines; verify IS weights non-degenerate with off-by-one policies; confirm no collective-from-thread (would hang — smoke catches it).
5. **Future work** (only if gen-bound after A): N-ahead window + staleness drop (Decision 2B), DPPO masks (3B), partial-rollout resume if we ever sync mid-gen, teacher-version stamps if teacher ever trains.
