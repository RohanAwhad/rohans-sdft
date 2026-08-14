# PRIME-RL Async Rollout Engine — Deep Research

> Generated: 2026-08-12 | Repo: `~/3_resources/external_libs/prime-rl` @ `e8abfa26`
> Sources: 100% code (no web) | Tags: #async-rollouts #off-policy #prime-rl #sdft #research

## TL;DR

- The local checkout is a **full rewrite** — none of the classic PRIME Intellect machinery (`SyncReplayBuffer`, `SyncGenerateExperiences`, `run_forward/backward`, `forward_only/backward_only` agents, `executor/llm/`, mcore backend) exists anymore.
- New design: **3 processes** — vLLM inference pool (GPUs) + lightweight CPU **orchestrator** (single asyncio loop) + **FSDP2 trainer** (torchrun). Architecturally very close to our SDFT layout (vLLM server + trainer).
- Async model: inference generates rollouts from policy `π_{max(0, n-k)}` while trainer trains step `n`, with `k = max_async_level` (code default **1**). `docs/async.md:3,32-39`.
- Backpressure is symmetric: orchestrator **stalls** (clears `checkpoint_ready`, waits on `STABLE` file) if it gets >k ahead of the trainer; trainer **stalls** (`wait_for_batch`) if it outruns the orchestrator.
- Off-policy is handled by (a) a staleness cap (`max_off_policy_steps=8`: inflight rollouts older than 8 weight-updates are **cancelled**) and (b) token-level importance ratio `exp(π_logp − μ_logp)` in the loss.
- Weight sync is **every trainer step**, blocking, via NCCL broadcast (trainer rank 0 = root, vLLM workers = other ranks, pause/NCCL_READY/resume handshake) or shared-FS HF export + `STABLE` marker.

## Overview

PRIME-RL (PRIME Intellect) trains RL agents with an asynchronous off-policy pipeline: a dedicated inference pool continuously generates rollouts (continuous batching) while a separate trainer process consumes them one global step behind. The separation between producer (inference) and consumer (trainer) is mediated by a CPU-only orchestrator, and the only synchronization points are (1) batch handoff via filesystem/ZMQ, and (2) weight broadcast.

Why it matters for us: our SDFT trainer already has this exact process topology — vLLM server (GPUs) + logprob server + FSDP trainer — but a *synchronous* loop where 120s of generation starves the teacher/trainer GPUs. PRIME-RL's new architecture is a direct template for overlapping our generation with training.

## Key Findings

### 1. Process topology: launcher splits GPUs 3-ways

`src/prime_rl/entrypoints/rl.py` — console script `rl`:

- `rl_local()` splits node GPUs into three contiguous groups (rl.py:116-124):
  `infer_gpus`, `trainer_gpus`, `teacher_gpus`.
- Spawns in order: inference (vLLM, rl.py:189-204) → teacher inference if distillation (rl.py:227-259) → orchestrator (**no GPU**, rl.py:268-290) → trainer via `torchrun --nproc-per-node` (rl.py:303-339).
- Every subprocess is monitored; any crash tears down everything (rl.py:360-383).

### 2. Orchestrator: single asyncio event loop + Scheduler

`src/prime_rl/orchestrator/orchestrator.py:86` — `orchestrate()`:

- Sets up inference pool (`StaticInferencePool`/`ElasticInferencePool`, `utils/client.py:58,97`), optional teacher pool (orchestrator.py:130-141), `Buffer` (prompts only), and `Scheduler` (orchestrator.py:232-244).
- `Scheduler` (scheduler.py:47) cites AReal + PipelineRL in its docstring — the overlap engine.

Core scheduling state (scheduler.py:100-117):
- `inflight_requests: dict[asyncio.Task, InflightRequest]` — continuous-batching window.
- `groups: dict[int, GroupState]` — one prompt × `rollouts_per_example` rollouts, completed independently.
- `checkpoint_ready: asyncio.Event` — **gate that pauses scheduling while weights change**.
- `self.step` (orchestrator step) vs `self.ckpt_step` (policy version) — the gap **is the async level** (scheduler.py:106).

Rollout dispatch: `schedule_rollout()` (scheduler.py:180-229) picks the least-loaded client (`_select_least_loaded_client`, sticky per group for prefix-cache reuse, scheduler.py:153-164) and fires `asyncio.create_task(env.run_rollout(...))` — async HTTP to vLLM. `_schedule_next_request()` tops up to `max_inflight_rollouts` (scheduler.py:240-263).

### 3. The async overlap contract: `max_async_level`

Canonical doc `docs/async.md`:
- "Inference produces rollouts from policy `π_{max(0,n-k)}` while trainer produces `π_n`" (async.md:32-39).
- "With k=1 and trainer/inference step timings being equal, this allows to run without any idle time on either the trainer or inference" (async.md:3).
- Doc claims default k=2; **code default is 1** (`configs/orchestrator.py:1004-1010`). Trust code.

Policy-update loop (scheduler.py:269-344):
- Polls every 1s (`update_policy_loop`, scheduler.py:269-273).
- `_compute_next_ckpt_step()`: non-strict = `max(async_away, latest_ckpt_step)` where `async_away = max(step - max_async_level, 0)` (scheduler.py:275-280).
- `_apply_policy_update()` (scheduler.py:282-313): if the trainer is lagging, **clears `checkpoint_ready`** (pauses scheduling), waits for the `STABLE` marker, calls `inference_pool.update_weights(...)`, re-sets the event.

NCCL broadcast forces `max_async_level == 1` (validator, `configs/orchestrator.py:1089-1094`).

### 4. Stale-rollout culling (their "replay buffer" replacement)

There is **no sample-level replay across steps**. Off-policyness = stale-policy *in-flight* rollouts:

- Each `InflightRequest` carries `off_policy_steps` (scheduler.py:26-34).
- After each policy update, `_update_off_policy()` (scheduler.py:346-371) **cancels** groups with `off_policy_steps >= max_off_policy_steps` (default **8**) and increments the rest. Cancelled slots refill naturally.
- Metrics: `scheduler/async_level`, `max_off_policy_level` (scheduler.py:501-524).

### 5. Weight sync: NCCL broadcast w/ pause-resume handshake

Trainer side (`src/prime_rl/trainer/rl/broadcast/`):
- Broadcast **every step** (train.py:238-257), blocking, before `wait_for_batch`.
- NCCL: trainer master (rank 0) creates `StatelessProcessGroup` where **trainer = rank 0 and every inference GPU is a rank** (nccl.py:112-163, world = `inference_world_size + 1` at nccl.py:189). Layer-by-layer broadcast of the state dict (nccl.py:34-85).

Handshake (files as signals, no RPC):
1. Trainer writes `STABLE` into `broadcasts/step_{n}/` (nccl.py:209-237).
2. Orchestrator waits for it (scheduler.py:291), then POSTs `/pause` to all engines → touches `NCCL_READY` marker → POSTs `/update_weights` → POSTs `/resume` (`utils/client.py:268-308`).
3. Trainer blocks on `NCCL_READY` before broadcasting (nccl.py:239-245).

Filesystem alternative: HF export + `STABLE` marker (`broadcast/filesystem.py:38-110`); vLLM loads via `/update_weights` endpoint (inference server router, `server.py:197-201`).

DeepEP exists but is **trainer MoE expert-parallelism only** (`configs/trainer.py:21,249-275`), not weight sync.

### 6. Off-policy loss math

`src/prime_rl/trainer/rl/loss.py` (`default_loss_fn`, loss.py:107-163):
- Rollout-time logprobs `inference_logprobs` arrive per-token; trainer recomputes current-policy `trainer_logprobs`.
- `log_importance_ratio = trainer_logprobs - inference_logprobs`; `importance_ratio = exp(...)` (loss.py:137-138).
- DPPO-Binary TV trust-region masking: mask tokens where `exp(trainer_lp) - exp(inference_lp)` exceeds `dppo_mask_high/low` (defaults 0.2/0.2, loss.py:108-135).
- `pg_loss = keep_mask * advantages * ratio` — **unclipped** ratio × advantage (code diverges from async.md's `min(π/μ, δ)` formula; code wins). `kl_loss = loss_mask * log_ratio²` (loss.py:148-150).
- Optional teacher distillation: `advantages = adv_tau*A + teacher_tau*(teacher_lp − trainer_lp).detach()` (loss.py:141-146).

### 7. Data transport: filesystem or ZMQ, no shared memory

- Orchestrator: rollouts → `interleave_rollout` → `TrainingSample`s (msgspec, `transport/types.py:5-27`) → `TrainingBatch{examples, step, run_idx}` → `training_batch_sender.send` (orchestrator.py:564-569).
- Filesystem (default): atomic msgpack `rollouts/step_{n}/train_rollouts.bin` (`transport/filesystem.py:21-30`); trainer-side Packer polls, token-packs, shards to `rank_{i}.bin` per DP rank (packer.py:92/284).
- ZMQ alternative: PUSH/PULL + PUB/SUB (`transport/zmq.py`).
- Trainer `DataLoader.wait_for_batch()` blocks until master packs a batch (data.py:174-183) — the trainer-side backpressure.

### 8. Trainer: torchtitan-style FSDP2 (not Megatron)

- FSDP2 via `torch.distributed.fsdp.fully_shard` (trainer/model.py:394-474); no megatron/nemo imports.
- Per step: broadcast weights → wait_for_batch → micro-batch forward/backward → clip → optimizer.step (train.py:231-622).

## Data Flow (verified call chain)

```
Scheduler.generate_batch(step=n) (scheduler.py:373)
  → schedule_rollout → env.run_rollout → HTTP POST /v1/chat/completions/tokens to vLLM
  → compute_advantages (orchestrator.py:428) → filters (orchestrator.py:431)
  → interleave_rollout → TrainingSample (orchestrator.py:506-547)
  → TrainingBatch(step=n) → FS train_rollouts.bin (atomic)
  → Trainer Packer.pack → MicroBatch grid → rank_{i}.bin
  → DataLoader.wait_for_batch → forward → AIPO loss (ratio math) → backward → optimizer.step
  → broadcast_weights(step=n) → NCCL → vLLM /update_weights (pause/resume handshake)
```

## Gotchas & Pitfalls

1. **Docs are stale** — async.md says k=2 default and `min(π/μ, δ)` clipping; code default is k=1 and the implemented loss is unclipped ratio + DPPO masks. Always trust code here.
2. NCCL weight broadcast is **collective and blocking**; it forces `max_async_level=1`. Deeper overlap would need the filesystem broadcast path.
3. Weight updates **pause the inference pool** (drain in-flight) — async-ness comes from the inflight window between updates, not concurrent updates.
4. The `STABLE`/`NCCL_READY` marker-file protocol is the whole handshake — no distributed state store.
5. Old PRIME Intellect design (forward-only/backward-only executors, SyncReplayBuffer, DeepEP-based sync) is not in this repo anymore — if we port, we port the *new* design.

## Sources (all code, this checkout)

1. `docs/async.md` (async contract, AIPO objective)
2. `src/prime_rl/entrypoints/rl.py` (launcher/GPU split)
3. `src/prime_rl/orchestrator/orchestrator.py` (event loop, setup)
4. `src/prime_rl/orchestrator/scheduler.py` (overlap engine, off-policy culling, policy-update gating)
5. `src/prime_rl/trainer/rl/train.py` (trainer loop, weight broadcast cadence)
6. `src/prime_rl/trainer/rl/broadcast/nccl.py` + `filesystem.py` (weight sync backends)
7. `src/prime_rl/inference/vllm/server.py` + `worker/nccl.py` (custom endpoints, NCCL receiver)
8. `src/prime_rl/utils/client.py` (inference pool, pause/NCCL_READY/resume)
9. `src/prime_rl/trainer/rl/loss.py` (IS ratio, DPPO masks)
10. `src/prime_rl/transport/{types,filesystem,zmq}.py` (batch transport)
11. `src/prime_rl/configs/{orchestrator,trainer,rl,shared}.py` (defaults with line refs)
