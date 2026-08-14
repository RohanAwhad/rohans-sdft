# VERL Async Rollout Engine — Deep Research

> Generated: 2026-08-12 | Repo: `~/3_resources/external_libs/verl` @ `535c4779` (volcengine fork, "v1" trainer era)
> Sources: 100% code (no web) | Tags: #async-rollouts #off-policy #verl #ray #sdft #research

## TL;DR

- This fork is the **v1 trainer era**: upstream's `recipe/one_step_off_policy`, `recipe/fully_async`, `recipe/prime` are gone (migrated to `verl-recipe` submodule); the current path is `trainer.v1.trainer_mode ∈ {sync, colocate_async, separate_async}`.
- Async = **fire-and-forget agent-loop workers + an external TransferQueue KV store**: prompts go in with `status=pending`, rollouts come out keyed `{uid}_{session}_{index}`, the trainer never waits on generation — it polls the queue.
- Rollouts run in `AgentLoopWorker` **Ray actors** (default 8) hitting vLLM/SGLang async **server replicas** (in-process SPMD rollouts are retired).
- **Partial rollout**: when a weight sync aborts in-flight generations, the client silently **resumes** the truncated generation from `prompt + tokens_so_far` (newer policy) and stitches outputs — aborts are invisible to the agent loop (`llm_server.py:404-456`).
- Staleness control: `ReplayBufferAsync` drops (or wait-stalls on) trajectories older than `max_off_policy_threshold` (**8**) weight-versions, then refills with fresh prompts (`replay_buffer.py:503-579`).
- Statistical off-policy correction: full toolkit in `rollout_corr_helper.py` — truncated-IS / IcePop weights, hard rejection-sampling masks, bypass mode, Decoupled PPO with `parameter_sync_step`.

## Overview

VERL (volcengine fork) is a Ray-actorized PPO stack. Everything is a Ray actor; the driver builds colocated worker groups (actor+critic+rollout in one process per GPU) plus async LLM server replicas, and v1 trainers decouple generation from training by pushing prompts into a TransferQueue and polling finished trajectories. Three trainer modes offer progressively more overlap: `sync` (rollout sleeps while training), `colocate_async` (rollout aborts+resumes around training), `separate_async` (dedicated standalone rollout GPUs + hybrid-engine mode switching on trainer GPUs).

For our SDFT trainer the relevant ideas are: (1) producer/consumer decoupling via a queue, (2) weight-version stamps on trajectories for staleness eviction, (3) abort-resume partial rollouts, (4) the IS/rejection-sampling correction toolbox.

## Key Findings

### 1. Entry points and trainer registry

- `verl/trainer/main_ppo.py:167` — Hydra entry; dispatches on `config.trainer.use_v1` (main_ppo.py:184-193). `TaskRunnerV1` **force-enables** `transfer_queue.enable = True` (main_ppo.py:144).
- Trainer registry (`verl/trainer/ppo/v1/trainer_base.py:1835-1862`): `sync` → `PPOTrainerSync`, `colocate_async` → `PPOTrainerColocateAsync`, `separate_async` → `PPOTrainerSeparateAsync` (selected by `trainer.v1.trainer_mode`, `config/ppo_trainer.yaml:227-228`).
- `TaskRunnerV1.init_agent_loop_manager` (main_ppo.py:112-132) builds `AgentLoopManagerTQ` with `llm_client`, teacher client, reward-loop handles; then `trainer.fit(agent_loop_manager)` (main_ppo.py:153-156).

### 2. Worker topology

- `ActorRolloutRefWorker` (`verl/workers/engine_workers.py:446-817`): actor + ref + rollout engine colocated in one process per GPU. `update_weights()` (engine_workers.py:719-805): mode `naive` = direct in-process sync (rollout `resume` → param generator → `rollout.update_weights` → resume KV cache); any other mode = `CheckpointEngine.send_weights(...)` for disaggregated transfer.
- `RayWorkerGroup` (1 actor per GPU) + method dispatch via `Dispatch.ONE_TO_ALL` / `DP_COMPUTE` decorators; async execution via `blocking=False` → Ray ObjectRefs (engine_workers.py:241).

### 3. Agent loops: fire-and-forget + TransferQueue

- `AgentLoopWorker` (`verl/experimental/agent_loop/agent_loop.py:497-1141`): `generate_sequences` spawns one asyncio task **per sample** (agent_loop.py:656-667), each running `SingleTurnAgentLoop.run` → `LLMServerClient.generate` → sticky/least-loaded server replica via `GlobalRequestLoadBalancer` (llm_server.py:262-283).
- V1 variant `AgentLoopWorkerTQ.generate_sequences` (`verl/trainer/ppo/v1/agent_loop_tq.py:59-105`) — **fire-and-forget**: creates background tasks, returns immediately. Results are published per output via `tq.async_kv_batch_put` under key `{uid}_{session_id}_{index}` with tags `{status, prompt_len, response_len, seq_len, global_steps, min/max_global_steps}` (agent_loop_tq.py:193-227). `min/max_global_steps` = weight-version stamps for staleness bookkeeping.
- `AgentLoopManagerTQ.generate_sequences` (agent_loop_tq.py:243-257): chunk + dispatch, returns `None` — outputs arrive via the queue. This **replaces** upstream's `DataProtoFuture`; the trainer overlaps compute by never waiting on generation.

### 4. Trainer sampling: ReplayBuffer over TransferQueue

`verl/trainer/ppo/v1/replay_buffer.py`:

- `PPOTrainer._step_once()` (trainer_base.py:536-586): `replay_buffer.sample(...)` **blocks** until enough finished groups exist — the consumer-side sync point.
- `ReplayBufferAsync.sample` (replay_buffer.py:541-579): loop of `_sync_metadata_from_transfer_queue` → evict `stale ∪ DAPO ∪ failed` → `refill_fn(len(evicted))` → select oldest-finished by `prompt_global_steps`.
- Staleness eviction (replay_buffer.py:503-512): a prompt is stale when `global_steps - prompt_global_steps[uid] + 1 > max_off_policy_threshold` (default **8**, `ppo_trainer.yaml:252`).
- Strategy `drop` (default): evict stale + refill. Strategy `wait` (replay_buffer.py:524-539): **blocks sampling** while any pending/running prompt has hit the threshold, so it finishes and gets trained instead of dropped.
- Assert: with `drop`, every selected span ≤ threshold (replay_buffer.py:570-577).
- Warmup: async trainers submit `num_warmup_batches` (=1, `ppo_trainer.yaml:237/243`) extra prompt batches at start so generation runs ahead (`trainer_colocate_async.py:40-46`, `trainer_separate_async.py:138-144`).

### 5. Partial rollout: abort-resume client

`FullyAsyncLLMServerClient.generate` (`verl/workers/rollout/llm_server.py:345-461`) — the key overlap primitive:

1. On each attempt, calls `super().generate(prompt_ids + final_output.token_ids, ...)` (llm_server.py:404-415) — resume from prompt + tokens generated so far.
2. Merges tokens/logprobs/routing into `final_output` (llm_server.py:417-432); tracks `min/max_global_steps` across attempts (llm_server.py:434-438).
3. Recomputes remaining budget `original_max_tokens - len(generated)` (llm_server.py:441-445).
4. If `stop_reason in ("aborted","abort")` → sleep 1s and retry (llm_server.py:447-456). For v1 trainer `should_retry` is **always True** (comment at llm_server.py:449).

Server side: `abort_all_requests` = `engine.pause_generation(wait_for_inflight_requests=False, clear_cache=...)` (vllm_async_server.py:852-884). Trainer callbacks drive it: `PPOTrainerColocateAsync.on_sample_end()` = `abort_replicas()` then `sleep_replicas()` (trainer_colocate_async.py:55-59); `on_step_end()` = `update_weights` + `resume_generation_replicas()` (trainer_colocate_async.py:48-53).

### 6. Weight sync to rollout servers

`verl/checkpoint_engine/base.py` — `CheckpointEngineManager.update_weights(global_steps)` (base.py:486-529):

- `naive` (colocated): direct in-process sync, replicas sleep/wake around it (base.py:494-496).
- Disaggregated (`nccl`/`nixl`/`mooncake`/`delta_sharded`): **abort in-flight** → build temp WorkerGroup over replica workers → **release KV cache** (so NCCL writes into weight buffers) → `build_process_group` (NCCL topology, base.py:403-428) → actor `send_weights` + rollout `receive_weights` → finalize → resume KV cache → **resume generation** (base.py:499-536).
- Trainer engines export `(name, tensor)` generators: FSDP DTensor `full_tensor()` (fsdp/transformer_impl.py:949-1008) or Megatron-Bridge `export_hf_weights` (megatron/transformer_impl.py:1015-1045).
- vLLM transfer: bucketed ZMQ/CUDA-IPC `BucketedWeightSender` (vllm_rollout.py:222-234); SGLang: HTTP `update_weights_from_tensor` per bucket (sglang_rollout.py:344-364).

### 7. Off-policy correction toolbox

`verl/trainer/ppo/rollout_corr_helper.py`:

- `compute_rollout_correction_weights` (rollout_corr_helper.py:522-657): token-level `exp(clamp(log_ratio, ±20))`; sequence-level `exp(masked_sum(log_ratio))`; **TIS** clamp to threshold; **IcePop** zero-out outside `[lower, upper]`; optional batch-normalize to mean 1.0.
- `compute_rollout_rejection_mask` (rollout_corr_helper.py:197-413): hard trust region — token divergences `k1=-log r`, `k2=½(log r)²`, `k3=r−1−log r`; seq-level sum/mean/max variants ANDed into `modified_response_mask = response_mask × final_mask`.
- Entry: `compute_rollout_correction_and_add_to_batch` (rollout_corr_helper.py:1013-1067) — always overwrites `response_mask`; called from `_compute_advantage` (trainer_base.py:1607-1617). IS weights consumed in `core_algos.py` loss functions (e.g. core_algos.py:1364-1365).
- **Bypass mode**: `old_log_probs = rollout_log_probs` — zero-cost old-logprob recompute (rollout_corr_helper.py:1109-1144; trainer_base.py:1485-1493).
- **Decoupled PPO** (3 policies π_rollout / π_old / π_θ): `parameter_sync_step` (default 4 in separate_async, `ppo_trainer.yaml:246`) mini-batch updates per weight sync; π_old anchored via CPU save/restore (`trainer_separate_async.py:103-127`).

### 8. separate_async: the current flagship recipe

`verl/trainer/ppo/v1/trainer_separate_async.py:39-207`:
- Requires non-naive checkpoint backend (nccl/nixl/mooncake/delta, `:59-61`) and standalone rollout GPUs (`rollout.nnodes > 0`).
- Two LLMServerManagers: hybrid (on trainer GPUs) + **standalone rollout replicas** (trainer_separate_async.py:86-89); two CheckpointEngineManagers (`:91-97`).
- **Hybrid engine mode switching**: trainer GPUs join/leave the rollout balancer when idle — `switch_to_rollout()`/`switch_to_trainer()` (`:180-203`); the actual switch policy (`should_switch_to_rollout`) is a **stubbed TODO** (`:205-207`).
- Rollout metrics include standalone GPUs in the throughput denominator (`:167-178`).

## Data Flow (verified call chain)

```
dataloader → _submit_batch_to_rollout (tq.kv_batch_put uid→status=pending)
  → AgentLoopManagerTQ.generate_sequences (fire-and-forget)
  → AgentLoopWorkerTQ._run_prompt (status=running)
  → SingleTurnAgentLoop.run → FullyAsyncLLMServerClient.generate
    (on abort: resume from prompt+generated tokens, stitch, retry)
  → _agent_loop_postprocess → tq.async_kv_batch_put ({uid}_{sess}_{idx}, staleness tags)
  → ReplayBufferAsync.sample (evict stale > threshold=8 → refill; pick oldest-finished)
  → KVBatchMeta → old_log_prob → ref → values → advantage
    → compute_rollout_correction_and_add_to_batch (IS weights + RS mask)
  → update_actor → checkpoint_manager.update_weights
    (abort → release KV → NCCL send/receive → resume generation)
```

## Gotchas & Pitfalls

1. **File layout is version-dependent** — `agent_loop_manager.py`, `megatron_workers.py`, `ray_async_trainer.py` don't exist in this fork; v1 consolidated them. Any verl docs/blogs referencing them are stale.
2. `transfer_queue.enable: False` in yaml but **force-enabled** by `TaskRunnerV1.run` (main_ppo.py:144) — TransferQueue (Ascend's kv store lib) is a hard dependency of v1.
3. vLLM in-process SPMD rollouts are retired — `vllm_rollout.generate_sequences` raises `NotImplementedError` (vllm_rollout.py:252-267). All generation is server-based.
4. Async sampling correctness depends on the weight-version stamps (`min/max_global_steps`) — they flow through `output.extra_fields` and must survive postprocessing.
5. The `wait` staleness strategy can deadlock-ish block if generation stalls; `drop` silently wastes compute on discarded rollouts. Threshold=8 balances both.
6. `separate_async`'s hybrid-engine switching (the nice part — trainer GPUs serving rollouts when idle) is **not implemented** yet (`should_switch_to_rollout` returns False).

## Sources (all code, this checkout)

1. `verl/trainer/main_ppo.py` (entry, TaskRunnerV1, force-enable TQ)
2. `verl/trainer/ppo/v1/trainer_base.py` (registry, fit/step/_step_once)
3. `verl/trainer/ppo/v1/trainer_sync.py`, `trainer_colocate_async.py`, `trainer_separate_async.py`
4. `verl/trainer/ppo/v1/agent_loop_tq.py` (fire-and-forget dispatch, TQ publication)
5. `verl/trainer/ppo/v1/replay_buffer.py` (async sampler, staleness eviction)
6. `verl/experimental/agent_loop/agent_loop.py` (AgentLoopWorker, AgentLoopManager)
7. `verl/workers/rollout/llm_server.py` (LLMServerClient, FullyAsyncLLMServerClient, LLMServerManager)
8. `verl/workers/engine_workers.py` (ActorRolloutRefWorker, update_weights)
9. `verl/checkpoint_engine/base.py` (CheckpointEngineManager, abort/resume flow)
10. `verl/trainer/ppo/rollout_corr_helper.py` (IS weights, rejection sampling, bypass)
11. `verl/trainer/config/ppo_trainer.yaml` (v1 config defaults)
