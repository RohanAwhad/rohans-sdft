# Trainer — SDFT Megatron Training Loop (`trainer.py`)

## Role

`megatron_trainer/trainer.py` is the training loop. Launched via
`torchrun --nproc_per_node=N -m megatron_trainer.trainer` (N trainer ranks), it
orchestrates the full on-policy SDFT step:

1. **Rollout** (rank 0 only): generate completions via vLLM (`RagEnv` or
   `ApiAdapterEnv`), build the privileged teacher prompt.
2. **Broadcast** the rollout payload to all ranks.
3. **Teacher log-probs** (every rank independently): frozen/EMA teacher via the
   logprob server's TCP data plane.
4. **Student forward + chunked reverse-KL** via a Megatron-Core `output_processor`
   hook, with gradient accumulation + `no_sync`.
5. **Weight sync** to vLLM + logprob server each optimizer step.
6. **Checkpoints** to `OUTPUT_DIR/step_{N}` / `OUTPUT_DIR/epoch_{N}`.

## Distributed setup & backend contract

- `torchrun` provides `LOCAL_RANK/RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`;
  `init_distributed_trainer()` (`model_utils.py:30`) initializes the nccl process
  group and selects `cuda:{LOCAL_RANK}`.
- **`GRAD_ACCUM_STEPS % world_size == 0`** is asserted (`trainer.py:98`) and
  `local_accum_steps = GRAD_ACCUM_STEPS // world_size` (`trainer.py:102`). Each
  rank trains on a disjoint slice of the step's examples, so the loss math is
  per-rank; cross-rank communication is only for broadcast, loss aggregation,
  and weight sync.
- **Backend**: MCore `TorchFullyShardedDataParallel` (FSDP), wrapped via
  `register_fsdp_module_mappings` (`model_utils.py:138`), which teaches
  the bridge's `AutoMapping` the FSDP-prefixed module class names.
- **Optimizer** (`trainer.py:141`): `torch.optim.AdamW` with `lr=LEARNING_RATE,
  betas=(0.9, 0.95), weight_decay=0.01`.
- **LR scheduler** (`LR_SCHEDULER`, `constant` default): `constant` = fixed
  `LEARNING_RATE` (current behavior, no scheduler object). `cosine` = linear
  warmup for `min(10% of total optimizer steps, 100)` steps, then cosine decay
  to 0 over the remaining steps (`transformers.get_cosine_schedule_with_warmup`).
  Total optimizer steps = `steps_per_epoch × NUM_EPOCHS`, where
  `steps_per_epoch = len(filtered_dataset) // GRAD_ACCUM_STEPS` — so warmup
  depends on the collator's load-time drop filter. Built after dataset load
  (`trainer.py:161-167`) since total steps aren't known at optimizer creation.

## Model loading

- `load_model(HF_MODEL_PATH)` (`model_utils.py:84`) converts HF weights → MCore
  `GPTModel` in-memory via `AutoBridge`; the bridge instance is cached for the
  export passes.
- **Activation recompute** is set on the built model's config (full, uniform,
  `recompute_num_layers=1`, `model_utils.py:118-120`) — required for
  long-completion backward (thinking sequences hit 4k–8k tokens).
- `vocab_size` comes from the **padded model vocab** (`trainer.py:114-115`),
  not the tokenizer's; it sizes the teacher-logprob buffers.
- The HF copy is moved to CPU after load to free GPU memory
  (`model_utils.py:129-132`).

## Data flow — one optimizer step

Rank 0 pulls `GRAD_ACCUM_STEPS` examples per optimizer step; `drop_last=True`
on the DataLoader (`trainer.py:156`). **Before** the DataLoader is built, the
collator runs a **one-time load-time pass** — `filter_dataset` (`trainer.py:154`,
`collator.py:266`) drops over-budget examples and returns a filtered dataset
(`dataset.select(valid_indices)`, `collator.py:294`) that is what the DataLoader
iterates. Over-budget examples never reach the per-batch collator (see
`collator.md`). In `online_feedback` mode, examples with an **empty golden
answer** are also dropped here — reflection requires a golden to grade.

### 1. Rollout (rank 0, `trainer.py:246-314`)

- Builds one env per example (`RagEnv` when `ENV_TYPE=rag`, `ApiAdapterEnv`
  when `ENV_TYPE=api_adapter`) and runs them **concurrently** via
  `ThreadPoolExecutor(max_workers=min(32, len(envs)))`.
- `RagEnv` (`env/rag_env.py:21`): `vllm_generate` → optionally grades via the
  reflector (`use_reflector = HINDSIGHT_FIELD == "online_feedback"`) and rebuilds
  the privileged prompt from feedback (golden chunk optional + golden answer +
  detailed feedback; `env/rag_env.py:60`).
- `ApiAdapterEnv` (`env/api_adapter_env.py:96`): multi-turn adapter↔API loop with
  thinking-budget splitting (`env/api_adapter_env.py:186`); successful adapter
  verdicts are cached in a rank-0 `success_cache` and re-injected as hindsight on
  later failing rollouts (`trainer.py:287-294`).
- Rollout payload (`trainer.py:296-304`): `{prompt_text, completion_text,
  completion_log_probs, privileged_information_prompt}` per example — the
  log-probs are the rollout (vLLM) proposal's per-token logps used for
  importance sampling. A `pass_rate` is computed (episode verdicts for
  api_adapter, reflector verdicts for rag).

### 2. Broadcast & shard (`trainer.py:289-293`)

`dist.broadcast_object_list(rollout_data, src=0)`; each rank slices
`rollout_data[rank*local_accum_steps : (rank+1)*local_accum_steps]`.

### 3. Per-microstep: teacher log-probs (`trainer.py:336-383`)

- `completion_ids` = encode(`completion_text`, no special tokens), skipped if
  empty, truncated to `GEN_MAX_NEW_TOKENS`.
- **Rollout log-probs → IS tensor** (`trainer.py:344-368`, only when
  `IS_WEIGHTING`): `completion_log_probs` → `(C,)` fp32 tensor on device;
  `None` entries become `NaN` (never-sampled tokens, excluded from the IS
  weight); truncated to `len(completion_ids)`, and any residual length gap is
  NaN-padded with a warning.
- `cond_ids` = encode(`privileged_information_prompt`, truncation,
  `max_length=TEACHER_MAX_PROMPT_LEN`).
- `request_teacher_log_probs_tcp(token_ids=cond_ids+completion_ids,
  prompt_len=len(cond_ids), ...)` (`logprob_client.py:84`) returns a
  `(C, vocab_size)` bf16 tensor on device.
  - **Protocol** (`logprob_client.py:97-115`): send `<i prompt_len><i seq_len>
    <seq_len×i ids>`; recv `<i completion_len>` then `completion_len × vocab × 2`
    bytes of fp16, viewed zero-copy into a **preallocated recv buffer**
    (`logprob_client.py:63`, sized `GEN_MAX_NEW_TOKENS × vocab × 2`) and one H2D
    copy to bf16.

### 4. Per-microstep: student forward + reverse-KL (`trainer.py:386-411`)

- `prompt_ids` = encode(`prompt_text`, truncation, `max_length=STUDENT_MAX_PROMPT_LEN`);
  `input_ids = cat([prompt_ids, completion_ids])`, contiguous `position_ids`.
- `make_kl_processor(...)` (`chunked_head.py:113`) builds the MCore
  `output_processor` hook; `model(input_ids, position_ids, attention_mask=None,
  output_processor=kl_processor)` returns `(loss, metrics)`. When the rollout
  log-probs tensor is present (and `IS_WEIGHTING`), the loss is rescaled by the
  per-sequence TIS weight (see `chunked_head.md` — "Importance-sampling
  weighting").

### 5. Backward + gradient accumulation (`trainer.py:413-430`)

- `is_final = (micro_step == local_accum_steps - 1)`; non-final micro-steps run
  under the FSDP-wrapped model's `no_sync()`; `scaled_loss = loss / local_accum_steps`.
- Loss/metrics/sample/completion-len are accumulated.

### 6. Optimizer step (`trainer.py:441-445`)

The FSDP-wrapped model's `finish_grad_sync()` → `clip_grad_norm_(..., MAX_GRAD_NORM=1.0)`
→ `optimizer.step()` → `scheduler.step()` (no-op when `LR_SCHEDULER=constant`).

### 7. Aggregation & logging (`trainer.py:447-489`)

- `[accum_loss_sum, accum_samples]` are `all_reduce`d so rank 0 reports the
  global mean loss.
- `wandb.log` (`train/loss`, `train/completion_length`, `train/lr` — the
  **current scheduler LR** (`scheduler.get_last_lr()[0]`), falling back to
  `LEARNING_RATE` when `LR_SCHEDULER=constant`, `train/grad_norm` — the total
  gradient norm captured from `clip_grad_norm_`'s return value at
  `trainer.py:442`,
  `reflector/pass_rate` or `episode/pass_rate`, sdpo signal metrics, `is/*`
  importance-sampling metrics (see `chunked_head.md`), and — for
  api_adapter every 10 optimizer steps — an `episode/sample` conversation table).
- Per-step `TIMING` line: `total/gen/teacher/student/loss_bwd/optim/wsync`
  (`trainer.py:513-517`).

### 8. Weight sync & checkpoints (`trainer.py:432-448`)

- **Every optimizer step**, **all ranks participate** (the export/gather passes are
  FSDP collectives):
  - `sync_weights_to_logprob_server(model, logprob_comm, rank)` (`logprob_client.py:242`)
    — skipped when `TEACHER_MODEL_PATH` is set (frozen teacher keeps its own
    weights). Under FSDP every rank consumes `gather_raw_params_iter` in
    lockstep (collective all-gather); only rank 0 broadcasts on the logprob NCCL
    group. **EMA blending happens server-side** in the `/sync_weights` handler.
  - `sync_weights_to_vllm(model, device, groups, rank)` (`vllm_utils.py:188`).
    HTTP control sequence per instance (`vllm_utils.py:154-184`): `pause` →
    `start_weight_update` → `update_weights` (thread) → NCCL
    `trainer_send_weights` of the HF-format export iterator → `finish_weight_update`
    → `resume`. Non-zero ranks consume the collective export passes in lockstep.
- **Checkpoints** (`save_hf_checkpoint`, `model_utils.py:220`): every `SAVE_EVERY`
  optimizer steps to `OUTPUT_DIR/step_{N}`; at epoch end to
  `OUTPUT_DIR/epoch_{N}`; a final `step_{optimizer_step}` on completion. Only
  rank 0 writes files (FSDP still requires all ranks in the export pass).
- `dist.barrier()` after weight sync before the next step.

## Chunked LM head (`chunked_head.py`)

Reverse-KL is computed via a chunked LM-head pass as an MCore `output_processor`
hook. The trainer's contract with it: `make_kl_processor(prompt_len,
completion_ids, teacher_log_probs, eos_token_id, device, rollout_log_probs=None,
is_weighting=True, is_cap=2.0)` (`chunked_head.py:113`), then
`model(input_ids, position_ids, attention_mask=None, output_processor=...)`
returns `(loss, metrics)` (`trainer.py:401-411`). Full detail — single head
call, `ChunkedRowKL` analytic backward, memory profile, importance-sampling
weighting, sdpo + `is/*` metrics — in `chunked_head.md`.

## Rollout envs (`env/`)

Rollout is abstracted behind `BaseEnv` (`env/base.py:6`). The trainer builds one
env per example (`RagEnv` for `ENV_TYPE=rag`, `ApiAdapterEnv` for
`ENV_TYPE=api_adapter`), runs them concurrently, then reads `completion_text` +
`privileged_information_prompt` (see §1). Full contract in `envs/base.md`
(+ `envs/ragenv.md`, `envs/api_adapter_env.md`).

## Checkpoint layout

`OUTPUT_DIR` (default `./output`):
- `step_{N}/` — `model.safetensors` + HF `config.json` + `tokenizer_*` at every
  `SAVE_EVERY` optimizer steps and on completion.
- `epoch_{N}/` — same at each epoch end.

## Hyperparameter contract

Defaults read by `config.py` at import time; **Critical** = must be set right or
the job 400s / OOMs / drops examples.

| Var | Default | Critical | Used where |
|---|---|---|---|
| `LEARNING_RATE` | `5e-5` | — | optimizer |
| `LR_SCHEDULER` | `constant` | — | `constant` = fixed LR; `cosine` = linear warmup of `min(10% of total optimizer steps, 100)` then cosine decay to 0 over `steps_per_epoch × NUM_EPOCHS` |
| `BATCH_SIZE` | `1` (fixed) | — | dataloader; effective batch = `1 × GRAD_ACCUM_STEPS` |
| `GRAD_ACCUM_STEPS` | `32` | **yes** | must be `% world_size == 0`; `local_accum_steps = GRAD_ACCUM_STEPS / world_size` |
| `NUM_EPOCHS` | `10` | — | outer loop |
| `MAX_GRAD_NORM` | `1.0` (hardcoded) | — | grad clip before `optimizer.step()` |
| `EMA_ALPHA` | `0.05` | — | server-side teacher EMA (ignored when `TEACHER_MODEL_PATH` set) |
| `STUDENT_MAX_PROMPT_LEN` | `2048` | **yes** | student prompt truncation; `≤ MAX_TOTAL_LEN − GEN_MAX_NEW_TOKENS` |
| `TEACHER_MAX_PROMPT_LEN` | `2048` | **yes** | teacher cond truncation |
| `MAX_TOTAL_LEN` | `8192` | **yes** | import-time guard: `STUDENT + GEN ≤ MAX_TOTAL_LEN` |
| `STUDENT_THINKING` | `0` | — | `1` → thinking on (Qwen `enable_thinking=True`; gpt-oss analysis channel); logged to wandb as `student_thinking` |
| `THINKING_BUDGET` | `512` | api_adapter | adapter thinking split (`env/api_adapter_env.py:182`); logged to wandb as `thinking_budget` |
| `GEN_MAX_NEW_TOKENS` | `MAX_TOTAL_LEN − STUDENT_MAX_PROMPT_LEN` | **yes** | rollout `max_tokens`, completion truncation, TCP recv-buffer size |
| `GEN_TEMPERATURE` / `GEN_TOP_P` | `1.0` / `1.0` | — | `vllm_generate` |
| `IS_WEIGHTING` | `1` | — | `1` → rescale the reverse-KL loss by the per-sequence TIS weight (rollout log-probs vs current policy); `0` → unweighted loss |
| `IS_CAP` | `2.0` | — | TIS truncation cap on the per-token ratio `exp(policy_logp − rollout_logp)` |
| Optimizer betas / wd / eps | `(0.9, 0.95)` / `0.01` / `1e-8` | — | fixed in `trainer.py:141` |

## Hard invariants

- **`GRAD_ACCUM_STEPS % world_size == 0`** — asserted at startup (`trainer.py:98`);
  each rank owns `local_accum_steps` examples per optimizer step.
- **`STUDENT_MAX_PROMPT_LEN + GEN_MAX_NEW_TOKENS ≤ MAX_TOTAL_LEN`** — `config.py`
  raises at import (see `launch_trainer.md`).
- **Weight-sync export passes are FSDP collectives** — every rank must enter
  `export_hf_weights_iter` / `gather_raw_params_iter` in lockstep
  (`model_utils.py:167,181`); only rank 0 does the actual NCCL send + HTTP.
- **Optimizer wraps the unwrapped model's params** — the optimizer is built on
  `model.parameters()`, then the FSDP-wrapped model's `finish_grad_sync()` is
  called before clip/step.
- **Frozen teacher (`TEACHER_MODEL_PATH` set) skips logprob weight sync** —
  otherwise the per-step NCCL + EMA sync runs.

## Known gotchas

- The wandb config `backend` field is **hardcoded `"megatron-bridge-ddp"`**
  (`trainer.py:174`) — a stale label now that the backend is FSDP-only (see `TODOS.md`).
- `RagEnv._build_privileged_prompt_from_feedback` assumes the last message is a
  user message (`env/rag_env.py:59-63`) — with the new OLS format the last
  message is a `tool` message, so online-feedback hints would land inside the
  tool response. Fix (when online_feedback + OLS runs): reuse `collator._append_hint`.
- The TCP logprob recv buffer is sized by `GEN_MAX_NEW_TOKENS` at first
  connection (`logprob_client.py:63`) — a later increase of `GEN_MAX_NEW_TOKENS`
  mid-run is not honored.
- `vllm_generate` timeouts at 180s (`vllm_utils.py:66`); long thinking-budgeted
  generations can hit it.
