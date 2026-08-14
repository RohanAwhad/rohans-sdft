# TODOS

Checklist of pending cleanups / follow-ups. Tick items off as they land.

## Async rollouts (producer thread, N-ahead window)

Spec: `docs/megatron_trainer/async_rollouts.md` (research: `docs/research/async_rollouts_porting_analysis.md`).
Feature flag `ASYNC_ROLLOUT` (default `0`, off); sync path preserved byte-identical.

- [x] `megatron_trainer/config.py` — `ASYNC_ROLLOUT = os.environ.get("ASYNC_ROLLOUT", "0") == "1"` (+ `N_ASYNC` in-flight bound, default `2*GRAD_ACCUM_STEPS`)
- [x] `megatron_trainer/trainer.py` — rank-0 producer thread; bounded `queue.Queue(maxsize=N_ASYNC+world_size+1)`; per-sample streaming pushes; `_ROLLOUT_SENTINEL` epoch-end; main path `get()` → broadcast → train → sync; meta (`full_pass_rate`, `reflector_fallback_count`, `success_cache`) ships with the batch; TIMING gains `producer_wait`/`gen_overlap`; wandb config gains `async_rollout` (supersedes the original `Queue(1)` + `Event`/`gen_ready.set()` design — see async_rollouts.md)
- [x] `megatron_trainer/train_full.sh` — `-e ASYNC_ROLLOUT` + `-e N_ASYNC` passthrough
- [x] Docs — `docs/megatron_trainer/async_rollouts.md` (design, layers, verdicts) + README §streaming async rollouts
- [x] Smoke: sync run (flag off, byte-identical) → async run (flag on) → compare TIMING lines; verify IS weights sane with off-by-one policies; no collectives in thread (hang = bug) — full Layer 1-3 campaign + eval table in async_rollouts.md
- [ ] Future work (spec "Future work"): staleness-aware drop / DPPO masks; partial-rollout resume; separate orchestrator; teacher version stamps (N-ahead window implemented; `policy_version`/`policy_lag` landed)

## Make student thinking configurable (`STUDENT_THINKING`)

Knob: `STUDENT_THINKING` (default `"0"`), truthy `"1"`. One flag flips **both** student and
teacher renders (reverse-KL alignment). Specs: `docs/megatron_trainer/collator.md`,
`docs/megatron_trainer/launch_trainer.md`, `docs/megatron_trainer/envs/ragenv.md`.

- [x] `megatron_trainer/config.py` — `STUDENT_THINKING` env read + import guard (only Qwen/gpt-oss validated)
- [x] `megatron_trainer/collator.py` — `enable_thinking=STUDENT_THINKING` (`_render_tokens`), channel suffix `analysis` vs `final` selected by `IS_GPT_OSS + STUDENT_THINKING` (`_append_channel`)
- [x] `megatron_trainer/env/rag_env.py:69` — rebuild render flag-gated
- [x] `megatron_trainer/train_full.sh` — `-e STUDENT_THINKING` + `-e THINKING_BUDGET` passthrough
- [x] `megatron_trainer/trainer.py:171-186` — wandb `student_thinking` / `thinking_budget`
- [x] Docs updated (collator.md, launch_trainer.md, ragenv.md, trainer.md)
- [ ] **Deferred — two-phase truncated CoT in rag_env.** Thinking-on generation is still a single vLLM pass (`RagEnv.run()`); long CoT can truncate mid-think with no closing tag/answer. Add shared `vllm_utils` helper (api_adapter's two-phase pattern): phase 1 `max_tokens=THINKING_BUDGET`, force-close `</think>` on `finish_reason=="length"`, phase 2 with remaining budget. Add `config.py` guard `1 ≤ THINKING_BUDGET < GEN_MAX_NEW_TOKENS`.
- [ ] **Deferred — api_adapter_env.** `enable_thinking=True` still hardcoded (lines 193, 302, 324); flag-gate + verify two-phase split on the default-off path.

## Make `MODEL_NAME` / `TRAIN_DATA_PATH` required (no defaults)

Spec updated (`docs/megatron_trainer/launch_trainer.md`): both are required with no default; the
script must **fail fast** if either is unset. Code still has defaults — change later.

- [x] `megatron_trainer/train_full.sh:49` — `MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}` → require it, fail if unset (`:?` guard)
- [x] `megatron_trainer/train_full.sh:97` — `TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/workspace/.../train_sdft.jsonl}"` → require it, fail if unset (`:?` guard)
- [x] `megatron_trainer/config.py:5` — `MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")` → raise if unset
- [x] `megatron_trainer/config.py:77-80` — `TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "...")` → raise if unset

## Make `TRAINER_BACKEND` default to `fsdp`

Spec updated (`docs/megatron_trainer/launch_trainer.md`). Code defaults still `ddp` — change later.

- [x] `megatron_trainer/train_full.sh:105` — `-e TRAINER_BACKEND="${TRAINER_BACKEND:-ddp}"` → default `fsdp` (superseded by removal below)
- [x] `megatron_trainer/config.py:30` — `TRAINER_BACKEND = os.environ.get("TRAINER_BACKEND", "ddp")` → default `fsdp` (superseded by removal below)
- [x] Smoke with no `TRAINER_BACKEND` set (FSDP path, bitsandbytes no longer loaded)

## Remove DDP trainer backend + 8-bit AdamW (FSDP-only)

Spec updated (`docs/megatron_trainer/trainer.md`, `launch_trainer.md`): only `fsdp` is
documented. Code still has the `ddp`/`bitsandbytes` path — remove later.

- [x] `megatron_trainer/trainer.py` — drop the `TRAINER_BACKEND` if/else branches (wrap,
      optimizer, `finish_grad_sync`, weight-sync/ckpt rank guards); hardcode the MCore FSDP path
- [x] Rename the `ddp_model` handle to the FSDP-wrapped model (`fsdp_model`)
- [x] `megatron_trainer/config.py:29-30` — drop `TRAINER_BACKEND` env read + `ddp` comment
- [x] `megatron_trainer/train_full.sh:105` — drop `-e TRAINER_BACKEND` passthrough
- [x] Remove `bitsandbytes` from the container install (`megatron_trainer/train_full.sh:119`)
- [x] wandb `backend` config hardcoded to `fsdp` (supersedes TODO 6 item 1)

## Log `train/grad_norm` to wandb

Spec updated (`docs/megatron_trainer/trainer.md` §7). Code doesn't capture it yet.

- [x] `megatron_trainer/trainer.py:435` — capture the return value: `grad_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)`
- [x] `megatron_trainer/trainer.py:457-468` — add `"train/grad_norm"` (global norm via all-reduced norm²) to `log_dict` + per-step log line

## Expand wandb run config

Run config (`trainer.py:171-186`) omits several knobs and hardcodes `backend`. Add them:

- [x] `megatron_trainer/trainer.py` — `"backend": "fsdp"` (done via FSDP-only change)
- [x] `megatron_trainer/trainer.py:183-208` — add to `config`: `max_grad_norm` (`MAX_GRAD_NORM`), `max_total_len` (`MAX_TOTAL_LEN`), `student_max_prompt_len` (`STUDENT_MAX_PROMPT_LEN`), `teacher_max_prompt_len` (`TEACHER_MAX_PROMPT_LEN`), `thinking_budget` (`THINKING_BUDGET`), `ema_alpha` (`EMA_ALPHA`), `teacher_model` (`TEACHER_MODEL_PATH`)

## Remove `LOGPROB_BATCH_SIZE` + batch logprob endpoint

Removed from spec (`docs/megatron_trainer/launch_trainer.md`). Batch path is unused
(trainer uses per-rank TCP) — deleted entirely.

- [x] `megatron_trainer/train_full.sh` — drop `-e LOGPROB_BATCH_SIZE=...` passthrough
- [x] `megatron_trainer/logprob_server.py` — drop the env read, `BatchLogprobRequest`, and the whole `/logprobs_batch` endpoint
- [x] `megatron_trainer/logprob_client.py` — delete `request_teacher_log_probs_batch_http`; `megatron_trainer/trainer.py` — drop its import

## Spec/code drift: `--max-model-len`

Spec now says `--max-model-len $MAX_TOTAL_LEN` (`docs/megatron_trainer/launch_trainer.md`), code still hardcodes 16384.

- [x] `megatron_trainer/train_full.sh:162` — `--max-model-len 16384` → `--max-model-len "$MAX_TOTAL_LEN"` (passthrough already in place at `:123`)
- [ ] Smoke: defaults (8192) still fine; OLS set `MAX_TOTAL_LEN=16384`

## Port-scan race: 30s stagger

vLLM EngineCore scans free ports; concurrent starts collide. Currently mitigated by `sleep 30` (doc note: hard constraint, revisit).

- [ ] Set per-instance `VLLM_PORT` env scan base so each EngineCore scans a disjoint range
- [ ] If disjoint ranges hold, drop the `sleep 30` stagger
