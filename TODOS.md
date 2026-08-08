# TODOS

Checklist of pending cleanups / follow-ups. Tick items off as they land.

## Make student thinking configurable (`STUDENT_THINKING`)

Spec: `docs/megatron_trainer/thinking.md`. Currently fully hardcoded — Qwen renders
with `enable_thinking=False` (collator), gpt-oss forces the **final** channel
(`<|channel|>final<|message|>`, commit `475db67`), api_adapter hardcodes
`enable_thinking=True`. No env knob exists.

- [ ] `megatron_trainer/config.py` — add `STUDENT_THINKING` (default `"0"`, truthy `"1"`); raise at import if `STUDENT_THINKING=1` and model is neither Qwen nor gpt-oss
- [ ] `megatron_trainer/collator.py:27,32` — channel suffix selectable: `analysis` (`<|channel|>analysis<|message|>`, thinking) vs `final` (current), driven by `IS_GPT_OSS + STUDENT_THINKING`
- [ ] `megatron_trainer/collator.py:122` — `enable_thinking=False` → `enable_thinking=STUDENT_THINKING` (Qwen path; applies to both student and teacher renders — must stay in sync)
- [ ] `megatron_trainer/env/rag_env.py:69` — `enable_thinking=False` → flag-gated
- [ ] `megatron_trainer/env/api_adapter_env.py:193,302,324` — replace hardcoded `enable_thinking=True` with the flag (note: default-off flips api_adapter behavior — verify two-phase `THINKING_BUDGET` split on default path)
- [ ] `megatron_trainer/env/rag_env.py:48` — thinking-on truncated CoT: force-close `</think>` + continue (reuse api_adapter two-phase pattern via shared `vllm_utils` helper)
- [ ] `megatron_trainer/train_full.sh` — add `-e STUDENT_THINKING=...` and `-e THINKING_BUDGET=...` passthrough (THINKING_BUDGET currently never reaches the container)
- [ ] `megatron_trainer/trainer.py:171-186` — wandb config: add `student_thinking`, `thinking_budget` (merges with the "Expand wandb run config" item below)
- [ ] Update stale docs: `collator.md:28,85` (say gpt-oss forces analysis channel — code is final since `475db67`), `launch_trainer.md:233` ("no launcher knob" no longer true)

## Make `MODEL_NAME` / `TRAIN_DATA_PATH` required (no defaults)

Spec updated (`docs/megatron_trainer/launch_trainer.md`): both are required with no default; the
script must **fail fast** if either is unset. Code still has defaults — change later.

- [ ] `megatron_trainer/train_full.sh:49` — `MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}` → require it, fail if unset
- [ ] `megatron_trainer/train_full.sh:94` — `TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/workspace/.../train_sdft.jsonl}"` → require it, fail if unset
- [ ] `megatron_trainer/config.py:5` — `MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")` → raise if unset
- [ ] `megatron_trainer/config.py:77-80` — `TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "...")` → raise if unset

## Make `TRAINER_BACKEND` default to `fsdp`

Spec updated (`docs/megatron_trainer/launch_trainer.md`). Code defaults still `ddp` — change later.

- [ ] `megatron_trainer/train_full.sh:105` — `-e TRAINER_BACKEND="${TRAINER_BACKEND:-ddp}"` → default `fsdp`
- [ ] `megatron_trainer/config.py:30` — `TRAINER_BACKEND = os.environ.get("TRAINER_BACKEND", "ddp")` → default `fsdp`
- [ ] Smoke with no `TRAINER_BACKEND` set (FSDP path, bitsandbytes no longer loaded)

## Remove DDP trainer backend + 8-bit AdamW (FSDP-only)

Spec updated (`docs/megatron_trainer/trainer.md`, `launch_trainer.md`): only `fsdp` is
documented. Code still has the `ddp`/`bitsandbytes` path — remove later.

- [ ] `megatron_trainer/trainer.py` — drop the `TRAINER_BACKEND` if/else branches (wrap `:122-137`,
      optimizer `:139-146`, `finish_grad_sync` `:382-383`, weight-sync/ckpt rank guards `:434,469,474`);
      hardcode the MCore FSDP path
- [ ] Rename the `ddp_model` handle to the FSDP-wrapped model
- [ ] `megatron_trainer/config.py:29-30` — drop `TRAINER_BACKEND` env read + `ddp` comment
- [ ] `megatron_trainer/train_full.sh:105` — drop `-e TRAINER_BACKEND` passthrough
- [ ] Remove `bitsandbytes` from the container install (`megatron_trainer/train_full.sh:119`)

## Log `train/grad_norm` to wandb

Spec updated (`docs/megatron_trainer/trainer.md` §7). Code doesn't capture it yet.

- [ ] `megatron_trainer/trainer.py:384` — capture the return value: `grad_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)`
- [ ] `megatron_trainer/trainer.py:404-409` — add `"train/grad_norm": grad_norm` to `log_dict`

## Expand wandb run config

Run config (`trainer.py:171-186`) omits several knobs and hardcodes `backend`. Add them:

- [ ] `megatron_trainer/trainer.py:174` — replace hardcoded `"backend": "megatron-bridge-ddp"` with `TRAINER_BACKEND`
- [ ] `megatron_trainer/trainer.py:171-186` — add to `config`: `max_grad_norm` (`MAX_GRAD_NORM`), `max_total_len` (`MAX_TOTAL_LEN`), `student_max_prompt_len` (`STUDENT_MAX_PROMPT_LEN`), `teacher_max_prompt_len` (`TEACHER_MAX_PROMPT_LEN`), `thinking_budget` (`THINKING_BUDGET`), `ema_alpha` (`EMA_ALPHA`), `teacher_model` (`TEACHER_MODEL_PATH`)

## Remove `LOGPROB_BATCH_SIZE` (server-side batching)

Removed from spec (`docs/megatron_trainer/launch_trainer.md`). Code usage still present — remove later.

- [ ] `megatron_trainer/train_full.sh:106` — drop `-e LOGPROB_BATCH_SIZE=...` passthrough
- [ ] `megatron_trainer/logprob_server.py:51` — drop the env read
- [ ] `megatron_trainer/logprob_server.py:270,282-283` — `/logprobs_batch` no longer chunks; process the full batch in one pass
- [ ] Verify single-pass batching holds on OLS-sized batches (GPU mem bound)

## Spec/code drift: `--max-model-len`

Spec now says `--max-model-len $MAX_TOTAL_LEN` (`docs/megatron_trainer/launch_trainer.md`), code still hardcodes 16384.

- [ ] `megatron_trainer/train_full.sh:142` — `--max-model-len 16384` → `--max-model-len "$MAX_TOTAL_LEN"`
- [ ] Smoke: defaults (8192) still fine; OLS set `MAX_TOTAL_LEN=16384`

## Port-scan race: 30s stagger

vLLM EngineCore scans free ports; concurrent starts collide. Currently mitigated by `sleep 30` (doc note: hard constraint, revisit).

- [ ] Set per-instance `VLLM_PORT` env scan base so each EngineCore scans a disjoint range
- [ ] If disjoint ranges hold, drop the `sleep 30` stagger
