# TODOS

Checklist of pending cleanups / follow-ups. Tick items off as they land.

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

## Remove 8-bit AdamW (bitsandbytes)

Only used by the `ddp` backend; dead once FSDP is the only backend.

- [ ] `megatron_trainer/trainer.py:139-146` — drop the `bitsandbytes` import + `bnb.optim.AdamW8bit` branch (FSDP torch AdamW is the only optimizer)
- [ ] `megatron_trainer/config.py:29` — drop the `ddp` = AdamW8bit comment
- [ ] Remove `bitsandbytes` from the container install (`megatron_trainer/train_full.sh:119`) if nothing else uses it

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
