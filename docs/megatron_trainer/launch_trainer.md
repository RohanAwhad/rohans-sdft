# Launch — SDFT Training Job (`train_full.sh`)

## Role

`megatron_trainer/train_full.sh` is the canonical launcher. One invocation builds a single
NeMo container (`nvcr.io/nvidia/nemo:26.06`) that hosts the full SDFT stack on M+N+1 H100 GPUs:

- **M vLLM servers** — rollouts (generation)
- **N trainer ranks** — Megatron-Core student + reverse-KL training (via `torchrun`)
- **1 logprob server** — frozen-teacher log-probs

## GPU layout contract

```
bash megatron_trainer/train_full.sh [GPU_START=3] [NUM_TRAINERS=2] [NUM_VLLM_GPUS=1]
```

| Role | Host GPUs (`--device nvidia.com/gpu=`) | Container `CUDA_VISIBLE_DEVICES` |
|---|---|---|
| vLLM instance `i` (0..M-1) | `GPU_START + i` | `i` |
| Trainer ranks (N) | `GPU_START + M` .. `GPU_START + M + N - 1` | `TRAINER_FIRST` (=M) .. `TRAINER_LAST` (=NUM_GPUS−2), via `torchrun --nproc_per_node=N` |
| Logprob server | `GPU_START + M + N` | `NUM_GPUS − 1` (last visible GPU) |

Example (all 6 GPUs, `train_full.sh 0 2 3`): vLLM=0,1,2 → trainer=3,4 → logprob=5.

Host GPU order is preserved into the container, so `--device` order = `CUDA_VISIBLE_DEVICES` order.

**Ports:** `VLLM_PORT` (default 8001) is the base; instance `i` gets `VLLM_PORT + i*100` (8001, 8101, 8201).
Each vLLM also gets a NCCL `--master-port VLLM_NCCL_MASTER_PORT_BASE + i*100` (default base 29500).
Logprob: HTTP `LOGPROB_PORT` (default 8010), TCP `LOGPROB_TCP_PORT` (default 8011). Ports are spaced
100 apart so vLLM's internal ports don't collide.

**Parallel runs on one node:** `GPU_START` already selects a distinct GPU range. To co-locate a
second job, give it a distinct port block — `VLLM_PORT`, `VLLM_NCCL_MASTER_PORT_BASE`,
`LOGPROB_PORT`, and `LOGPROB_TCP_PORT` are all overridable via env. Example (GPUs 3/4/5, ports
8101/29600/8020/8021):

```
VLLM_PORT=8101 VLLM_NCCL_MASTER_PORT_BASE=29600 LOGPROB_PORT=8020 LOGPROB_TCP_PORT=8021 \
  bash megatron_trainer/train_full.sh 3 2 1
```

## Component contract

### 1. vLLM servers (start first, staggered)

- Launched inside the container per instance: `python /workspace/megatron_trainer/start_vllm_patched.py`
  (patches the prometheus `_IncludedRouter` crash and no-ops `initialize_layerwise_reload` for
  gpt-oss fused params).
- Args: `--max-model-len $MAX_TOTAL_LEN --dtype bfloat16 --gpu-memory-utilization 0.8 --weight-transfer-config '{"backend":"nccl"}' --enforce-eager --no-enable-log-requests`.
  `--max-model-len` is **derived from `MAX_TOTAL_LEN`**, not hardcoded (see **Hard invariants**).
  Since `config.py` enforces `STUDENT_MAX_PROMPT_LEN + GEN_MAX_NEW_TOKENS ≤ MAX_TOTAL_LEN`, this
  guarantees vLLM never HTTP-400s a rollout (`prompt + max_tokens ≤ --max-model-len`).
  Each instance `i` is launched with `--master-port $((VLLM_NCCL_MASTER_PORT_BASE + i*100))` for its
  NCCL weight-transfer engine (see **Ports** above).
- `VLLM_SERVER_DEV_MODE=1` is required — it exposes the dev endpoints (`/init_weight_transfer_engine`,
  `/update_weights`, `/pause`/`/resume`) the trainer uses for per-step weight sync.
- Instances start **30s apart** (`sleep 30` after each) to avoid the port-scan race.
  - (rohan): This right now is a hard constraint. Will check up on how to solve this later # TODO
- Readiness: `curl /v1/models`, polled up to 120×2s per instance.
- Log: `logs/vllm_$i.log`.

### 2. Logprob server

- `python -m megatron_trainer.logprob_server` on the last GPU.
- No readiness gate beyond process start (trainer waits for it via `wait_for_logprob_server`).
- Log: `logs/logprob_server.log`.

### 3. Trainer (`torchrun`)

- `CUDA_VISIBLE_DEVICES=<TRAINER_FIRST..TRAINER_LAST> torchrun --nproc_per_node=N -m megatron_trainer.trainer`
- Runs in the foreground of the container; the podman run tees everything to `logs/training.log`.
- The trainer also writes its own debug-level log to `logs/trainer.log` (in-container
  `/workspace/logs/trainer.log`, visible on the host via the workspace mount).

Order: all vLLM instances ready → logprob started → trainer. On completion, vLLM + logprob PIDs are killed.

## Env-var contract

Every var below is passed into the container with `-e` by `train_full.sh`; `config.py` reads them at
import time (fail-fast on invalid combos). **Critical** = must be set right or the job 400s/OOMs/drops examples.

### Model & teacher

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `MODEL_NAME` | **(required — no default)** | **(required — no default)** | **yes** | drives `IS_GPT_OSS` / `IS_QWEN` (family gates); `train_full.sh` fails fast if unset |
| `HF_MODEL_PATH` | `=$MODEL_NAME` | `=MODEL_NAME` | — | trainer load path |
| `TEACHER_MODEL_PATH` | `(empty)` | `(empty)` | — | set → frozen external teacher (mxfp4 auto-loads in 4-bit), NCCL teacher sync disabled; empty → teacher = EMA-blended student |

### Prompt budgets & context (high stakes)

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `STUDENT_MAX_PROMPT_LEN` | `1024` | `1024` | **yes** | OLS needs **14336**; 1024 is the conservative default |
| `TEACHER_MAX_PROMPT_LEN` | `2048` | `2048` | **yes** | OLS needs **15360**; 2048 drops most enriched examples |
| `GEN_MAX_NEW_TOKENS` | `6144` | `MAX_TOTAL_LEN − STUDENT_MAX_PROMPT_LEN` | **yes** | OLS uses 1024 |
| `MAX_TOTAL_LEN` | `8192` | `8192` | **yes** | `STUDENT + GEN ≤ MAX_TOTAL_LEN` or `config.py` raises at import; OLS needs ≥ 15360 (use 16384). **Also sets vLLM `--max-model-len`** — see **Hard invariants** |

Budget arithmetic (OLS): 14,336 + 1,024 = 15,360 ≤ 16,384 = `MAX_TOTAL_LEN = --max-model-len`. vLLM
hard-rejects any prompt where prompt + `max_tokens` > `--max-model-len` (HTTP 400) — this was the
original OLS failure. Because `--max-model-len` now tracks `MAX_TOTAL_LEN`, the `config.py` guard
(`STUDENT + GEN ≤ MAX_TOTAL_LEN`) is sufficient to prevent it.

### Training

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `NUM_EPOCHS` | `10` | `10` | — | |
| `GRAD_ACCUM_STEPS` | `32` | `32` | **yes** | must be divisible by num_trainers (assert in trainer.py); effective batch = `BATCH_SIZE(=1) × GRAD_ACCUM_STEPS` |
| `SAVE_EVERY` | `200` | `200` | — | optimizer steps between checkpoints to `OUTPUT_DIR/step_{N}` |
| `LEARNING_RATE` | `5e-5` | `5e-5` | — | |
| `EMA_ALPHA` | `0.05` | `0.05` | — | EMA for student→teacher blend (ignored when `TEACHER_MODEL_PATH` set) |
| `TRAINER_BACKEND` | `fsdp` | `fsdp` | — | **default**; MCore FSDP + torch AdamW (the `ddp`/8-bit path is being removed — see `TODOS.md`) |

### Data & collator

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `TRAIN_DATA_PATH` | **(required — no default)** | **(required — no default)** | **yes** | `tool_defs.json` loaded from `dirname(TRAIN_DATA_PATH)` (absent → `tools=None`); `train_full.sh` fails fast if unset |
| `HINDSIGHT_FIELD` | `online_feedback` | `enriched_user_response` | **yes** | `online_feedback` → reflector grades + env builds privileged prompt dynamically; `user_response`/`enriched_user_response` → static hint. OLS data has no enriched field → use `user_response` |

### Env & API

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `ENV_TYPE` | `rag` | `rag` | — | `rag` (vLLM + optional reflector) vs `api_adapter` (multi-turn litellm adapter loop) |
| `API_MODEL` | — | `vertex_ai/claude-haiku-4-5@20251001` | api_adapter only | litellm model |
| `MAX_ADAPTER_TURNS` | — | `5` | api_adapter only | |
| `REFLECTOR_MODEL` | — | `claude-sonnet-4-6@default` | online_feedback | Anthropic Vertex |
| `REFLECTOR_REGION` | — | `us-east5` | — | |
| `REFLECTOR_PROJECT_ID` | — | `(empty)` | online_feedback | required for reflector calls |
| `GEN_TEMPERATURE` | `0.7` | `0.7` | — | |
| `GEN_TOP_P` | — | `0.95` | — | |
| `STUDENT_THINKING` | `0` | `0` | — | `1` → student + teacher render with thinking on (Qwen: `enable_thinking=True`; gpt-oss: analysis channel). Only Qwen/gpt-oss validated — `config.py` raises otherwise. Completion stays a single vLLM pass (truncated-CoT split deferred) |
| `THINKING_BUDGET` | `512` | `512` | — | api_adapter thinking split (`env/api_adapter_env.py:182`); unused by the rag path while thinking-on generation is single-pass |

### Servers / ports

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `VLLM_PORT` | `8001` | `8000` | — | base port; instance `i` = `+i*100`; override for parallel jobs |
| `VLLM_PORTS` | (built from `NUM_VLLM_GPUS`) | (derived) | — | comma-separated list; trainer round-robins rollouts |
| `VLLM_NCCL_MASTER_PORT_BASE` | `29500` | — | parallel jobs | base NCCL master port for vLLM weight-transfer engines; instance `i` = `+i*100`; override to avoid colliding with another job |
| `LOGPROB_PORT` | `8010` | `8010` | parallel jobs | HTTP health/weight-sync; override for parallel jobs |
| `LOGPROB_TCP_PORT` | `8011` | `8011` | parallel jobs | teacher-logprob data plane; override for parallel jobs |

### Ops / observability

| Var | train_full.sh default | config.py default | Critical | Notes |
|---|---|---|---|---|
| `OUTPUT_DIR` | `/workspace/output_megatron` | `./output` | **yes** | checkpoints land here |
| `PYTORCH_CUDA_ALLOC_CONF` | `(empty)` | — | **yes** | **`expandable_segments:True` is required** — without it the backward pass OOMs from allocator fragmentation |
| `WANDB_PROJECT` | `sdft-online` | `sdft-online` | — | |
| `WANDB_ENTITY` | `(empty)` | — | — | |
| `WANDB_NAME` | `sdft-ddp-{model}-t{N}-e{E}` | — | — | |
| `WANDB_MODE` | `(empty)` | — | — | `disabled` for smokes |
| `VERTEXAI_LOCATION` | `us-east5` | — | api_adapter | |
| `VERTEXAI_PROJECT` | `(empty)` | — | api_adapter | |
| `LOGGING_LEVEL` | — | `DEBUG` | — | trainer file-sink level (`logs/trainer.log`) |

### Container plumbing (set by the script, not meant to be overridden)

`VLLM_SERVER_DEV_MODE=1`, `BNB_CUDA_VERSION=130`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
`RAYON_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, `MASTER_ADDR=127.0.0.1`, `PYTHONPATH=/workspace`,
`--ipc=host`, `--network=host`, `--pids-limit=-1`, `--add-host <hostname>:127.0.0.1`.

## Hard invariants (do not violate)

- **`--max-model-len` = `MAX_TOTAL_LEN`.** `train_full.sh:142` passes `$MAX_TOTAL_LEN` through to
  vLLM instead of a hardcoded value, so the server context always matches the trainer budgets.
  Safe by construction: the `config.py` guard (`STUDENT_MAX_PROMPT_LEN + GEN_MAX_NEW_TOKENS ≤
  MAX_TOTAL_LEN`) is exactly vLLM's `prompt + max_tokens ≤ --max-model-len` requirement, so rollouts
  can't HTTP-400. OLS needs `MAX_TOTAL_LEN ≥ 15360` (use 16384); the old hardcoded 8192 was the OLS
  HTTP-400 root cause. `--max-model-len` also sizes vLLM's KV-cache reservation, so don't inflate it
  past the budgets.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** Run-21-era OOM was fragmentation, not
  capacity (5.75 GiB reserved-but-unallocated at 4.24 GiB free). Permanent default.
- **vLLM pinned to `==0.23`** — v0.25+ pulls `torchcodec` which needs system FFmpeg libs. Do not upgrade.
  Container installs: `vllm==0.23 bitsandbytes safetensors`, `humming-kernels[cu13]==0.1.4`,
  `kernels==0.14.1`, `litellm google-cloud-aiplatform tenacity fastapi uvicorn`; `triton_kernels` uninstalled.
- **`GRAD_ACCUM_STEPS % NUM_TRAINERS == 0`** (asserted in trainer.py) and effective
  `local_accum_steps = GRAD_ACCUM_STEPS / NUM_TRAINERS`.
- **`STUDENT_MAX_PROMPT_LEN + GEN_MAX_NEW_TOKENS ≤ MAX_TOTAL_LEN`** — `config.py` raises at import;
  launch fails fast before any model load.
- **Budgets must exceed the protected-set renders** (`system + tools ≤ STUDENT_MAX_PROMPT_LEN`,
  `system + tools + hint ≤ TEACHER_MAX_PROMPT_LEN`). Violating examples are **dropped at dataset load
  with warnings, not raised** (see `docs/megatron_trainer/collator.md`). Defaults 1024/2048 are legacy —
  on OLS data they would drop every example and the run aborts with "All examples dropped".
- **Ports spaced 100 apart + 30s stagger** between vLLM starts — port-scan race fix.
- **`--pids-limit=-1`** — required for multi-vLLM thread limits.
- **`--gpu-memory-utilization 0.8`** — bf16 weights (~42 GB for gpt-oss-20b) don't fit the 0.5 used
  for mxfp4.
- **Node-specific paths** (HF cache `/mnt/nvme5n1/rohan_patched_ckpts/hf-cache`, data, triton cache)
  are hardcoded — this launches on `rh-h100-01`; sync the repo and adapt mounts elsewhere.

## First-run verification checklist

Start a job, then inspect `logs/` (all host-visible via the workspace mount) in this order:

1. **`logs/vllm_$i.log`** — each instance prints "Application startup complete"; the
   `curl /v1/models` gate in `train_full.sh` confirms readiness ("vLLM on port ... ready").
2. **`logs/training.log`** — container run tee: confirms vLLM instances started, "Starting N-rank
   trainer", and streams the trainer's stdout/stderr.
3. **`logs/trainer.log`** (debug) — the trainer's own file sink:
   - "Loading dataset: <path>" → **drop-filter summary**. A healthy OLS run logs **no**
     "Dropping example idx=" lines; over-budget-hint drops appear as warnings (legacy
     enriched data at `TEACHER_MAX_PROMPT_LEN=2048` drops many — raise the budget if too many).
   - "Dataset: N examples, M steps/epoch" — confirms the filter didn't gut the set.
   - "Waiting for vLLM server" → "vLLM weight engine ready" → "Logprob weight engine ready"
     (only when `TEACHER_MODEL_PATH` empty).
   - "=== Epoch 1/N ===" then per-step "TIMING step=... total=... gen=... teacher=... student=..."
   - **No HTTP 400s** during rollout (grep `training.log`/`trainer.log` for `400` / `Client error`).
4. **`logs/logprob_server.log`** — server up; TCP teacher requests flowing per step.
5. **Checkpoint cadence** — `OUTPUT_DIR/step_{N}` at every `SAVE_EVERY`, `OUTPUT_DIR/epoch_{N}`
   at epoch end.

## Validation evidence

| Run / smoke | Layout & env | Result |
|---|---|---|
| T2 (gpt-oss-20b, FSDP + bf16 vLLM) | `train_full.sh 0 4 2`, `TRAINER_BACKEND=fsdp HINDSIGHT_FIELD=user_response NUM_EPOCHS=1 GRAD_ACCUM_STEPS=32 SAVE_EVERY=9999 WANDB_MODE=disabled` | 2 opt steps (loss 2.07 → 1.90); both vLLM `/update_weights` syncs + logprob sync all 200 OK; ckpts `epoch_1`/`step_2`; no deadlock |
| Smoke 8192-fix v1 | `MAX_TOTAL_LEN=8192 GEN_MAX_NEW_TOKENS=6144`, empty `PYTORCH_CUDA_ALLOC_CONF` | step 2 OOM at backward — 4.24 GiB free, 5.75 GiB fragmentation |
| Smoke 8192-fix v2 | only change: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | PASSED — step 2 loss=0.2257, worst-case C=6143, `epoch_1` ckpt |
| Run 21 | full set `maas_raft_v3.1` (399 → 12 steps/epoch), 10 epochs, `SAVE_EVERY=12`, expandable_segments | stopped early — teacher HTTP bottleneck (43% of step); led to TCP path |
| Run 22 | same base | crashed step 34 — cuDNN OOM; fixed by chunked head |
| Chunked smokes (`smoke_chunked{,2}`, `smoke_dual`) | FSDP, same GPU layout | step 2 worst-case C=6144 **no OOM**, trainer peak 58–66 GiB (was 74–78); loss 0.13–0.18 |
| Run 24 / epoch-ckpt runs | — | epoch checkpoints saved (411 tensors) |
| Run_5 (legacy analyze_research, gpt-oss/enriched) | legacy `HINDSIGHT_FIELD` path, default budgets | crashed at step 0: protected-set assert `ValueError` (enriched hint 2,171 > 2048) — the assert is now replaced by the load-time **drop filter** (see collator.md); relaunch skips over-budget hints with warnings |

## Constraints

- **`train_full.sh` is the only supported launch path.**
- Student thinking is launcher-controlled: `STUDENT_THINKING` (default `0` → `enable_thinking=False` / final channel, current behavior). `1` → thinking on (Qwen `enable_thinking=True`; gpt-oss analysis channel). Both student and teacher renders flip together.
- Budget semantics live in the collator spec — the launcher's job is to pass budgets through and keep
  `--max-model-len` consistent with them.

## Proposed changes to `megatron_trainer/train_full.sh`

- **Budget defaults are a foot-gun.** `STUDENT/TEACHER_MAX_PROMPT_LEN=2048` is legacy; consider
  defaulting to the OLS values (14336/15360) or deriving them from the dataset, and matching
  `MAX_TOTAL_LEN` / `GEN_MAX_NEW_TOKENS` so the `config.py` assert passes without manual override.
- **Thinking knobs passed through**: `STUDENT_THINKING` and `THINKING_BUDGET` now reach the
  container via `-e` (both defaulted in the script).
- **Make the parallel-job port block env-configurable** (currently two of the four ports are
  hardcoded, which blocks co-located runs on one node):
  - `LOGPROB_PORT`: currently `-e LOGPROB_PORT=8010` hardcoded → read from env with default 8010
    (`-e LOGPROB_PORT="${LOGPROB_PORT:-8010}"`).
  - vLLM NCCL master base: currently `DIST_PORT=$((29500 + i * 100))` hardcoded → derive from a new
    `VLLM_NCCL_MASTER_PORT_BASE` (default 29500) read from env and passed into the container.
  - `VLLM_PORT` and `LOGPROB_TCP_PORT` are already configurable — this makes all four overridable so
    the **Parallel runs on one node** example above works as documented.
- **No config-audit dump per run** (known landmine #9): effective limits are only recoverable from
  logs/env, not a file in `OUTPUT_DIR` — consider dumping the resolved config at startup.
