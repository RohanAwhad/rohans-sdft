# AGENTS.md

## What this is

Self-Distillation Fine Tuning (SDFT) research project. Work is organized as sequential tasks building toward a full training loop. See `.llm.md` for the SDFT algorithm description and `devlogs.md` for session history.

## Repo structure

- `task_1/` — NCCL weight transfer PoC (HF process → vLLM server). Complete.
- `task_2/` — Pure NCCL logprob server (no HTTP). Complete.
- `train_dir/` — Full SDFT training loop (reverse KL, on-policy). Active.

Each task has its own venv(s) and setup. There is no unified project-level venv or pyproject.toml.

## Target environment

- Runs on **node `rh-h100-01`** (H100 cluster), synced from local via git
- GPUs 0–3 available (`CUDA_VISIBLE_DEVICES` set per-process)
- Python 3.12 via `uv`
- Model: `Qwen/Qwen3-8B` (was 0.6B for testing)

## Per-task setup and run

### Task 1 (NCCL weight transfer)
```bash
bash task_1/setup.sh           # creates .vllm_venv + .hf_venv at repo root
bash task_1/start_server.sh    # GPU 0, terminal 1
.hf_venv/bin/python task_1/nccl_demo.py  # GPU 1, terminal 2
```

### Task 2 (pure NCCL logprob server)
```bash
# venv: task_2/.logprob_venv (created separately, not scripted)
GPU_TRAINER=2 GPU_SERVER=3 python task_2/test_e2e.py
# or: bash task_2/launch.sh
```

### train_dir (full SDFT loop)
```bash
bash train_dir/setup.sh           # creates train_dir/.venv
bash train_dir/start_vllm.sh      # GPU 0, terminal 1
bash train_dir/launch.sh           # GPU 1 (trainer) + GPU 2 (logprob server), terminal 2
```
Checkpoints saved to: `train_dir/output/epoch_{N}/`

## Gotchas

- **vLLM pinned to 0.23** — v0.25+ pulls `torchcodec` which requires system FFmpeg libs. Do not upgrade.
- **uv venvs lack pip** — install with `VIRTUAL_ENV=<path> uv pip install <pkg>`, not `pip install`.
- **vLLM spawns child processes** needing `ninja` on PATH — scripts must `export PATH="$REPO_ROOT/.vllm_venv/bin:$PATH"`.
- **`VLLM_SERVER_DEV_MODE=1`** is required to expose dev endpoints (`/init_weight_transfer_engine`, `/pause`, `/resume`, etc.).
- **NCCL broadcast is collective** — both ranks must call it simultaneously or it hangs.
- **task_2 imports are relative to `src/`** — run trainer/server from `task_2/` or set `PYTHONPATH=task_2/src`.
- Gemma 3 is gated on HF; that's why Qwen3-0.6B is used instead.
- **`device_map=DEVICE`** requires `accelerate` — install it in the venv.
- **`dtype=` not `torch_dtype=`** — `torch_dtype` is deprecated in newer transformers.
- **bitsandbytes required** for 8-bit Adam — needed to fit 8B model + optimizer on single 80GB GPU.
- **Two independent NCCL groups** coexist: `torch.distributed` (port 29500) for teacher, vLLM `NCCLWeightTransferEngine` (auto port) for inference.

## Training data

Dataset: `/home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/train_maas_sdft.jsonl` (on node).
Reference code for collator/privileged-info pattern: `~/rawhad/self_distillation/aligning_lm_from_user_interaction` (sdpo scripts).
