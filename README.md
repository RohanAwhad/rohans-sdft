# Self-Distillation Fine-Tuning (SDFT)

On-policy self-distillation with privileged information. A student model learns from a teacher that sees enriched context (hindsight), while the student sees only the original prompt. Weight sync via NCCL on H100s.

## Architecture

```
GPU 0: vLLM server       ← generates rollouts (HTTP)
GPU 1: Trainer (student)  ← forward/backward, orchestrates everything
GPU 2: Logprob server     ← teacher log-probs via NCCL (teacher = EMA of student + privileged info)
```

### Training loop (per step)
1. vLLM generates completion from student prompt
2. Student forward pass on `[prompt + completion]` (with grad)
3. Teacher scores `[privileged_prompt + completion]` via NCCL (no grad)
4. Reverse KL loss: `KL(p_student || p_teacher)`
5. Backward + grad accumulation (effective batch = 32)
6. Optimizer step + weight sync (EMA to teacher, full to vLLM)

### Weight sync
- **Teacher**: EMA blend `phi = alpha * theta + (1-alpha) * phi` (~9ms for 8B)
- **vLLM**: full weight transfer via `NCCLWeightTransferEngine` (~120ms for 8B)
- Sync happens every optimizer step (~140ms total, negligible vs ~2min/step)

## Setup & Run

```bash
# Setup (creates train_dir/.venv)
bash train_dir/setup.sh

# Terminal 1: vLLM server
MODEL_NAME=Qwen/Qwen3-8B bash train_dir/start_vllm.sh

# Terminal 2: trainer + logprob server
MODEL_NAME=Qwen/Qwen3-8B bash train_dir/launch.sh
```

### Parallel run on GPUs 3/4/5
```bash
GPU_VLLM=3 VLLM_PORT=8001 MODEL_NAME=Qwen/Qwen3-8B bash train_dir/start_vllm.sh
GPU_VLLM=3 GPU_TRAINER=4 GPU_LOGPROB_SERVER=5 VLLM_PORT=8001 NCCL_MASTER_PORT=29501 bash train_dir/launch.sh
```

## Config

All overridable via environment variables. See `train_dir/src/config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` | HF model name |
| `LEARNING_RATE` | `5e-5` | Constant LR |
| `GRAD_ACCUM_STEPS` | `32` | Effective batch size |
| `NUM_EPOCHS` | `10` | Training epochs |
| `EMA_ALPHA` | `0.05` | Teacher EMA blend factor |
| `GEN_MAX_NEW_TOKENS` | `2048` | Max completion length |
| `OUTPUT_DIR` | `./output` | Checkpoint directory (rolling) |
| `OFFLINE_OVERFIT` | `0` | Cache epoch 1 data, replay for remaining epochs |

## Megatron Bridge Implementation

Re-implementation of the SDFT training loop using NVIDIA Megatron Bridge / Megatron-Core instead of HuggingFace transformers. All code in `megatron_trainer/`.

### Results (Qwen3-8B, 10 epochs)

| Metric | Megatron Bridge | Reference (HF, run 14) |
|--------|----------------|----------------------|
| no_context | **34%** | 41% |
| with_context | **82%** | 83% |
| avg_loss (epoch 10) | 0.234 | comparable |

Loss trajectory:

| Epoch | avg_loss |
|-------|----------|
| 1 | 0.665 |
| 2 | 0.416 |
| 3 | 0.325 |
| 4 | 0.276 |
| 5 | 0.252 |
| 6 | 0.237 |
| 7 | 0.231 |
| 8 | 0.221 |
| 9 | 0.225 |
| 10 | 0.234 |

### Architecture

All 3 processes run inside a single NeMo container (`nvcr.io/nvidia/nemo:26.06`) with 3 GPUs:

```
Container (nvcr.io/nvidia/nemo:26.06)
├── GPU 0: vLLM server (on-policy rollouts, --enforce-eager)
├── GPU 1: Trainer (Megatron-Core GPTModel via AutoBridge)
└── GPU 2: Logprob server (Megatron-Core GPTModel, EMA-updated)
```

### Quick start

```bash
# Full 10-epoch training on GPUs 5,6,7:
bash megatron_trainer/train_full.sh 5

# Smoke test (1 epoch, grad_accum=2):
bash megatron_trainer/smoke_all_in_container.sh
```

### Key differences from HF implementation

| Component | HF (`train_dir/`) | Megatron (`megatron_trainer/`) |
|-----------|-------------------|-------------------------------|
| Model loading | `AutoModelForCausalLM` | `AutoBridge.from_hf_pretrained()` |
| Forward pass | backbone + selective lm_head | `model(input_ids, position_ids, attention_mask=None)` |
| Optimizer | `bitsandbytes.AdamW8bit` | same (`BNB_CUDA_VERSION=130`) |
| Weight sync to vLLM | direct NCCL (HF names) | Megatron→HF conversion via `export_hf_weights()` |
| Checkpointing | `model.save_pretrained()` | manual safetensors export (avoids distributed barrier) |
| Runtime | host Python venv | NeMo container (glibc 2.39 requirement) |

### Container gotchas

- `pip install --no-deps vllm==0.23 bitsandbytes` at startup (container has vLLM 0.20 which is incompatible)
- `BNB_CUDA_VERSION=130` (container CUDA 13.2, highest bnb binary is 13.0)
- `--enforce-eager` for vLLM (torch.compile incompatible with container's torch 2.12)
- `start_vllm_patched.py` monkey-patches prometheus `_IncludedRouter` crash
- `--add-host $(hostname):127.0.0.1` for NCCL hostname resolution
- `gradient_accumulation_fusion=False` in model config (no Megatron distributed optimizer)
- tokenizer_config.json fix: `extra_special_tokens` list→dict conversion

## Repo structure

```
task_1/            PoC: NCCL weight transfer (HF -> vLLM)         [complete]
task_2/            PoC: pure NCCL logprob server                   [complete]
train_dir/         Full SDFT training loop (HF transformers)       [complete]
  src/
    config.py          All hyperparams (env-overridable)
    collator.py        SDFTCollator (prompt + privileged conditional)
    trainer.py         Training loop, KL loss, forward_student
    logprob_server.py  Teacher process (GPU 2)
    nccl_comm.py       NCCL protocol (log-probs, EMA sync)
    vllm_utils.py      HTTP generation + NCCL weight sync
  setup.sh             Venv creation
  start_vllm.sh        vLLM server launcher
  launch.sh            Logprob server + trainer launcher
megatron_trainer/  SDFT with Megatron Bridge (NeMo container)      [complete]
  config.py            Config (same params, env-overridable)
  collator.py          SDFTCollator (unchanged)
  model_utils.py       AutoBridge init, HF weight export, checkpoint save
  nccl_comm.py         NCCL protocol (adapted for MCore GPTModel)
  vllm_utils.py        vLLM client + Megatron→HF weight conversion
  trainer.py           Training loop (Megatron-Core model)
  logprob_server.py    Teacher process (Megatron-Core model)
  start_vllm_patched.py  vLLM launcher with prometheus fix
  train_full.sh        Production launcher (10 epochs)
  smoke_all_in_container.sh  Smoke test launcher
  goal.md              Design doc + progress tracking
```

## Requirements

- H100 GPU(s) with NVLink
- Python 3.12, `uv`
- vLLM 0.23 (pinned, 0.25+ breaks)
- bitsandbytes (8-bit Adam for 8B model)
- accelerate (for `device_map`)

## WandB

Runs logged to `ronny21/sdpo-amortize`. See `devlogs.md` for experiment history and findings.
