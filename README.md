# Self-Distillation Fine-Tuning (SDFT)

On-policy self-distillation with privileged information. A student model learns from a teacher that sees enriched context (hindsight), while the student sees only the original prompt. Weight sync via NCCL on H100s.

## How to Run:

```bash
# Terminal 1: vLLM server (GPU 1, port 8004)
cd /home/lab/rawhad/self_distillation/rohans_sdft && \
GPU_VLLM=4 MODEL_NAME=/home/lab/rawhad/self_distillation/rohans_sdft/train_dir/output_run_14/epoch_10 VLLM_PORT=8010 \
bash train_dir/start_vllm.sh
```

```bash
# Terminal 2: Trainer (GPU 2) + Logprob server (GPU 3)
cd /home/lab/rawhad/self_distillation/rohans_sdft/train_dir && \
GPU_VLLM=4 \
GPU_TRAINER=5 \
GPU_LOGPROB_SERVER=6 \
MODEL_NAME=/home/lab/rawhad/self_distillation/rohans_sdft/train_dir/output_run_14/epoch_10 \
VLLM_PORT=8010 \
NCCL_MASTER_PORT=29501 \
TRAIN_DATA_PATH=/home/lab/rawhad/sdft_rag_experiment/data/raft_dataset/v2_500/dataset_train_sdft.jsonl \
HINDSIGHT_FIELD=enriched_user_response \
OUTPUT_DIR=/mnt/nvme7n1/sdft_rag_experiment_raft_post_ki \
GRAD_ACCUM_STEPS=32 \
NUM_EPOCHS=10 \
SAVE_EVERY=3 \
LEARNING_RATE=2e-6 \
WANDB_NAME=sdft_rag_from_base_raft_post_ki \
bash launch.sh
```

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

## Repo structure

```
task_1/     PoC: NCCL weight transfer (HF -> vLLM)        [complete]
task_2/     PoC: pure NCCL logprob server                  [complete]
train_dir/  Full SDFT training loop                        [active]
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
```

## Requirements

- H100 GPU(s) with NVLink
- Python 3.12, `uv`
- vLLM 0.23 (pinned, 0.25+ breaks)
- bitsandbytes (8-bit Adam for 8B model)
- accelerate (for `device_map`)

## WandB

Runs logged to `ronny21/sdpo-amortize`. See `devlogs.md` for experiment history and findings.
