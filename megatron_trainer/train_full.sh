#!/bin/bash
# Full SDFT training with DDP (all-in-container).
#
# Runs SDFT training on Qwen3-8B using N+2 H100 GPUs (1 vLLM + N trainers + 1 logprob)
# inside a single NeMo container.
#
# Usage:
#   bash megatron_trainer/train_full.sh [GPU_START=3] [NUM_TRAINERS=2]
#
# Layout: vLLM=GPU_START, trainers=GPU_START+1..N, logprob=GPU_START+N+1
#
# Example (GPUs 3-6, 2 trainers):
#   bash megatron_trainer/train_full.sh 3 2
#   → vLLM=3, trainers=4,5, logprob=6

set -euo pipefail

GPU_START=${1:-3}
NUM_TRAINERS=${2:-2}
GPU_VLLM=$GPU_START
GPU_LOGPROB=$((GPU_START + NUM_TRAINERS + 1))

# Build trainer GPU device flags
TRAINER_DEVICES=""
TRAINER_GPUS=""
for i in $(seq 1 $NUM_TRAINERS); do
    gpu=$((GPU_START + i))
    TRAINER_GPUS="${TRAINER_GPUS:+$TRAINER_GPUS,}$gpu"
    TRAINER_DEVICES="$TRAINER_DEVICES --device nvidia.com/gpu=$gpu"
done

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
VLLM_PORT=${VLLM_PORT:-8001}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/output_megatron"}

echo "=== SDFT DDP Full Training ==="
echo "GPUs: vLLM=$GPU_VLLM, Trainers=$TRAINER_GPUS, Logprob=$GPU_LOGPROB"
echo "NUM_TRAINERS=$NUM_TRAINERS, Model=$MODEL_NAME"

mkdir -p "$WORKSPACE/logs"

TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-megatron-train \
    --device "nvidia.com/gpu=$GPU_VLLM" \
    $TRAINER_DEVICES \
    --device "nvidia.com/gpu=$GPU_LOGPROB" \
    --ipc=host \
    --network=host \
    --add-host "$HOSTNAME_FIX:127.0.0.1" \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e RAYON_NUM_THREADS=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e MASTER_ADDR=127.0.0.1 \
    -e PYTHONPATH=/workspace \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_MODEL_PATH="$MODEL_NAME" \
    -e LOGPROB_PORT=8010 \
    -e VLLM_PORT="$VLLM_PORT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e NUM_EPOCHS="${NUM_EPOCHS:-10}" \
    -e GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}" \
    -e SAVE_EVERY="${SAVE_EVERY:-200}" \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e BNB_CUDA_VERSION=130 \
    -e WANDB_PROJECT="${WANDB_PROJECT:-sdft-online}" \
    -e WANDB_NAME="${WANDB_NAME:-sdft-ddp-$(basename $MODEL_NAME)-t${NUM_TRAINERS}-e${NUM_EPOCHS:-10}}" \
    -e WANDB_ENTITY="${WANDB_ENTITY:-}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}" \
    -e VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-}" \
    -e HINDSIGHT_FIELD="${HINDSIGHT_FIELD:-online_feedback}" \
    -e TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/workspace/train_dir/data/synthetic_algebra/train_sdft.jsonl}" \
    -e GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}" \
    -e GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-2048}" \
    -e EMA_ALPHA="${EMA_ALPHA:-0.05}" \
    -e LEARNING_RATE="${LEARNING_RATE:-5e-5}" \
    -e STUDENT_MAX_PROMPT_LEN="${STUDENT_MAX_PROMPT_LEN:-2048}" \
    -e TEACHER_MAX_PROMPT_LEN="${TEACHER_MAX_PROMPT_LEN:-2048}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
    -e ENV_TYPE="${ENV_TYPE:-rag}" \
    -e LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-4}" \
    -v "$WORKSPACE:/workspace:z" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c "
set -e
pip install --quiet --no-deps vllm==0.23 bitsandbytes safetensors 2>/dev/null
pip install --quiet litellm google-cloud-aiplatform tenacity fastapi uvicorn 2>/dev/null

NUM_GPUS=\$(nvidia-smi -L | wc -l)
LOGPROB_GPU=\$((NUM_GPUS - 1))
TRAINER_LAST=\$((NUM_GPUS - 2))

echo \"=== Starting vLLM on internal GPU 0 ===\"
CUDA_VISIBLE_DEVICES=0 python /workspace/megatron_trainer/start_vllm_patched.py \\
    --model \"\$MODEL_NAME\" \\
    --port \"\$VLLM_PORT\" \\
    --max-model-len 8192 \\
    --dtype bfloat16 \\
    --gpu-memory-utilization 0.5 \\
    --weight-transfer-config '{\"backend\":\"nccl\"}' \\
    --enforce-eager \\
    --no-enable-log-requests \\
    &>/workspace/logs/vllm.log &
VLLM_PID=\$!

echo \"=== Starting logprob server on internal GPU \$LOGPROB_GPU ===\"
CUDA_VISIBLE_DEVICES=\$LOGPROB_GPU python -m megatron_trainer.logprob_server \\
    &>/workspace/logs/logprob_server.log &
LOGPROB_PID=\$!

echo \"=== Waiting for vLLM... ===\"
for i in \$(seq 1 120); do
    curl -s http://localhost:\$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
echo \"vLLM ready\"

echo \"=== Starting $NUM_TRAINERS-rank trainer via torchrun on internal GPUs 1..\$TRAINER_LAST ===\"
TRAINER_CUDA=\$(seq -s, 1 \$TRAINER_LAST)
CUDA_VISIBLE_DEVICES=\$TRAINER_CUDA torchrun --nproc_per_node=$NUM_TRAINERS \\
    -m megatron_trainer.trainer 2>&1

echo \"=== Training complete ===\"
kill \$VLLM_PID \$LOGPROB_PID 2>/dev/null || true
" 2>&1 | tee "$WORKSPACE/logs/training.log"

echo "=== Done ==="
