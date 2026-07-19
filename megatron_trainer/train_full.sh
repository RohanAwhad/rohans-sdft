#!/bin/bash
# Full SDFT training with Megatron Bridge (all-in-container).
#
# Runs 10 epochs of SDFT training on Qwen3-8B using 3 H100 GPUs
# inside a single NeMo container. Estimated runtime: ~8 hours.
#
# Usage:
#   bash megatron_trainer/train_full.sh [GPU_START=3]
#
# The script uses GPU_START, GPU_START+1, GPU_START+2 for
# vLLM, trainer, and logprob server respectively.

set -euo pipefail

GPU_START=${1:-3}
GPU_VLLM=$GPU_START
GPU_TRAINER=$((GPU_START + 1))
GPU_LOGPROB=$((GPU_START + 2))

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
VLLM_PORT=${VLLM_PORT:-8001}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/output_megatron"}

echo "=== SDFT Megatron Full Training ==="
echo "GPUs: vLLM=$GPU_VLLM, Trainer=$GPU_TRAINER, Logprob=$GPU_LOGPROB"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"

mkdir -p "$WORKSPACE/logs"

TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-megatron-train \
    --device "nvidia.com/gpu=$GPU_VLLM" \
    --device "nvidia.com/gpu=$GPU_TRAINER" \
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
    -e WANDB_NAME="${WANDB_NAME:-sdft-megatron-$(basename $MODEL_NAME)-e${NUM_EPOCHS:-10}}" \
    -e WANDB_ENTITY="${WANDB_ENTITY:-}" \
    -e VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}" \
    -e HINDSIGHT_FIELD="${HINDSIGHT_FIELD:-online_feedback}" \
    -e TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/workspace/train_dir/data/synthetic_algebra/train_sdft.jsonl}" \
    -v "$WORKSPACE:/workspace:z" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c '
set -e
pip install --quiet --no-deps vllm==0.23 bitsandbytes safetensors 2>/dev/null
pip install --quiet litellm google-cloud-aiplatform tenacity fastapi uvicorn 2>/dev/null

echo "=== Starting vLLM on GPU 0 (internal) ==="
CUDA_VISIBLE_DEVICES=0 python /workspace/megatron_trainer/start_vllm_patched.py \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    --enforce-eager \
    --no-enable-log-requests \
    &>/workspace/logs/vllm.log &
VLLM_PID=$!

echo "=== Starting logprob server on GPU 2 (internal) ==="
CUDA_VISIBLE_DEVICES=2 python -m megatron_trainer.logprob_server \
    &>/workspace/logs/logprob_server.log &
LOGPROB_PID=$!

echo "=== Waiting for vLLM... ==="
for i in $(seq 1 120); do
    curl -s http://localhost:$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
echo "vLLM ready"

echo "=== Starting trainer on GPU 1 (internal) ==="
CUDA_VISIBLE_DEVICES=1 python -m megatron_trainer.trainer 2>&1

echo "=== Training complete ==="
kill $VLLM_PID $LOGPROB_PID 2>/dev/null || true
' 2>&1 | tee "$WORKSPACE/logs/training.log"

echo "=== Done ==="
