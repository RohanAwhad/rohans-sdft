#!/bin/bash
# SDFT Training with Megatron Bridge
#
# Architecture:
#   GPU 0: vLLM server (host Python, HF model)
#   GPU 1: Trainer (NeMo container, Megatron model)
#   GPU 2: Logprob server (NeMo container, Megatron model)
#
# The trainer + logprob server run inside the NeMo container.
# vLLM runs on the host with its own Python environment.
#
# Usage:
#   bash megatron_trainer/run.sh [--gpus 0,1,2] [--model Qwen/Qwen3-8B]

set -euo pipefail

# Defaults
GPU_VLLM=${GPU_VLLM:-0}
GPU_TRAINER=${GPU_TRAINER:-1}
GPU_LOGPROB_SERVER=${GPU_LOGPROB_SERVER:-2}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
VLLM_PORT=${VLLM_PORT:-8000}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

# Container entry point: install missing deps then run the given command
CONTAINER_ENTRYPOINT="pip install --quiet --no-deps vllm==0.23 bitsandbytes 2>/dev/null; exec"

# Common container flags
CONTAINER_COMMON=(
    --ipc=host
    --network=host
    --add-host "$HOSTNAME_FIX:127.0.0.1"
    -e CUDA_VISIBLE_DEVICES=0
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
    -e RAYON_NUM_THREADS=1
    -e TOKENIZERS_PARALLELISM=false
    -e MASTER_ADDR=127.0.0.1
    -e PYTHONPATH=/workspace
    -v "$WORKSPACE:/workspace:z"
    -v "$HF_CACHE:/root/.cache/huggingface:z"
    -v /home/lab/rawhad:/home/lab/rawhad:ro
    -w /workspace
)

echo "=== SDFT Megatron Training ==="
echo "GPUs: vLLM=$GPU_VLLM, Trainer=$GPU_TRAINER, Logprob=$GPU_LOGPROB_SERVER"
echo "Model: $MODEL_NAME"
echo "Workspace: $WORKSPACE"

# ---- Step 1: Start vLLM server on host ----
echo "[1/3] Starting vLLM server on GPU $GPU_VLLM..."
CUDA_VISIBLE_DEVICES=$GPU_VLLM \
VLLM_SERVER_DEV_MODE=1 \
VLLM_USE_V1=0 \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --weight-transfer-config '{"backend":"nccl"}' \
    --no-enable-log-requests \
    2>&1 | tee logs/vllm.log &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ---- Step 2: Start logprob server in container ----
echo "[2/3] Starting logprob server on GPU $GPU_LOGPROB_SERVER..."
podman run --rm -d \
    --name sdft-logprob \
    --device "nvidia.com/gpu=$GPU_LOGPROB_SERVER" \
    "${CONTAINER_COMMON[@]}" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_MODEL_PATH="$MODEL_NAME" \
    -e NCCL_MASTER_PORT=29500 \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c "$CONTAINER_ENTRYPOINT python -m megatron_trainer.logprob_server"

echo "Logprob server container started."

# ---- Step 3: Start trainer in container ----
echo "[3/3] Starting trainer on GPU $GPU_TRAINER..."
podman run --rm -it \
    --name sdft-trainer \
    --device "nvidia.com/gpu=$GPU_TRAINER" \
    "${CONTAINER_COMMON[@]}" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_MODEL_PATH="$MODEL_NAME" \
    -e NCCL_MASTER_PORT=29500 \
    -e VLLM_PORT="$VLLM_PORT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c "$CONTAINER_ENTRYPOINT python -m megatron_trainer.trainer"

# ---- Cleanup ----
echo "Training complete. Cleaning up..."
podman stop sdft-logprob 2>/dev/null || true
kill $VLLM_PID 2>/dev/null || true
echo "Done."
