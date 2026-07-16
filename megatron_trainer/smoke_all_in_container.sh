#!/bin/bash
# All-in-container smoke test: vLLM + trainer + logprob inside one NeMo container.
# Avoids cross-environment NCCL issues.

set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

GPU_VLLM=3
GPU_TRAINER=4
GPU_LOGPROB=5
VLLM_PORT=8001
MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_DIR="/workspace/output_smoke"

mkdir -p "$WORKSPACE/logs"

echo "=== SDFT All-in-Container Smoke Test ==="
echo "GPUs: vLLM=$GPU_VLLM, Trainer=$GPU_TRAINER, Logprob=$GPU_LOGPROB"

cleanup() {
    podman stop sdft-all-in-one 2>/dev/null || true
    podman rm sdft-all-in-one 2>/dev/null || true
    echo "Cleaned up."
}
trap cleanup EXIT

# Run all 3 processes inside one container
TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-all-in-one \
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
    -e NCCL_MASTER_PORT=29501 \
    -e VLLM_PORT="$VLLM_PORT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e NUM_EPOCHS=1 \
    -e GRAD_ACCUM_STEPS=2 \
    -e WANDB_MODE=disabled \
    -e SAVE_EVERY=9999 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_USE_V1=0 \
    -e BNB_CUDA_VERSION=130 \
    -v "$WORKSPACE:/workspace:z" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c '
set -e
pip install --quiet --no-deps vllm==0.23 bitsandbytes 2>/dev/null

echo "=== Starting vLLM on GPU 0 (internal) ==="
CUDA_VISIBLE_DEVICES=0 python /workspace/megatron_trainer/start_vllm_patched.py \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    --enforce-eager \
    --no-enable-log-requests \
    &>/workspace/logs/vllm_container.log &
VLLM_PID=$!

echo "=== Starting logprob server on GPU 2 (internal) ==="
CUDA_VISIBLE_DEVICES=2 python -m megatron_trainer.logprob_server \
    &>/workspace/logs/logprob_container.log &
LOGPROB_PID=$!

echo "=== Waiting for vLLM... ==="
for i in $(seq 1 120); do
    curl -s http://localhost:$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
echo "vLLM ready"

echo "=== Starting trainer on GPU 1 (internal) ==="
CUDA_VISIBLE_DEVICES=1 python -m megatron_trainer.trainer 2>&1

echo "=== Training done, shutting down ==="
kill $VLLM_PID $LOGPROB_PID 2>/dev/null || true
' 2>&1 | tee "$WORKSPACE/logs/all_in_one_smoke.log"

echo "=== Smoke test complete ==="
