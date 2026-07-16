#!/bin/bash
# Quick end-to-end smoke test: 1 optimizer step of SDFT with Megatron Bridge.
#
# Uses GPUs 3,4,5 to avoid interference.
# Runs 32 samples (= 1 optimizer step with grad_accum=32).

set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

GPU_VLLM=3
GPU_TRAINER=4
GPU_LOGPROB=5
VLLM_PORT=8001  # different from default to avoid conflicts
MODEL_NAME="Qwen/Qwen3-8B"
VLLM_VENV="/mnt/nvme0n1/rawhad/self_distillation/rohans_sdft/train_dir/.venv"
OUTPUT_DIR="/mnt/nvme0n1/rawhad/self_distillation/rohans-sdft-2/output_smoke"

mkdir -p "$WORKSPACE/logs" "$OUTPUT_DIR"

echo "=== SDFT Megatron Smoke Test ==="
echo "GPUs: vLLM=$GPU_VLLM, Trainer=$GPU_TRAINER, Logprob=$GPU_LOGPROB"
echo "Model: $MODEL_NAME"

cleanup() {
    echo "Cleaning up..."
    podman stop sdft-smoke-logprob 2>/dev/null || true
    podman rm sdft-smoke-logprob 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# ---- Step 1: Start vLLM server on host ----
echo "[1/3] Starting vLLM on GPU $GPU_VLLM (port $VLLM_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_VLLM \
VLLM_SERVER_DEV_MODE=1 \
VLLM_USE_V1=0 \
PATH="$VLLM_VENV/bin:$PATH" \
"$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --weight-transfer-config '{"backend":"nccl"}' \
    --no-enable-log-requests \
    2>&1 | tee "$WORKSPACE/logs/vllm_smoke.log" &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ---- Step 2: Start logprob server in container ----
# Common container flags
CONTAINER_FLAGS=(
    --ipc=host
    --network=host
    --add-host "$HOSTNAME_FIX:127.0.0.1"
    -e CUDA_VISIBLE_DEVICES=0
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
    -e RAYON_NUM_THREADS=1
    -e TOKENIZERS_PARALLELISM=false
    -e MASTER_ADDR=127.0.0.1
    -e PYTHONPATH=/workspace
    -e MODEL_NAME="$MODEL_NAME"
    -e HF_MODEL_PATH="$MODEL_NAME"
    -e NCCL_MASTER_PORT=29501
    -v "$WORKSPACE:/workspace:z"
    -v "$HF_CACHE:/root/.cache/huggingface:z"
    -v /home/lab/rawhad:/home/lab/rawhad:ro
    -w /workspace
)

echo "[2/3] Starting logprob server on GPU $GPU_LOGPROB..."
TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm -d \
    --name sdft-smoke-logprob \
    --device "nvidia.com/gpu=$GPU_LOGPROB" \
    "${CONTAINER_FLAGS[@]}" \
    nvcr.io/nvidia/nemo:26.06 \
    python -m megatron_trainer.logprob_server

echo "Logprob container started. Waiting 5s for init..."
sleep 5

# ---- Step 3: Start trainer in container ----
echo "[3/3] Starting trainer on GPU $GPU_TRAINER..."
TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-smoke-trainer \
    --device "nvidia.com/gpu=$GPU_TRAINER" \
    "${CONTAINER_FLAGS[@]}" \
    -e VLLM_PORT="$VLLM_PORT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e NUM_EPOCHS=1 \
    -e GRAD_ACCUM_STEPS=2 \
    -e WANDB_MODE=disabled \
    -e SAVE_EVERY=9999 \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c "pip install --quiet --no-deps vllm==0.23 bitsandbytes 2>/dev/null; exec python -m megatron_trainer.trainer" \
    2>&1 | tee "$WORKSPACE/logs/trainer_smoke.log"

echo "=== Smoke test complete ==="
