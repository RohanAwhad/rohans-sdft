#!/bin/bash
# Quick test: 4 optimizer steps + checkpoint export.
# Validates save_hf_checkpoint with the new manual safetensors export.
set -euo pipefail

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

GPU_VLLM=3; GPU_TRAINER=4; GPU_LOGPROB=5
VLLM_PORT=8001; MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_DIR="/workspace/output_ckpt_test"

mkdir -p "$WORKSPACE/logs"
podman stop sdft-ckpt-test 2>/dev/null || true
podman rm sdft-ckpt-test 2>/dev/null || true

echo "=== Checkpoint Export Test ==="
TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-ckpt-test \
    --device "nvidia.com/gpu=$GPU_VLLM" \
    --device "nvidia.com/gpu=$GPU_TRAINER" \
    --device "nvidia.com/gpu=$GPU_LOGPROB" \
    --ipc=host --network=host \
    --add-host "$HOSTNAME_FIX:127.0.0.1" \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e RAYON_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false \
    -e MASTER_ADDR=127.0.0.1 -e PYTHONPATH=/workspace \
    -e MODEL_NAME="$MODEL_NAME" -e HF_MODEL_PATH="$MODEL_NAME" \
    -e NCCL_MASTER_PORT=29501 -e VLLM_PORT="$VLLM_PORT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e NUM_EPOCHS=1 -e GRAD_ACCUM_STEPS=2 \
    -e WANDB_MODE=disabled -e SAVE_EVERY=2 \
    -e VLLM_SERVER_DEV_MODE=1 -e VLLM_USE_V1=0 \
    -e BNB_CUDA_VERSION=130 \
    -v "$WORKSPACE:/workspace:z" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c '
set -e
pip install --quiet --no-deps vllm==0.23 bitsandbytes safetensors 2>/dev/null

echo "=== Starting vLLM ==="
CUDA_VISIBLE_DEVICES=0 python /workspace/megatron_trainer/start_vllm_patched.py \
    --model "$MODEL_NAME" --port "$VLLM_PORT" --max-model-len 4096 \
    --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --weight-transfer-config "{\"backend\":\"nccl\"}" \
    --enforce-eager --no-enable-log-requests \
    &>/workspace/logs/vllm_ckpt.log &

echo "=== Starting logprob ==="
CUDA_VISIBLE_DEVICES=2 python -m megatron_trainer.logprob_server \
    &>/workspace/logs/logprob_ckpt.log &

echo "=== Waiting for vLLM ==="
for i in $(seq 1 120); do
    curl -s http://localhost:$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
echo "vLLM ready"

echo "=== Starting trainer (4 steps only) ==="
CUDA_VISIBLE_DEVICES=1 python -m megatron_trainer.trainer 2>&1

echo "=== Done ==="
ls -la "$OUTPUT_DIR"/
' 2>&1 | tee "$WORKSPACE/logs/ckpt_test.log" | grep -E "opt_step|Epoch|checkpoint|saved|HF|error|Error|Traceback|Done|==="
