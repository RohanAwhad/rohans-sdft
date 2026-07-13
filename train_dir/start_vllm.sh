#!/bin/bash
# Launch vLLM inference server on GPU 0 (run in terminal 1).
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV="$SCRIPT_DIR/.venv"

source "$VENV/bin/activate"
export PATH="$VENV/bin:$PATH"   # ninja must be on PATH for vLLM subprocesses
export VLLM_SERVER_DEV_MODE=1   # enables /pause, /resume, weight transfer endpoints

CUDA_VISIBLE_DEVICES="${GPU_VLLM:-0}" python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME:-Qwen/Qwen3-0.6B}" \
    --port "${VLLM_PORT:-8000}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 8192 \
    --weight-transfer-config '{"backend":"nccl"}' \
    --no-enable-log-requests
