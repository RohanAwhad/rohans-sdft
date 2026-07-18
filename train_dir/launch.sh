#!/bin/bash
# Launch logprob server (background) + trainer (foreground).
# Run in terminal 2, AFTER start_vllm.sh is up.
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV="$SCRIPT_DIR/.venv"

source "$VENV/bin/activate"
export PATH="$VENV/bin:$PATH"

# Defaults (override via env vars)
export GPU_VLLM="${GPU_VLLM:-0}"
export GPU_TRAINER="${GPU_TRAINER:-1}"
export GPU_LOGPROB_SERVER="${GPU_LOGPROB_SERVER:-2}"
export NCCL_MASTER_PORT="${NCCL_MASTER_PORT:-29500}"
export LOGGING_LEVEL="${LOGGING_LEVEL:-DEBUG}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start logprob server on GPU 2
echo "[launch] Starting logprob server on GPU $GPU_LOGPROB_SERVER ..."
CUDA_VISIBLE_DEVICES="$GPU_LOGPROB_SERVER" python -m src.logprob_server &
SERVER_PID=$!
echo "[launch] Logprob server PID=$SERVER_PID"

# Cleanup on exit
cleanup() {
    echo "[launch] Shutting down logprob server (PID=$SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Start trainer on GPU 1 (foreground — blocks until training completes)
echo "[launch] Starting trainer on GPU $GPU_TRAINER ..."
CUDA_VISIBLE_DEVICES="$GPU_TRAINER" python -m src.trainer
