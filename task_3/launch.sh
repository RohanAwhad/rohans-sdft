#!/bin/bash
# Launch adapter SDFT training.
# Run in terminal 2, AFTER start_vllm.sh is up.
#
# Override via env vars:
#   GPU_VLLM, GPU_TRAINER, GPU_LOGPROB_SERVER
#   VLLM_PORT, NCCL_MASTER_PORT
#   NUM_EPOCHS, LEARNING_RATE, GRAD_ACCUM_STEPS
#   TRAIN_DATA_PATH, OUTPUT_DIR
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV="$REPO_ROOT/train_dir/.venv"

source "$VENV/bin/activate"
export PATH="$VENV/bin:$PATH"
PYTHONPATH="$REPO_ROOT/train_dir/src:$REPO_ROOT:$PYTHONPATH"

# Defaults
export GPU_VLLM="${GPU_VLLM:-0}"
export GPU_TRAINER="${GPU_TRAINER:-1}"
export GPU_LOGPROB_SERVER="${GPU_LOGPROB_SERVER:-2}"
export NCCL_MASTER_PORT="${NCCL_MASTER_PORT:-29500}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export LOGGING_LEVEL="${LOGGING_LEVEL:-DEBUG}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Adapter config
export ADAPTER_MODE=1
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$REPO_ROOT/data/train.jsonl}"
export REASONING_BUDGET="${REASONING_BUDGET:-128}"
export GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-256}"
export HINDSIGHT_FIELD="user_response"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
export EMA_ALPHA="${EMA_ALPHA:-0.05}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/train_dir/output}"
export SAVE_EVERY="${SAVE_EVERY:-200}"

echo "[launch] Adapter SDFT training"
echo "  Model: $MODEL_NAME"
echo "  Data:  $TRAIN_DATA_PATH"
echo "  GPUs:  vLLM=$GPU_VLLM trainer=$GPU_TRAINER logprob=$GPU_LOGPROB_SERVER"
echo "  Output: $OUTPUT_DIR"

# Start logprob server on its GPU
echo "[launch] Starting logprob server on GPU $GPU_LOGPROB_SERVER ..."
CUDA_VISIBLE_DEVICES="$GPU_LOGPROB_SERVER" python -m src.logprob_server &
SERVER_PID=$!
echo "[launch] Logprob server PID=$SERVER_PID"

cleanup() {
    echo "[launch] Shutting down logprob server (PID=$SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Start trainer on its GPU (foreground)
echo "[launch] Starting trainer on GPU $GPU_TRAINER ..."
cd "$REPO_ROOT/train_dir"
CUDA_VISIBLE_DEVICES="$GPU_TRAINER" python -m src.trainer
