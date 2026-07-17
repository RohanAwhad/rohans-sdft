#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV="$SCRIPT_DIR/.venv"

echo "Creating venv at $VENV ..."
uv venv "$VENV" --python 3.12

echo "Installing dependencies ..."
VIRTUAL_ENV="$VENV" uv pip install \
    "vllm==0.23" \
    "anthropic[vertex]" \
    datasets \
    loguru \
    wandb \
    flash-attn

echo ""
echo "Setup complete."
echo "  vLLM server:  bash train_dir/start_vllm.sh"
echo "  Training:     bash train_dir/launch.sh"
