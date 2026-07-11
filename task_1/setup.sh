#!/usr/bin/env bash
# Setup script for task_1: NCCL weight transfer between HF and vLLM
# Creates two venvs and installs dependencies.
# Run from the repo root: bash task_1/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Setting up venvs in $REPO_ROOT ==="

# --- .vllm_venv: runs the vLLM inference server ---
echo ""
echo "--- Creating .vllm_venv ---"
cd "$REPO_ROOT"
uv venv .vllm_venv --python 3.12
VIRTUAL_ENV="$REPO_ROOT/.vllm_venv" uv pip install "vllm==0.23" openai
echo ".vllm_venv ready."

# --- .hf_venv: runs the training script (needs vllm for NCCLWeightTransferEngine) ---
echo ""
echo "--- Creating .hf_venv ---"
uv venv .hf_venv --python 3.12
VIRTUAL_ENV="$REPO_ROOT/.hf_venv" uv pip install "vllm==0.23" transformers openai
echo ".hf_venv ready."

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Start vLLM server:  bash task_1/start_server.sh"
echo "  2. Run the demo:       .hf_venv/bin/python task_1/nccl_demo.py"
