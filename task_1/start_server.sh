#!/usr/bin/env bash
# Start vLLM server on GPU 0 with dummy weights and NCCL weight transfer enabled.
# Run from the repo root: bash task_1/start_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
MODEL="Qwen/Qwen3-0.6B"

echo "Starting vLLM server on GPU 0 with dummy weights..."
echo "Model: $MODEL"
echo "Weight transfer backend: nccl"
echo ""
echo "Once the server is ready, run in another terminal:"
echo "  cd $REPO_ROOT && .hf_venv/bin/python task_1/nccl_demo.py"
echo ""

export PATH="$REPO_ROOT/.vllm_venv/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
"$REPO_ROOT/.vllm_venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --enforce-eager \
    --weight-transfer-config '{"backend": "nccl"}' \
    --load-format dummy \
    --gpu-memory-utilization 0.7 \
    --port 8000
