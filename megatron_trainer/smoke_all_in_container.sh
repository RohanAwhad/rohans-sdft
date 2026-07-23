#!/bin/bash
# All-in-container smoke test: vLLM + DDP trainers + logprob inside one NeMo container.
#
# Usage:
#   bash megatron_trainer/smoke_all_in_container.sh [GPU_START=3] [NUM_TRAINERS=2]
#
# Layout (4 GPUs): vLLM=GPU_START, trainers=GPU_START+1..N, logprob=GPU_START+N+1

set -euo pipefail

GPU_START=${1:-3}
NUM_TRAINERS=${2:-2}
GPU_VLLM=$GPU_START
GPU_LOGPROB=$((GPU_START + NUM_TRAINERS + 1))
VLLM_PORT=8001
MODEL_NAME="Qwen/Qwen3-8B"

# Build trainer GPU list (e.g. "4,5" for GPU_START=3, NUM_TRAINERS=2)
TRAINER_GPUS=""
TRAINER_DEVICES=""
for i in $(seq 1 $NUM_TRAINERS); do
    gpu=$((GPU_START + i))
    TRAINER_GPUS="${TRAINER_GPUS:+$TRAINER_GPUS,}$gpu"
    TRAINER_DEVICES="$TRAINER_DEVICES --device nvidia.com/gpu=$gpu"
done
# Internal CUDA indices for trainers (0-indexed after CUDA_VISIBLE_DEVICES remapping)
TRAINER_CUDA_INTERNAL=""
for i in $(seq 0 $((NUM_TRAINERS - 1))); do
    TRAINER_CUDA_INTERNAL="${TRAINER_CUDA_INTERNAL:+$TRAINER_CUDA_INTERNAL,}$i"
done

WORKSPACE=$(cd "$(dirname "$0")/.." && pwd)
HF_CACHE=${HF_HOME:-$HOME/.cache/huggingface}
HOSTNAME_FIX=$(hostname)
export TMPDIR=${TMPDIR:-/mnt/nvme0n1/podman_tmp}

mkdir -p "$WORKSPACE/logs"

echo "=== SDFT DDP Smoke Test ==="
echo "GPUs: vLLM=$GPU_VLLM, Trainers=$TRAINER_GPUS, Logprob=$GPU_LOGPROB"
echo "NUM_TRAINERS=$NUM_TRAINERS"

cleanup() {
    podman stop sdft-smoke 2>/dev/null || true
    podman rm sdft-smoke 2>/dev/null || true
}
trap cleanup EXIT

TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-smoke \
    --device "nvidia.com/gpu=$GPU_VLLM" \
    $TRAINER_DEVICES \
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
    -e LOGPROB_PORT=8010 \
    -e VLLM_PORT="$VLLM_PORT" \
    -e OUTPUT_DIR="/workspace/output_smoke" \
    -e NUM_EPOCHS=1 \
    -e GRAD_ACCUM_STEPS=$((NUM_TRAINERS * 2)) \
    -e WANDB_MODE=disabled \
    -e SAVE_EVERY=9999 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_USE_V1=0 \
    -e BNB_CUDA_VERSION=130 \
    -e VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-east5}" \
    -e VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-}" \
    -e HINDSIGHT_FIELD=online_feedback \
    -v "$WORKSPACE:/workspace:z" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c "
set -e
pip install --quiet --no-deps vllm==0.23 bitsandbytes safetensors 2>/dev/null
pip install --quiet litellm google-cloud-aiplatform tenacity fastapi uvicorn 2>/dev/null

NUM_GPUS=\$(nvidia-smi -L | wc -l)
LOGPROB_GPU=\$((NUM_GPUS - 1))
# Trainer GPUs: 1..NUM_GPUS-2 (internal indices)
TRAINER_LAST=\$((NUM_GPUS - 2))

echo \"=== Starting vLLM on internal GPU 0 ===\"
CUDA_VISIBLE_DEVICES=0 python /workspace/megatron_trainer/start_vllm_patched.py \\
    --model \"\$MODEL_NAME\" \\
    --port \"\$VLLM_PORT\" \\
    --max-model-len 8192 \\
    --dtype bfloat16 \\
    --gpu-memory-utilization 0.85 \\
    --weight-transfer-config '{\"backend\":\"nccl\"}' \\
    --enforce-eager \\
    --no-enable-log-requests \\
    &>/workspace/logs/vllm_smoke.log &
VLLM_PID=\$!

echo \"=== Starting logprob server on internal GPU \$LOGPROB_GPU ===\"
CUDA_VISIBLE_DEVICES=\$LOGPROB_GPU python -m megatron_trainer.logprob_server \\
    &>/workspace/logs/logprob_smoke.log &
LOGPROB_PID=\$!

echo \"=== Waiting for vLLM... ===\"
for i in \$(seq 1 120); do
    curl -s http://localhost:\$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done
echo \"vLLM ready\"

echo \"=== Starting $NUM_TRAINERS-rank trainer via torchrun on internal GPUs 1..\$TRAINER_LAST ===\"
TRAINER_CUDA=\$(seq -s, 1 \$TRAINER_LAST)
CUDA_VISIBLE_DEVICES=\$TRAINER_CUDA torchrun --nproc_per_node=$NUM_TRAINERS \\
    -m megatron_trainer.trainer 2>&1

echo \"=== Training done ===\"
kill \$VLLM_PID \$LOGPROB_PID 2>/dev/null || true
" 2>&1 | tee "$WORKSPACE/logs/smoke.log"

echo "=== Smoke test complete ==="
