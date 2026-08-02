#!/usr/bin/env bash
# FSDP spike launcher (gpt-oss-20b).
#   bash megatron_trainer/test_fsdp_spike.sh [GPU_START=2] [NUM_TRAINERS=4]
#
# NOTE: hf-home is mounted WITHOUT :z (SELinux relabel fails on this NVMe fs).

set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
HF_CACHE="${HF_CACHE:-/mnt/nvme5n1/rohan_patched_ckpts/hf-cache}"
GPU_START="${1:-2}"
NUM_TRAINERS="${2:-4}"
TMPDIR="${TMPDIR:-/mnt/nvme0n1/podman_tmp}"
export TMPDIR

GPUS=""
for ((i = 0; i < NUM_TRAINERS; i++)); do
    GPUS="$GPUS --device nvidia.com/gpu=$((GPU_START + i))"
done

mkdir -p logs
echo "=== FSDP spike: model=$MODEL_NAME trainers=$NUM_TRAINERS gpus=$GPU_START..$((GPU_START + NUM_TRAINERS - 1)) ==="

podman run --rm \
    --ipc=host --network=host --pids-limit=-1 \
    --add-host "$(hostname):127.0.0.1" \
    $GPUS \
    -e MODEL_NAME="$MODEL_NAME" \
    -e HF_MODEL_PATH="$MODEL_NAME" \
    -e LOGGING_LEVEL="${LOGGING_LEVEL:-DEBUG}" \
    -v "$WORKSPACE:/workspace:z" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c "
torchrun --nproc_per_node=$NUM_TRAINERS -m megatron_trainer.test_fsdp_spike 2>&1 | tee logs/fsdp_spike.log
"
