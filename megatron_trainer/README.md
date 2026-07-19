# Megatron Trainer

SDFT training with Megatron Bridge inside NeMo container. All 3 processes (vLLM, trainer, logprob server) run in one container.

## Prerequisites

```bash
gcloud auth application-default login  # one-time per node
```

## Run

```bash
cd /home/lab/rawhad/self_distillation/rohans_sdft && \
TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --name sdft-megatron-adapter \
    --device nvidia.com/gpu=0 \
    --device nvidia.com/gpu=1 \
    --device nvidia.com/gpu=2 \
    --ipc=host \
    --network=host \
    --add-host $(hostname):127.0.0.1 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e RAYON_NUM_THREADS=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e MASTER_ADDR=127.0.0.1 \
    -e PYTHONPATH=/workspace \
    -e MODEL_NAME=Qwen/Qwen3-8B \
    -e HF_MODEL_PATH=Qwen/Qwen3-8B \
    -e NCCL_MASTER_PORT=29500 \
    -e VLLM_PORT=8004 \
    -e TRAIN_DATA_PATH=/workspace/train_dir/data/synthetic_algebra/train_sdft.jsonl \
    -e HINDSIGHT_FIELD=online_feedback \
    -e OUTPUT_DIR=/workspace/output_megatron_adapter_run \
    -e MAX_ADAPTER_TURNS=1 \
    -e GEN_TEMPERATURE=1.0 \
    -e GRAD_ACCUM_STEPS=32 \
    -e NUM_EPOCHS=100 \
    -e SAVE_EVERY=500 \
    -e LEARNING_RATE=2e-6 \
    -e EMA_ALPHA=0.05 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_USE_V1=0 \
    -e BNB_CUDA_VERSION=130 \
    -e VERTEXAI_LOCATION=us-east5 \
    -e WANDB_PROJECT=api-adapter \
    -e WANDB_ENTITY=ronny21 \
    -e WANDB_NAME=sdft_megatron_adapter_run \
    -v $(pwd):/workspace:z \
    -v ~/.cache/huggingface:/root/.cache/huggingface:z \
    -v ~/.config/gcloud:/root/.config/gcloud:ro \
    -v ~/.netrc:/root/.netrc:ro \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash -c '
set -e
pip install --quiet --no-deps vllm==0.23 bitsandbytes safetensors 2>/dev/null
pip install --quiet litellm google-cloud-aiplatform tenacity 2>/dev/null

CUDA_VISIBLE_DEVICES=0 python /workspace/megatron_trainer/start_vllm_patched.py \
    --model "$MODEL_NAME" --port "$VLLM_PORT" --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.5 --weight-transfer-config "{\"backend\":\"nccl\"}" \
    --enforce-eager --no-enable-log-requests &>/workspace/logs/vllm.log &
VLLM_PID=$!

CUDA_VISIBLE_DEVICES=2 python -m megatron_trainer.logprob_server \
    &>/workspace/logs/logprob_server.log &
LOGPROB_PID=$!

for i in $(seq 1 120); do
    curl -s http://localhost:$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    sleep 2
done

CUDA_VISIBLE_DEVICES=1 python -m megatron_trainer.trainer 2>&1
kill $VLLM_PID $LOGPROB_PID 2>/dev/null || true
'
```

To use different GPUs (e.g., 3/4/5), change `--device nvidia.com/gpu=X` flags and adjust `NCCL_MASTER_PORT`/`VLLM_PORT` to avoid conflicts.
