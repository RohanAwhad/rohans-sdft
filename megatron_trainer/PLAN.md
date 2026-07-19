# DDP Implementation Plan

Three phases. Each is independently testable. Phase 1 doesn't break existing single-rank training.

Reference: `DESIGN.md` for architecture, `IMPLEMENTATION.md` for detailed per-file changes.

---

## Phase 1: HTTP logprob server + trainer-side client (no DDP yet)

Goal: swap NCCL logprob protocol for HTTP, validate the communication path works with single-rank trainer.

### Files to change

1. **`logprob_server.py`** — full rewrite
   - `init_distributed_standalone()` (world_size=1)
   - FastAPI: `/logprobs` (binary response), `/health`
   - `/init_weight_sync` (PyNcclCommunicator init via `_stateless_init_process_group`)
   - `/sync_weights` (NCCL receive loop + EMA blend)

2. **New `logprob_client.py`** (or add to `vllm_utils.py`)
   - `request_teacher_log_probs_http()` — JSON request, binary response decode
   - `init_logprob_weight_engine()` — mirrors `init_vllm_weight_engine()` pattern
   - `sync_weights_to_logprob_server()` — mirrors `sync_weights_to_vllm()` pattern

3. **`model_utils.py`** — add `init_distributed_standalone()`

4. **`config.py`** — add `LOGPROB_PORT`, `LOGPROB_BASE_URL`

5. **`trainer.py`** — minimal changes:
   - Swap `request_teacher_log_probs` → `request_teacher_log_probs_http`
   - Swap weight sync calls (NCCL command protocol → HTTP+PyNcclCommunicator)
   - Still single-rank, still bitsandbytes optimizer

6. **`nccl_comm.py`** — remove:
   - `CMD_TEACHER_LOGPROBS`, `send_command`, `recv_command`
   - `request_teacher_log_probs`, `handle_teacher_log_probs`
   - Keep `broadcast_weights_ema` as reference (unused)

### Test

Single-rank trainer + HTTP logprob server + vLLM on 3 GPUs.
Validate training runs, loss decreases, same behavior as current NCCL version.

---

## Phase 2: DDP + MegatronDDP + Megatron optimizer

Goal: multi-rank training with gradient parallelism.

### Files to change

1. **`model_utils.py`** — add `init_distributed_trainer()` (torchrun env vars, local_rank)

2. **`trainer.py`** — the main changes:
   - torchrun-compatible init (`LOCAL_RANK`, `RANK`, `WORLD_SIZE` from env)
   - Manual `MegatronDDP(config=model.config, ddp_config=..., module=model)` wrapping
   - `get_megatron_optimizer(OptimizerConfig(...), model_chunks=[ddp_model])` — replace bitsandbytes
   - Remove manual `clip_grad_norm_` (internal to Megatron optimizer)
   - `ddp_model.zero_grad_buffer()` instead of `optimizer.zero_grad()`
   - Rank 0 rollout + `dist.broadcast_object_list()` to all ranks
   - Data slicing: each rank takes `items[rank*M:(rank+1)*M]`
   - Loss scaling: `loss / local_accum_steps` (not `loss / GRAD_ACCUM_STEPS`)
   - `ddp_model.no_sync()` for non-final micro-steps
   - Gate rank 0 operations: vLLM weight sync, logprob weight sync, wandb logging, checkpointing
   - Unwrap `model.module` for weight export functions

3. **`config.py`** — remove `NCCL_MASTER_PORT` (torchrun manages)

### Test

2-rank trainer + HTTP logprob server + vLLM on 4 GPUs. Validate:
- Both ranks load model successfully
- Rollout data broadcasts correctly
- Gradients are synced (loss values consistent across ranks)
- Weight sync to both vLLM and logprob server works
- Checkpointing works (rank 0 only)

---

## Phase 3: Launch scripts + full training run

Goal: production-ready launch + validate training quality.

### Files to change

1. **Launch scripts** — update `smoke_all_in_container.sh` and `train_full.sh` for 4-GPU layout
   - vLLM: GPU 0 (separate process)
   - Trainers: GPUs 1,2 via `torchrun --nproc_per_node=2`
   - Logprob server: GPU 3 (separate process)

2. **Container changes** — pip install updates:
   - Add: `fastapi uvicorn`
   - Remove: `bitsandbytes`

3. **`README.md`** — update with new launch commands

### Training run config

Same hyperparams as the single-rank baseline (`megatron_test_1`), just 4 GPUs with DDP:

```
Model:              Qwen/Qwen3-8B
Data:               /home/lab/rawhad/self_distillation/rohans_sdft/train_dir/data/synthetic_algebra/train_sdft.jsonl
HINDSIGHT_FIELD:    online_feedback
GRAD_ACCUM_STEPS:   32
NUM_EPOCHS:         100
LEARNING_RATE:      2e-6
EMA_ALPHA:          0.05
MAX_ADAPTER_TURNS:  1
GEN_TEMPERATURE:    1.0
SAVE_EVERY:         500
VLLM_PORT:          8004
WANDB_PROJECT:      api-adapter
WANDB_ENTITY:       ronny21
WANDB_NAME:         megatron_ddp_test_1
```

GPU layout (rh-h100-01):

```
GPU 0: vLLM              (separate process, --max-model-len 8192 --gpu-memory-utilization 0.5)
GPU 1: Trainer rank 0     (torchrun)
GPU 2: Trainer rank 1     (torchrun)
GPU 3: Logprob server     (separate process, HTTP + PyNcclCommunicator)
```

Pip installs in container:

```
pip install --quiet --no-deps vllm==0.23 safetensors
pip install --quiet litellm google-cloud-aiplatform tenacity fastapi uvicorn
```

Volume mounts (same as baseline):

```
-v $(pwd):/workspace:z
-v ~/.cache/huggingface:/root/.cache/huggingface:z
-v ~/.config/gcloud:/root/.config/gcloud:ro
-v ~/.netrc:/root/.netrc:ro
-v /home/lab/rawhad:/home/lab/rawhad:ro
```

### Test

1. Run full training (100 epochs, synthetic algebra) on 4 GPUs
2. Compare wandb metrics against single-rank baseline (`megatron_test_1`)
3. Validate: loss curves, pass_rate, training speed improvement

---

## Risk notes

- **Phase 1 is the safe one** — validates HTTP logprob + PyNcclCommunicator weight sync
  without touching the training loop. If anything breaks, it's isolated to the
  communication layer.
- **Phase 2 is the big change** — DDP + new optimizer + new gradient accumulation.
  Most likely source of bugs.
- **Phase 3 is polish** — launch scripts, docs, full validation.
