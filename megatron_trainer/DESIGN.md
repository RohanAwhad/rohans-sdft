# DDP SDFT Training — Design

## GPU Layout (N trainer GPUs)

```
GPU 0:     vLLM inference server
GPU 1:     Trainer rank 0  (master — handles all external I/O)
GPU 2:     Trainer rank 1
...
GPU N:     Trainer rank N-1
GPU N+1:   Logprob server  (reference model)
```

Minimum: 4 GPUs (1 vLLM + 2 trainers + 1 logprob server)

## Process Groups

Three fully independent NCCL groups. No shared world. Each is self-contained.

| Group | Participants | Purpose |
|-------|-------------|---------|
| `torch.distributed` world | trainer ranks only, world_size=N | Megatron DDP gradient all-reduce + distributed optimizer |
| vLLM `NCCLWeightTransferEngine` | trainer rank 0 <-> vLLM | Weight sync to vLLM (self-contained) |
| Logprob NCCL group | trainer rank 0 <-> logprob server | Weight sync to logprob server (standalone PyNcclCommunicator, same pattern as vLLM's engine) |

## Communication Protocols

- **Weight sync** (large tensors, ~16GB): NCCL — both to vLLM and logprob server
- **Logprob requests** (token IDs in, floats out): HTTP — simple request/response, no process group coordination
- **Megatron DDP**: gradient all-reduce + optimizer state sharding via manual `MegatronDDP` wrapping
- **DDP gradients**: Megatron DDP handles gradient all-reduce within trainer world
- **Rollout data distribution**: NCCL broadcast/scatter within DDP group

## Per-Step Data Flow

```
 1. Rank 0: generate completions via vLLM (HTTP to localhost)
 2. Rank 0: broadcast/scatter rollout sequences to all trainer ranks
 3. Each rank: request reference logprobs from logprob server (HTTP, independently)
 4. All ranks: forward pass -> current policy logprobs (with grad)
 5. All ranks: compute reverse KL loss
 6. All ranks: backward pass
 7. DDP: all-reduce gradients across torch.distributed group
 8. All ranks: optimizer step
 9. Rank 0: sync weights -> vLLM (NCCL, NCCLWeightTransferEngine)
10. Rank 0: sync weights -> logprob server (NCCL, standalone group)
```

## Rank 0 Responsibilities

Rank 0 is the only rank that talks to the outside world:
- Rollouts via vLLM HTTP API
- Weight sync to vLLM (NCCL)
- Weight sync to logprob server (NCCL)
- wandb logging

All other ranks only participate in:
- Receiving broadcast data from rank 0
- Requesting reference logprobs from logprob server (HTTP, independently)
- Forward/backward pass
- DDP gradient all-reduce
- Optimizer step

## Launch Pattern

```bash
# GPU 0: vLLM (unchanged)
CUDA_VISIBLE_DEVICES=0 python start_vllm_patched.py \
    --model Qwen/Qwen3-8B --port 8004 ...

# GPU 3: logprob server (HTTP mode, standalone NCCL for weight sync)
CUDA_VISIBLE_DEVICES=3 python -m megatron_trainer.logprob_server --port 8010

# GPUs 1,2: trainer ranks (torchrun for DDP)
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
    -m megatron_trainer.trainer --logprob-url http://localhost:8010
```

## What Changes vs Current Code

| Component | Current | New |
|-----------|---------|-----|
| `trainer.py` | Single rank, rank 0 in world_size=2 | Manual MegatronDDP wrapping, Megatron distributed optimizer, rank 0 is master |
| `logprob_server.py` | NCCL recv/send for data + weights | HTTP server (FastAPI) for logprob requests, standalone PyNcclCommunicator for weight sync |
| `torch.distributed` init | world_size=2 (trainer+logprob) | Trainers: world_size=N via torchrun (Megatron DDP). Logprob server: world_size=1 standalone (for Megatron model loading). No shared world. |
| Rollout | Rank 0 does rollouts, uses directly | Rank 0 does rollouts, broadcasts to all ranks |
| Launch script | 3 processes, shared torch.distributed | 3 independent processes: vLLM, torchrun (N ranks), logprob server |

## Key API Details (from exploration)

### DDP Wrapping

Manual `MegatronDDP` wrapping after `load_model()` (which uses `wrap_with_ddp=False`):

```python
from megatron.core.distributed import DistributedDataParallel as MegatronDDP, DistributedDataParallelConfig
ddp_model = MegatronDDP(config=model.config, ddp_config=DistributedDataParallelConfig(), module=model)
```

- `provide_distributed_model(wrap_with_ddp=True)` exists but is avoided — it re-creates
  the model internally. Manual wrapping lets `load_model()` stay the same for trainer
  and logprob server.
- `ddp_model.module` unwraps to raw model (for weight export, checkpointing).
- `ddp_model.no_sync()` context manager for gradient accumulation (skip all-reduce on non-final micro-steps).

### Optimizer

```python
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
opt_config = OptimizerConfig(optimizer='adam', lr=LEARNING_RATE, bf16=True,
                             clip_grad=MAX_GRAD_NORM, use_distributed_optimizer=True)
optimizer = get_megatron_optimizer(opt_config, model_chunks=[ddp_model])
```

- Gradient clipping is internal (`clip_grad` in config) — no manual `clip_grad_norm_()`.
- `use_distributed_optimizer=True` shards optimizer state across DP ranks.

### Logprob Weight Sync (PyNcclCommunicator)

```python
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import _stateless_init_process_group
pg = _stateless_init_process_group(master_address, master_port, rank, world_size=2, device)
comm = PyNcclCommunicator(group=pg, device=device)
comm.broadcast(tensor, src=0)
```

- `PyNcclCommunicator` takes a process group object, not raw rank/world_size.
- `_stateless_init_process_group()` creates the group (same mechanism as vLLM's `NCCLWeightTransferEngine`).
- Init requires HTTP handshake (trainer sends master_addr/port to logprob server, both create their side of the group).
