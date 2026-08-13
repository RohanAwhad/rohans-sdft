# SDFT Training — Design

> Partially historical. Current behavior specs live in this dir: `trainer.md`,
> `launch_trainer.md`, `async_rollouts.md`. This file keeps the system-level view.

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
| `torch.distributed` world | trainer ranks only, world_size=N | MCore FSDP grad sync (`TorchFullyShardedDataParallel`) + torch AdamW |
| vLLM `NCCLWeightTransferEngine` | trainer rank 0 <-> vLLM | Weight sync to vLLM (self-contained) |
| Logprob NCCL group | trainer rank 0 <-> logprob server | Weight sync to logprob server (standalone PyNcclCommunicator, same pattern as vLLM's engine) |

## Communication Protocols

- **Weight sync** (large tensors, ~16GB): NCCL — both to vLLM and logprob server
- **Logprob requests** (token IDs in, floats out): TCP data plane (length-prefixed binary, per-rank keepalive) + HTTP control plane (`/health`, `/init_weight_sync`, `/sync_weights`)
- **Gradients**: FSDP — `no_sync()` during accumulation, `finish_grad_sync()` before clip/step
- **Rollout data distribution**: NCCL broadcast within trainer world (`broadcast_object_list`)

## Per-Step Data Flow

```
 1. Rank 0: generate completions via vLLM (HTTP to localhost),
            capture per-token rollout log-probs (logprobs=1, processed mode)
 2. Rank 0: broadcast/scatter rollout sequences + log-probs to all trainer ranks
 3. Each rank: request reference logprobs from logprob server (TCP, independently)
 4. All ranks: forward pass -> current policy logprobs (with grad)
 5. All ranks: compute reverse KL loss, rescaled by the per-sequence
            importance-sampling weight (TIS: clamped mean of per-token
            exp(policy_logp - rollout_logp)); see chunked_head.compute_is_weight
 6. All ranks: backward pass
 7. FSDP: `finish_grad_sync()` (gradient all-reduce deferred by `no_sync`)
 8. All ranks: optimizer step
 9. All ranks: collective FSDP export passes; rank 0 syncs weights -> vLLM (NCCL, NCCLWeightTransferEngine)
10. All ranks: collective FSDP export passes; rank 0 syncs weights -> logprob server (NCCL, standalone group)
```

## Rank 0 Responsibilities

Rank 0 is the only rank that talks to the outside world:
- Rollouts via vLLM HTTP API
- Weight sync to vLLM (NCCL)
- Weight sync to logprob server (NCCL)
- wandb logging

All other ranks only participate in:
- Receiving broadcast data from rank 0
- Requesting reference logprobs from logprob server (TCP, independently)
- Forward/backward pass
- FSDP grad sync (collective, `no_sync`/`finish_grad_sync`)
- Optimizer step

## Launch Pattern

```bash
# GPU 0: vLLM (dev mode, weight-transfer endpoints)
bash train_dir/start_vllm.sh

# logprob server + trainers: nemo container, train_full.sh handles both
bash megatron_trainer/train_full.sh   # full detail: docs/megatron_trainer/launch_trainer.md
```

## Key API Details (from exploration)

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
