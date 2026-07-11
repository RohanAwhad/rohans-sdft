# Task 2: Pure NCCL Logprob Server

Two-process system that uses NCCL (GPU-to-GPU) communication to transfer model weights and compute logprobs — no HTTP involved.

## Architecture

- **Rank 0 — Trainer** (`src/trainer.py`): Holds the training copy of the model. Accepts interactive commands via stdin.
- **Rank 1 — Server** (`src/server.py`): Holds a reference copy of the model. Runs a blocking command loop, reacting to signals from rank 0.
- **Comm layer** (`src/nccl_comm.py`): NCCL-based communication utilities — command signals, token/logprob transfer, and full weight sync via `dist.broadcast`.

## How it works

1. Both processes load the same model (`Qwen/Qwen3-0.6B` by default) on separate GPUs
2. Trainer sends commands to server via NCCL broadcast signals (`CMD_LOGPROBS`, `CMD_SYNC_WEIGHTS`, `CMD_SHUTDOWN`)
3. **Logprobs**: Trainer broadcasts token IDs → server runs forward pass → server broadcasts logprobs back
4. **Weight sync**: Both ranks call `broadcast_weights(model, src=0)` — every parameter is broadcast from trainer to server, overwriting the server's model in-place

## Usage

```bash
GPU_TRAINER=2 GPU_SERVER=3 python test_e2e.py
```

Or via the launch script:

```bash
bash launch.sh
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GPU_TRAINER` | `2` | CUDA device for trainer (rank 0) |
| `GPU_SERVER` | `3` | CUDA device for server (rank 1) |
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` | HuggingFace model to load |
| `MASTER_PORT` | `29500` | NCCL rendezvous port |

## E2E test

The test (`test_e2e.py`) proves NCCL weight push works:

1. Get initial logprobs (v1)
2. Perturb trainer weights randomly
3. Sync perturbed weights to server via NCCL
4. Get updated logprobs (v2)
5. Compare — success if `mean |diff| > 0.01`
