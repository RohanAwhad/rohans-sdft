# Task 1: NCCL Weight Transfer (HF -> vLLM)

Demonstrates NCCL-based weight transfer from an HF training process to a running vLLM inference server. Proof-of-concept for online weight sync during training.

## Architecture

```
┌─────────────────────┐              ┌─────────────────────┐
│   HF Process        │    NCCL      │   vLLM Server       │
│   (GPU 1)           │ ──────────>  │   (GPU 0)           │
│                     │  data plane  │                     │
│ Load model          │              │ Starts with dummy   │
│ Perturb weights     │    HTTP      │ weights              │
│ Send via NCCL       │ ──────────>  │ Receives weights    │
│                     │ ctrl plane   │ Serves inference    │
└─────────────────────┘              └─────────────────────┘
```

- **Control plane**: HTTP endpoints on vLLM server (`VLLM_SERVER_DEV_MODE=1`)
- **Data plane**: NCCL via `NCCLWeightTransferEngine` (vLLM built-in)
- **No Ray required**

## Setup

```bash
# From repo root
bash task_1/setup.sh
```

Creates two venvs (both need vLLM for NCCL engine):
- `.vllm_venv` - runs the vLLM inference server
- `.hf_venv` - runs the training/sender script

## Usage

```bash
# Terminal 1: start vLLM server on GPU 0
bash task_1/start_server.sh

# Terminal 2: run the demo
.hf_venv/bin/python task_1/nccl_demo.py
```

## Demo flow

1. **Phase 1**: Query vLLM with dummy weights -> gibberish output
2. **Phase 2**: Load real Qwen3-0.6B on GPU 1, transfer via NCCL -> sensible output
3. **Phase 3**: Randomly perturb weights, transfer again -> garbled output

## Key vLLM APIs used

| Side | API | Purpose |
|------|-----|---------|
| Server | `--weight-transfer-config '{"backend":"nccl"}'` | Enable NCCL weight transfer |
| Server | `--load-format dummy` | Start with random weights |
| Server | `POST /init_weight_transfer_engine` | Set up NCCL group |
| Server | `POST /start_weight_update` | Begin weight update |
| Server | `POST /update_weights` | Receive weights via NCCL |
| Server | `POST /finish_weight_update` | Finalize update |
| Server | `POST /pause`, `POST /resume` | Pause/resume inference |
| Trainer | `NCCLWeightTransferEngine.trainer_init()` | Init NCCL on sender side |
| Trainer | `NCCLWeightTransferEngine.trainer_send_weights()` | Broadcast weights |

## Requirements

- 2 GPUs (tested on H100)
- `vllm==0.23` (0.25+ requires FFmpeg for torchcodec)
- `ninja` must be on PATH (included in venv)

## Adapted from

`vllm/examples/rl/rlhf_http_nccl.py` in the [vLLM repo](https://github.com/vllm-project/vllm).
