# Self-Distillation Dev Logs

## 2025-07-11 - Task 1: NCCL Weight Transfer (HF -> vLLM)

### Goal
Demonstrate NCCL-based weight transfer from an HF training process to a running vLLM inference server. Proof-of-concept for online weight sync during training.

### Architecture
- **Control plane**: HTTP endpoints on vLLM server (`VLLM_SERVER_DEV_MODE=1`)
- **Data plane**: NCCL via `NCCLWeightTransferEngine` (vLLM built-in)
- **GPU 0**: vLLM server (TP=1, `--load-format dummy`)
- **GPU 1**: HF model (trainer side)
- **No Ray** - uses vLLM's HTTP+NCCL pattern from `examples/rl/rlhf_http_nccl.py`

### Key discovery
- Both sides need vLLM installed (trainer imports `NCCLWeightTransferEngine`)
- vLLM already depends on transformers, so both venvs are similar
- `VLLM_SERVER_DEV_MODE=1` enables dev endpoints: `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/pause`, `/resume`

### Files
- `task_1/setup.sh` - venv creation + deps
- `task_1/start_server.sh` - launches vLLM server on GPU 0
- `task_1/nccl_demo.py` - trainer-side script (3 phases: dummy, real, perturbed)

### Model
- `google/gemma-3-270m-it` (small, fits single GPU easily)

### Status
- [ ] Test on node 01 (rh-h100-01)
