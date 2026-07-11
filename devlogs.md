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
- `Qwen/Qwen3-0.6B` (non-gated, fits single GPU easily)

### Gotchas encountered
- `uv venv` doesn't include pip; use `VIRTUAL_ENV=... uv pip install` instead
- vLLM 0.25.0 unconditionally imports `torchcodec` (video support) which needs FFmpeg system libs; pinned to `vllm==0.23`
- vLLM spawns child processes (EngineCore) that need `ninja` on PATH; must `export PATH="$REPO_ROOT/.vllm_venv/bin:$PATH"` in start script
- Gemma 3 is gated on HF; switched to Qwen3-0.6B

### Status
- [x] Tested on node 01 (rh-h100-01) - all 3 phases pass
  - Phase 1 (dummy weights): gibberish output confirmed
  - Phase 2 (real weights via NCCL): sensible output confirmed
  - Phase 3 (perturbed weights via NCCL): garbled output confirmed

## 2025-07-11 - SDFT Training Loop (train_dir/)

### Goal
Full on-policy Self-Distillation Fine-Tuning loop using reverse KL divergence.

### Architecture (4 processes, 3 GPUs)
- **GPU 0**: vLLM server — rollout generation via HTTP `/v1/completions`
- **GPU 1**: Trainer — student model, backward pass, orchestrator
- **GPU 2**: Logprob server — teacher log-probs via pure NCCL
- **GPU 3**: spare

### Communication
- Trainer <-> vLLM: HTTP (generation) + NCCL via `NCCLWeightTransferEngine` (weight sync)
- Trainer <-> Logprob server: pure NCCL via `torch.distributed` (log-probs + weight sync)
- Two independent NCCL groups coexist without conflict

### Training loop (per step)
1. Collator produces `prompt_text` (student) and `conditional_text` (teacher, with `enriched_user_response`)
2. vLLM generates completion from `prompt_text` (HTTP)
3. Student forward: `[prompt + completion]` → logits at completion positions (with grad)
4. Teacher log-probs: send `[cond_prompt + completion]` to logprob server → receive full `(C, V)` log_softmax via NCCL
5. Reverse KL: `KL(p_student || p_teacher) = sum_v p_s(v) * (log p_s(v) - log p_t(v))`, averaged over tokens
6. Backward + gradient accumulation (effective batch = 32)

### Key design decisions
- **Reverse KL** (not SDPO policy gradient) — full distribution-level distillation
- **Full (C, V) log-softmax transfer** — on H100 NVLink (~900 GB/s), 512 * 151936 * 4 bytes = ~300MB takes <0.4ms
- **Custom training loop** (not HF Trainer) — vLLM + NCCL coordination too custom for Trainer's compute_loss
- **vLLM loads real weights** — all 3 models start from same checkpoint, sync at epoch boundaries only
- **Per-sample NCCL** for teacher — 0.6B model is fast, batching adds protocol complexity

### Config
- Model: Qwen/Qwen3-0.6B
- LR: 2e-6, constant, AdamW
- Batch: 1 * 32 grad_accum = 32 effective
- Epochs: 10
- Data: 400 examples (train_maas_sdft.jsonl), hindsight=enriched_user_response

### Files
- `train_dir/setup.sh` — venv creation
- `train_dir/start_vllm.sh` — vLLM server on GPU 0
- `train_dir/launch.sh` — logprob server (bg) + trainer (fg)
- `train_dir/src/config.py` — all hyperparams (env-overridable)
- `train_dir/src/collator.py` — SDFTCollator (adapted from reference OnPolicySDFTCollator)
- `train_dir/src/nccl_comm.py` — NCCL protocol with full logits transfer
- `train_dir/src/logprob_server.py` — teacher process (GPU 2)
- `train_dir/src/vllm_utils.py` — HTTP client + weight sync (task 1 pattern)
- `train_dir/src/trainer.py` — main loop + reverse KL loss

### Status
- [ ] Implemented, not yet tested on node
