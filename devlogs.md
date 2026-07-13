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

### Status (0.6B)
- [x] Tested end-to-end on node with Qwen3-0.6B
- [x] Loss ~0.64 at epoch 1, weight sync <0.2s, ~20s/optimizer step
- [x] SDPO signal metrics added to wandb (signal_mean, signal_std, len_signal_mean, policy_logp, critic_logp, eos_*)
- [x] EMA teacher weight update: `phi = 0.01 * theta + 0.99 * phi` via `broadcast_weights_ema` with `torch.lerp_`
- [x] Chunked KL computation (KL_CHUNK=128) to reduce peak memory
- [x] bf16 NCCL transfer for teacher log-probs (not float32)

## 2025-07-11 - Scaling to Qwen3-8B

### Problem
8B model + fp32 AdamW on single 80GB GPU = OOM. Model+optimizer baseline ~76 GB, leaving ~3.5 GB for forward/backward.

### Fixes applied (iterative)
1. **Selective lm_head**: `model.model()` (backbone only, hidden states ~30 MB) then `model.lm_head(completion_hidden)` on completion positions only — avoids full `(1, S, V)` logits allocation (~1.16 GB)
2. **Gradient checkpointing on backbone**: wrapped `model.model()` call in `torch.utils.checkpoint.checkpoint(use_reentrant=False)` — hidden states recomputed during backward, not stored
3. **`dtype=` not `torch_dtype=`**: fixed deprecated kwarg — model was likely loading in fp32 (~33 GB) instead of bf16 (~16 GB)
4. **`device_map=DEVICE`**: load directly to GPU, skip CPU→GPU copy (requires `accelerate`)
5. **bitsandbytes AdamW8bit**: halves optimizer state memory (~8 GB vs ~32 GB)
6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: reduces CUDA memory fragmentation

### Result
- Model loads at **16.38 GB** (confirmed bf16)
- First opt_step completed: **loss=0.5377**, comp_len=592, no OOM
- Training running as `rohan-sdft-onpolicy-rohans_data-run-3` on wandb (entity=ronny21, project=sdpo-amortize)
- Checkpoints: `/home/lab/rawhad/self_distillation/rohans_sdft/train_dir/output/epoch_{N}/`

### Dependencies added
- `bitsandbytes==0.49.2`
- `accelerate==1.14.0`

## 2025-07-12 — Training Runs & Hyperparameter Search

### Run 3 (Qwen3-8B, epoch-level sync)
- First successful 8B run after OOM fixes
- Epoch-level weight sync, EMA alpha=0.01, LR=2e-6
- Stopped early — moved to step-level sync

### Run 4 (step-level sync, 10 epochs)
- **Key change**: weight sync after every optimizer step (~140ms overhead)
- Loss: 0.56 → 0.64 → 0.69 → 0.75 → 0.74 → 0.74 → 0.72 → 0.72 → 0.71 → 0.71
- Loss plateaus at ~0.71. Rising initially then stabilizing.
- Weight broadcast timing: ~140-165ms total (EMA ~9ms, vLLM ~120ms)

### Run 5 (cosine LR, 30 epochs)
- Cosine LR schedule from 2e-6 → 0 over 390 steps
- Loss plateaued same as run 4 (~0.69-0.72 range)
- Cosine didn't help vs constant LR

### Run 6 (constant LR + warmup, 10 epochs)
- 1-epoch linear warmup, then constant LR=2e-6
- Loss: same plateau ~0.70
- Warmup had no meaningful effect

### Run 7 (asynth_v1 dataset, GPUs 3/4/5)
- Different dataset: `/home/lab/rawhad/sdg-ki-eval/data/eshwar_datasets/asynth_v1_sdft.jsonl`
- Ran in parallel with run 6 on separate GPUs (3/4/5, vLLM port 8001, NCCL port 29501)

### Run 8 (on-policy overfit, 32 samples)
- 32-sample subset, 500 epochs, rolling checkpoint
- Loss flat at ~0.65-0.72 after 40 epochs — NOT overfitting
- **Root cause**: on-policy = vLLM regenerates completions every epoch (different text each time). The model never trains on the same data twice. Can't overfit a moving target.

### Run 9 (offline overfit, OFFLINE_OVERFIT=1)
- Epoch 1: generate + cache (prompt, completion_ids, teacher_log_probs)
- Epochs 2+: replay cached data, no generation, no teacher NCCL, no weight sync
- **Loss went down**: 0.95 → 0.55 over 45 epochs (crashed at 45 due to NCCL heartbeat timeout on idle logprob server)
- **But model didn't learn**: 4/32 correct vs 3/32 for base model
- **Diagnosis**: reverse KL is mode-seeking → student concentrates mass on teacher's modes, overshoots on high-prob tokens → signal_mean goes negative → student gets sharper but not smarter
- Reverse KL on wrong completions teaches distribution matching, not correctness

### Run 10 (forward KL, on-policy, 32 samples)
- Switched to forward KL: KL(p_teacher || p_student)
- Forward KL = mode-covering, forces student to spread mass where teacher does
- After 67 epochs: model still didn't ingest knowledge
- **Conclusion**: neither KL direction transfers privileged info effectively on its own

### Run 11 (reverse KL, LR=5e-5, EMA alpha=0.05, in progress)
- Reverted to reverse KL
- Bumped LR 25x: 2e-6 → 5e-5
- Bumped EMA alpha 5x: 0.01 → 0.05 (teacher tracks student faster)
- Hypothesis: higher LR + faster teacher tracking = stronger learning signal
- **Status**: running, showing promising results

## Key Findings

### Weight sync timing (8B model)
- EMA broadcast (trainer → teacher): ~9ms
- vLLM sync (trainer → vLLM): ~120ms
- Total per-step overhead: ~140ms (negligible vs ~2min/step)

### Loss plateau analysis
- Reverse KL plateaus at ~0.7 on-policy — this is the irreducible KL from information asymmetry (teacher has privileged info student doesn't)
- Loss starts LOW (~0.49) because student=teacher at init, then RISES as student diverges from slowly-moving teacher
- EMA alpha=0.01 too conservative: teacher barely moves, student runs ahead

### Overfitting experiments
- On-policy can't overfit: data changes every epoch (vLLM regenerates)
- Offline overfit confirms optimizer+reverse KL works mechanically (loss drops)
- But matching distributions on wrong completions ≠ learning correct answers
- Forward KL also failed to transfer knowledge (67 epochs, no improvement)

### Current best config
- Model: Qwen/Qwen3-8B
- Loss: reverse KL
- LR: 5e-5, constant
- EMA alpha: 0.05
- Optimizer: AdamW8bit (bitsandbytes)
- Grad accum: 32 (effective batch)
- Weight sync: step-level (every optimizer step)
- Rolling checkpoint (keep only latest epoch)
