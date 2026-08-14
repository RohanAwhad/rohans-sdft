# Self-Distillation Dev Logs

## 2026-08-14 - Phases 0-2 complete: Layer 1 PASS + streaming overlap verified

- **Phase 2 verified on rh-h100-12** (64-sample smoke, ASYNC_ROLLOUT=1, temp=1.0): 16 steps, epoch drained cleanly, `TIMING ... producer_wait=0.0s gen_overlap=11.4s` — generation fully hidden behind training; whole run ~3 min vs sync ~15 min.
- **Two bugs found + fixed by the streaming smoke**:
  1. Rust tokenizer is **not thread-safe** ("Already borrowed"): in async mode the collator runs in the producer thread while the main thread encodes — fixed with a separate tokenizer instance for the producer side (collator + envs + produce()).
  2. `step_done_q` token put in streaming mode would deadlock the main path (maxsize=1, nobody consumes) — now gated on `ASYNC_IN_ORDER`.
- `verify_layer1.py` added to the repo (Layer 1 script: within-run assignment + cross-mode stream checks).
- Commits: 96fec5c (devlogs), f2bc1a8 (tokenizer + step_done_q fixes), 4586195 (verify tool + lag on TIMING).
- Remaining per plan: Phase 3 verification campaign (Layer 2/3 A/B + evals), Phase 4 tuning, Phase 5 producer-refactor issue.

## 2026-08-14 - Layer 1 verification: determinism findings + replay-based PASS

- **Determinism rabbit hole (important findings, all verified on rh-h100-12)**:
  - vLLM 0.23 per-request `seed` is **not cross-restart deterministic**: two fresh engines, same prompts, same seed → different completions (in-session repeats DO match). Confirmed with a standalone boot→generate→kill→boot→generate test.
  - With `GEN_TEMPERATURE=0` (greedy) + `TRAINER_SEED` (fixed shuffle): sync-vs-sync runs match exactly on **batch 0 + step 1**, then drift — training-kernel numerics (flash-attn/TE atomics) accumulate and flip greedy argmax ties from batch 1 on. Cross-run bit-identity beyond step 1 is impossible without `use_deterministic_algorithms` (breaks the TE stack).
  - Even batch-0 greedy differs sync-vs-async (different arrival dynamics → different chunked-prefill batching → argmax flips). vLLM-internal numerics are not cross-process-structure reproducible.
- **Layer 1 restructured into a replay-based procedure** (this is now the canonical verification):
  - `RECORD_ROLLOUT_PATH` dumps every vLLM result keyed by prompt hash; `ROLLOUT_REPLAY_PATH` replays them — both modes train on byte-identical rollout data, isolating the plumbing from vLLM numerics.
  - `DEBUG_ROLLOUT_HASH` logs `ROLLOUT_HASH` (produced, batch/idx) + `CONSUME_HASH` (consumed, step/rank/micro) on every rank.
  - Results on 16-sample smoke (G=4, W=2): sync within-run assignment **PASS**, async in-order within-run assignment **PASS** (column-major `r*L+k` verified on every microbatch), cross-mode produced hash streams **identical**, cross-mode step-1/2 loss/grad_norm **bit-identical**, steps 3-4 match to 3 decimals (kernel-numerics drift, same as sync-vs-sync). **Layer 1 PASS.**
- Commits: 03f4d3b (greedy + docs), ee05039 (CONSUME_HASH + 3-part Layer 1), 8af1979 (replay mode), c2c94ee (removed accidentally committed bench_forward.py).
- Streaming smoke (ASYNC_ROLLOUT=1, real temp=1.0, 64 samples) launched — Phase 2 verification in flight.

## 2026-08-13 - Streaming rollouts: Phases 0-2 implemented, Layer 1 in flight

- Branch `ra/async-rollout` rebased onto v0.1.0. Plan finalized in `plan.md` (5 phases). Commits: 8d1f996 (config plumbing), 40ed0e0 (in-order restructure), 9e60237 (streaming producer).
- **Phase 0**: `ASYNC_ROLLOUT` / `ASYNC_IN_ORDER` / `N_ASYNC` (default 2×GRAD_ACCUM_STEPS) in config.py; `IS_CAP` default 2.0→5.0; `TRAINER_SEED`/`VLLM_SEED` determinism knobs (vLLM per-request seed via completions API); passthrough in train_full.sh/smoke_all_in_container.sh/.env.example; wandb config gains async_rollout/async_in_order/n_async.
- **Phase 1**: `produce()` extracted (sync + in-order share it — per-sample metas via `_sample_meta`/`_aggregate_pass_rate` keep stats identical between modes); `_train_sample()`/`_step_tail()` shared; rank-0 producer thread + bounded queue (`maxsize=N_ASYNC+W+1`) + per-microbatch `_pull_microbatch` (pop W, broadcast); in-order producer pushes column-major (`s = r*L + k`) so rank r sees exactly its sync-mode samples.
- **Bug found via play.py sim**: done-Event check-then-block race — producer can set done while consumer is already blocked in `q.get()` → permanent hang. Fixed with a **queue sentinel** (`_ROLLOUT_SENTINEL` pushed last — signal travels through the queue, immune to the race). Sim verified: 12 steps, rank assignments bit-identical to sync slicing, 15 residual samples dropped.
- In-order overlap prevention uses a **token queue** (`step_done_q`, maxsize=1) instead of an Event — a persistent Event accumulates stale sets and lets the producer run ahead (overlap → Layer 1 breaks).
- **Phase 2**: `_produce_streaming` — ThreadPoolExecutor(N_ASYNC) + sliding window of futures, per-sample push on completion (completion-order reordering = the feature), `fut.result()` re-raise → crash-hard via excepthook; exact `submit_limit = steps_per_epoch * GRAD_ACCUM_STEPS`; sentinel at end. Sim verified: 384 pushed == consumed, completion-order reordering confirmed.
- policy_version stamped at generation start (`_OPTIMIZER_STEP` global), `policy_lag` mean/max + TIMING `producer_wait`/`gen_overlap` logged (async only).
- **Layer 1 on rh-h100-12** (in flight): sync baseline smoke (GPU 0-3, `HF_HOME=/mnt/nvme0n1/rawhad_hf`, `VLLM_PORT=8007`, `TRAINER_SEED=42 VLLM_SEED=42`, smoke_sdft.jsonl 64 samples, 16 steps) → then async in-order run → compare opt_step log lines + epoch_1 safetensors checksums.
- Gotchas: tmux session died on first attempt (podman statfs error — HF_HOME unset); node was on stale branch `ra/analyze-kd-agentic-search` @ 41cef02 with local smoke tweak (equivalent change already in branch → discarded, `checkout -B` to origin).

## 2026-08-12 - Async rollout research (PRIME-RL + VERL) + spec

- Deep-researched both async rollout engines (code-only, no web): `~/3_resources/external_libs/prime-rl` @ e8abfa26 and `verl` @ 535c4779 (volcengine fork, v1 era). Docs: `docs/research/RESEARCH_async_rollouts_prime_rl.md`, `RESEARCH_async_rollouts_verl.md`, `async_rollouts_porting_analysis.md`.
- Key discovery: local prime-rl is a **full rewrite** (no SyncReplayBuffer / forward-only agents / executor dir — old design gone). New design: vLLM pool + CPU asyncio orchestrator + torchrun FSDP2 trainer, `max_async_level=1`, staleness cap `max_off_policy_steps=8`, token-ratio IS loss — architecturally closest to our SDFT layout.
- verl v1: fire-and-forget agent loops + TransferQueue, `ReplayBufferAsync` staleness eviction (threshold 8, drop/wait), partial-rollout abort-resume (`FullyAsyncLLMServerClient`), Decoupled PPO `parameter_sync_step`, trainer modes sync/colocate_async/separate_async.
- Porting recommendation: producer thread on rank 0, 1-ahead (sync-then-gen), keep broadcast handoff + existing `IS_WEIGHTING`/`IS_CAP`; ~100-line diff in `trainer.py`. N-ahead + DPPO masks + partial rollout = future work if gen-bound.
- Spec written: `docs/megatron_trainer/async_rollouts.md` (`ASYNC_ROLLOUT` flag, off by default) + TODOS section. This work lives on branch `ra/async-rollout`.

## 2026-08-12 - Smoke on rh-h100-12 + repo sync

- Smoke attempt on rh-h100-12: port 8001 taken by lab's trl vLLM (GPUs 6,7) → crash `Address already in use`. Patched `smoke_all_in_container.sh:15` to `VLLM_PORT=${VLLM_PORT:-8001}` (local + node sed). Port 8011 collided with the logprob **TCP** default (`LOGPROB_TCP_PORT`); 8012 taken; finally relaunched with `VLLM_PORT=8007` in tmux `sdft-smoke`. Result pending.
- Repo sync: local branch was behind origin (remote gained reflector-fix 61bcd62 + merges) → rebased our 6 TODO commits, pushed to `ra/analyze-kd-agentic-search`, pulled on node (41cef02).

## 2026-08-12 - TODOs batch: grad_norm logging, wandb config, max-model-len, batch-path deletion

- `trainer.py`: capture `clip_grad_norm_` return → extend existing SUM all-reduce to 3 elems (loss, samples, grad_norm²) so rank 0 logs the **global** norm (FSDP shard-local norms sum to global). `train/grad_norm` in wandb `log_dict` + per-step log line.
- `trainer.py`: wandb config now includes `max_grad_norm`, `max_total_len`, `student/teacher_max_prompt_len`, `ema_alpha`, `teacher_model` (empty = internal EMA teacher); imports `EMA_ALPHA`, `MAX_TOTAL_LEN`.
- `train_full.sh`: `--max-model-len "$MAX_TOTAL_LEN"` (was hardcoded 16384); `MAX_TOTAL_LEN` passthrough already existed.
- Deleted unused batch logprob path entirely (per request, not un-chunked): `LOGPROB_BATCH_SIZE` env + passthrough, `BatchLogprobRequest`, `/logprobs_batch` endpoint, `request_teacher_log_probs_batch_http`, trainer import. Trainer uses per-rank TCP only.
- Port-scan race TODO dropped from scope (user decision).
- Verification: only Mac-side (`py_compile` / `bash -n`) + zero leftover refs via rg. **Full cluster smoke still pending** — covers all 4 changes + the `MAX_TOTAL_LEN` smoke item.

## 2026-08-11 - FSDP-only trainer (TODOs 3+4) + smoke on ai-innovation-h100-12

- `trainer.py`: removed DDP + bitsandbytes AdamW8bit branches; hardcoded MCore FSDP wrap + torch AdamW; `ddp_model` → `fsdp_model`; weight-sync/ckpt now unconditional (FSDP export is collective); wandb `backend` → `"fsdp"`.
- `config.py`: dropped `TRAINER_BACKEND` env read.
- `train_full.sh`: dropped `-e TRAINER_BACKEND` + `bitsandbytes` install; optional envs (`WANDB_*`, `TEACHER_MODEL_PATH`, ...) only passed when set (empty `-e WANDB_BASE_URL=""` crashed wandb Settings); `:z` mount suffix conditional on `getenforce` (permissive nodes can't relabel lab-owned files).
- Smoke: `train_full.sh 0 2 1` with no `TRAINER_BACKEND`, Qwen3-8B, 100-sample eval jsonl, 2 GPUs, 50 optimizer steps. FSDP wrap + torch AdamW logged; 0 bitsandbytes references; `TIMING step=50` OK; vLLM + logprob weight syncs 200 OK; ckpts `epoch_1`/`step_50` saved. avg_loss=0.46.
- Node quirks (ai-innovation-h100-12-preserve): had to kill lab's vLLM jobs on GPUs 2,3,6,7; HF cache copied to rawhad-owned dir (Megatron-Bridge lock file needs write); port 8001 occupied → `VLLM_PORT=8101`.


## 2026-08-11 - Required-env fail-fast (TODO 2)

- `MODEL_NAME` / `TRAIN_DATA_PATH` now required (no defaults) per `launch_trainer.md`.
- `train_full.sh`: `:?` guards at lines 49-50 (exit 1 with message before container launch).
- `config.py`: raise `ValueError` at import if either is unset/empty (line 5/102).
- Deleted `megatron_trainer/repro_fixed_8192.py` (one-off 2026-08-03 OOM repro; conclusion recorded in 2026-08-03 entry, cap enforced in config).
- Verified: unset MODEL_NAME → exit 1; unset TRAIN_DATA_PATH → exit 1; both set → proceeds to podman launch.


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
- All epoch checkpoints saved

## 2025-07-13 — Data Enrichment & Multi-Dataset Runs

### enriched_user_response pipeline
- Script: `sdg-ki-eval/scripts/generate_enriched_user_response.py`
- Uses Claude Sonnet 4 (via AnthropicVertex) to produce focused doc excerpts from source docs
- Input: source_docs.jsonl (69 MaaS doc chunks) + question + golden answer
- Output: minimal documentation excerpt that supports the answer (without including the answer itself)
- 50 concurrent workers, checkpoints every 100 rows, idempotent (skips existing)

### Enriched datasets generated
| Dataset | Rows | Source | Time |
|---------|------|--------|------|
| `sdg_hub_sft_sdft_enriched.jsonl` | 3000 | sdg_hub_sft_sdft.jsonl | ~7 min |
| `oumi_sdft_enriched.jsonl` | 3000 | oumi_sdft.jsonl | ~7 min |
| `asynth_kd_sdft_enriched.jsonl` | 3000 | asynth_kd_sdft.jsonl | ~7 min |

All at: `/home/lab/rawhad/sdg-ki-eval/data/eshwar_datasets/`

### Run 12 (asynth_v1, LR=5e-5, EMA=0.05, 10 epochs)
- Same hyperparams as run 11 but full asynth_v1 dataset (not 32 samples)
- GPUs 3/4/5, vLLM port 8001, NCCL port 29501
- Stopped early

### Run 13 (sdg_hub enriched, LR=5e-5, EMA=0.05, 10 epochs)
- First run with enriched sdg_hub data (3000 samples)
- GPUs 0/1/2, vLLM port 8000
- vLLM max-model-len bumped 4096→8192 (some sdg_hub prompts ~2100 tokens)
- Stopped early

### Run 14 (rohans_data, LR=5e-5, EMA=0.05, 10 epochs)
- 400 samples, `train_maas_sdft.jsonl` (has `enriched_user_response`)
- Loss=0.3618 at step 1, ran well
- Completed 10 epochs, final avg_loss=0.154
- Checkpoints: `output_run_14/epoch_{N}/`

### Infra changes
- **Step-level checkpoints**: save every `SAVE_EVERY` steps (default 200) to `step_{N}/` instead of epoch-level
- **`HINDSIGHT_FIELD` env var**: configurable collator field (default `enriched_user_response`, set to `user_response` for non-enriched data)
- **vLLM max-model-len**: 4096→8192 (sdg_hub prompts can be ~2100 tokens)

## 2026-08-03 - Fixed 8192-Token FSDP Reproduction

### Goal
Reproduce the gpt-oss-20b backward OOM with a deterministic 2048-token prompt and 6144-token completion.

### Result
- Added `megatron_trainer/repro_fixed_8192.py` with four-rank MCore FSDP, Adam state warmup, and eight fixed accumulation microsteps.
- All eight reverse-KL backward passes completed on four H100 80GB GPUs.
- Peak allocated memory was 71.66 GiB per rank; recompute was active (`full/uniform/1`).
- Student logits are fp32 with shape `(1, 8192, 201088)`.
- The earlier 6.14 GiB OOM came from allowing an 8192-token completion in addition to the prompt, not an 8192-token total sequence.

### Decision
- Cap training sequences at 8192 total tokens: 2048 prompt + 6144 generation.
- Validate the cap in config so overlength launches fail before model loading.
- CP=2 is not required for this sequence length.

### Run 15 (sdg_hub non-enriched, 2 epochs, SAVE_EVERY=10)
- 3000 samples, `sdg_hub_sft_sdft.jsonl` (no `enriched_user_response`)
- `HINDSIGHT_FIELD=user_response` (teacher sees correct answer only, no docs)
- Stopped early

### Run 16 (train_rag_knowledge, 2 epochs, SAVE_EVERY=10, in progress)
- 400 samples, `train_rag_knowledge.jsonl` (has `enriched_user_response`)
- Eval set available: `eval_rag_knowledge.jsonl`
- opt_step=1 loss=0.1385, 26 total steps
- Checkpoints: `output_run_16/step_{N}/` (steps 10, 20, final at 26)

## 2026-08-02 — Phase 1: FSDP backend for gpt-oss-20b (verified)

### Problem
DDP+bnb on openai/gpt-oss-20b OOM'd (~250 GB/rank needed vs 80 GB). Moved trainer to MCore-native FSDP (torch AdamW, no bnb, 4 trainers, GRAD_ACCUM_STEPS=32).

### Implementation (TRAINER_BACKEND=fsdp, default stays ddp)
- `trainer.py`: wrap via `TorchFullyShardedDataParallel(config, DistributedDataParallelConfig(use_distributed_optimizer=False), model)`; `finish_grad_sync()` before clip; torch AdamW instead of bnb.
- Weight sync became collective: FSDP export/gather passes are NCCL collectives on the 4-rank mesh → **all ranks** must enter them; rank 0 keeps/sends/writes.
  - vLLM sync: rank 0 runs existing path; ranks 1-3 consume the same export passes.
  - logprob sync: new `gather_raw_params_iter()` (raw `model.parameters()` order, `param.full_tensor()` lockstep, rank 0 broadcasts on the separate 2-rank logprob NCCL group). Server code unchanged (order-indexed protocol).
  - `save_hf_checkpoint(..., rank=rank)`: export on all ranks, file writes only on rank 0.
- `train_full.sh`: `-e TRAINER_BACKEND` passthrough.

### Blockers found & fixed (important for future runs)
1. **Bridge deadlock was a misuse, not a bug**: `bridge.export_hf_weights` already all-gathers DTensors per-tensor (`uneven_dtensor_to_full_tensor`); it's a collective — must be called on ALL ranks ("All ranks get full tensors").
2. **`fully_shard` replaces module classes** (FSDPColumnParallelLinear etc.) → bridge `_detect_parallelism_type` fails on unknown names. Fix: `register_fsdp_module_mappings()` (model_utils.py) registers the 5 FSDP-prefixed classes into `AutoMapping`.
3. **Qwen3 tied-embedding + FSDP gap (MCore bug, production-safe)**: tied output_layer owns no weight; `shared_embedding_or_output_weight()` returns the at-rest embedding DTensor → native matmul gets mixed Tensor/DTensor. gpt-oss-20b (untied) unaffected. Workaround for tests: patch `GPTModel.shared_embedding_or_output_weight` to `full_tensor()`.
4. DDP on 20b impossible even for fwd-only: `_ddp_init_helper` eagerly allocates 38.96 GB fp32 grad buffer.

### Verification (T1 + T1b, all PASSED)
- T1 (`test_fsdp_t1.py`, gpt-oss-20b, 4 ranks): FSDP wrap → 1 micro-step (no_sync+finish_grad_sync+AdamW) → collective `save_hf_checkpoint` → export-pass hashes == checkpoint hashes (6 tensors) → raw-order gather == checkpoint embed/lm_head (34 hashes) → **peak 53.51 GB/rank**.
- T1b (`test_ddp_fsdp_parity.py`, Qwen3-0.6B, 4 ranks): DDP vs FSDP forward loss **bit-identical (diff=0.0)**.
- Launchers: `test_fsdp_t1.sh` / `test_ddp_fsdp_parity.sh` (podman, HF_CACHE=/mnt/nvme5n1/rohan_patched_ckpts/hf-cache, no `:z` on cache mounts).

### Next
- T2 smoke: full loop 1 epoch (vLLM ×2 GPUs + FSDP trainer ×4 + logprob ×1), SAVE_EVERY=9999, wandb off, GPUs 0-1/2-5/6. Validates NCCL group interleaving (FSDP mesh + vLLM group + logprob group) in production flow.
- Then full 10-epoch gpt-oss-20b run.

## 2026-08-02 — T2 smoke PASSED (gpt-oss-20b, FSDP + bf16 vLLM)

### vLLM bf16 fix (unblocks weight updates on gpt-oss)
- `openai/gpt-oss-20b` config.json declares `quantization_config.quant_method=mxfp4` → vLLM 0.23 auto-serves `gpt_oss_mxfp4` and `_load_weights_mxfp4` rejects bf16 HF tensors: `/update_weights` → 500 `KeyError: 'layers.0.mlp.experts.w13_weight'`.
- Fix: serve **`unsloth/gpt-oss-20b-BF16`** (pure bf16 conversion, no `quantization_config` in config.json, same arch/names, untied). Trainer still loads `openai/gpt-oss-20b` (identical weights).
- `train_full.sh`: `--gpu-memory-utilization 0.5 → 0.8` (bf16 weights = 42GB vs 13.6GB mxfp4; 0.8 → 65GB budget, fits).
- Download: `HF_HOME=/mnt/nvme5n1/rohan_patched_ckpts/hf-cache huggingface-cli download unsloth/gpt-oss-20b-BF16` (~42GB, cached at `models--unsloth--gpt-oss-20b-BF16`).
- Earlier fix in this run series: `start_vllm_patched.py` no-ops `initialize_layerwise_reload` on both `reload.layerwise` and `reload` bindings (`start_weight_update` was 500 `w13_weight already exists`).

### T2 result (2 optimizer steps, 64-example slice, 1 epoch)
- Layout `train_full.sh 0 4 2`: vLLM GPUs 0-1 (ports 8001/8101, 65.6GB ea), trainers 2-5 (~70-75GB), logprob GPU 6.
- `TRAINER_BACKEND=fsdp HINDSIGHT_FIELD=user_response NUM_EPOCHS=1 GRAD_ACCUM_STEPS=32 SAVE_EVERY=9999 WANDB_MODE=disabled OUTPUT_DIR=/mnt/nvme5n1/rohan_patched_ckpts/sdft_gptoss_20b_smoke`
- opt_step=1 loss=2.07, opt_step=2 loss=1.90; epoch avg_loss 0.97-1.09 across ranks.
- **Both vLLM syncs (steps 1+2) succeeded**: `POST /start_weight_update`/`/update_weights`/`/finish_weight_update` all 200 OK on 8001 and 8101. Logprob sync OK. No deadlocks — NCCL group interleaving validated end-to-end.
- Checkpoints: `epoch_1` + `step_2` (411 tensors each) → `/mnt/nvme5n1/rohan_patched_ckpts/sdft_gptoss_20b_smoke/`.
- Timing step 2: total=100.5s (gen=42.0 teacher=29.4 student=11.9 loss_bwd=6.1 optim=1.8 wsync=9.2).
- Logs: `logs/training.log`, `logs/t2_bf16.log`, `logs/vllm_{0,1}.log`.

### Next
- Full 10-epoch gpt-oss-20b run (same env, full train set `data/maas_raft_v3.1/train_sdft.jsonl`), then eval checkpoints on host via transformers.

## 2026-08-04 - Smoke 8192-fix: v1 OOM (fragmentation) → v2 PASSED; run 21 launched

### Context
- Reproduced the exact t2 smoke shape (64-sample `smoke_sdft.jsonl`, `GRAD_ACCUM_STEPS=32`, 4-rank FSDP, `unsloth/gpt-oss-20b-BF16`, 1 epoch = 2 optimizer steps) with the 8192-total cap fix (2048 prompt + 6144 gen).

### Smoke v1 (OOM, `logs/smoke_8192fix.log`)
- Step 1 passed (226.6s), step 2 OOM'd at `scaled_loss.backward()` (trainer.py:432).
- `Tried to allocate 4.60 GiB`, 4.24 GiB free, 74.93 in use; 65.15 allocated + **5.75 reserved-but-unallocated** (fragmentation).
- Needed 69.75 GiB < 71.66 GiB repro peak → NOT a capacity problem; allocator fragmentation from variable-length completions.
- Cause: `PYTORCH_CUDA_ALLOC_CONF` was empty (train_full.sh:100 defaults it empty); previous smokes passed it explicitly.

### Smoke v2 (PASSED, `logs/smoke_8192fix_v2.log`)
- Only change: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- step 1: 191.9s, loss=0.2483, comp_len=1954 · step 2: 277.0s, loss=0.2257, **comp_len=6143 (worst case exercised)**.
- Epoch 1 done, `epoch_1` ckpt saved (411 tensors). `expandable_segments` is REQUIRED for this stack — treat as a permanent launch default.

### Run 21 (launched 2026-08-04, 10 epochs, in progress)
- Full train set `data/maas_raft_v3.1/train_sdft.jsonl` (399 samples → 12 steps/epoch → 120 steps), `SAVE_EVERY=12` (= every epoch) + epoch-end ckpts.
- Env: run 20 base + `MAX_TOTAL_LEN=8192 GEN_MAX_NEW_TOKENS=6144 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `OUTPUT_DIR=/mnt/nvme5n1/rohan_patched_ckpts/sdft_gptoss_20b_run21`; WANDB run 21 (amortize-maas).
- Expected ~8-12h (run 20 was 3h40m at 2048-gen; 6144-gen steps measured 200-280s).

## Known Issues / Landmines (tracked, NOT yet fixed)

Consolidated list of latent issues that will break the trainer under dataset/config changes. Fix later.

| # | Issue | Where | Bites when |
|---|-------|-------|------------|
| 1 | **vLLM prompt not pre-truncated**: collator passes full prompt text to vLLM gen; truncation to `STUDENT_MAX_PROMPT_LEN` only happens in trainer forwards (trainer.py:153, 403). vLLM rejects prompt > `max-model-len − max_tokens` (8192−6144=2048). | collator.py / rag_env.py | Dataset with prompts > 2048 tok |
| 2 | **Logprob server has no length guard**: bounded only by trainer-side caps; a too-long request OOMs it silently. | logprob_server.py | Any change in teacher-side caps |
| 3 | **Dataset field contract unvalidated**: collator needs `prompt`/`user_response`/`enriched_user_response` (or `HINDSIGHT_FIELD` override); wrong schema = KeyError 8h into a run. | collator.py:83-115 | Switching datasets |
| 4 | **Dataset size % GRAD_ACCUM_STEPS silently dropped** (`drop_last=True`, steps_per_epoch = len//accum). No warning. | trainer.py:246 | Small/new datasets |
| 5 | **No `MAX_STEPS` support** — 2-step smoke on real data requires a code patch. | trainer.py:295-302 | Smoke on non-64-sample data |
| 6 | **Node-specific paths hardcoded**: HF cache (`/mnt/nvme5n1/rohan_patched_ckpts/hf-cache`), data (`/workspace/data/maas_raft_v3.1/`, `/home/lab/rawhad/...`). | train_full.sh, config.py, AGENTS.md | Running on another node |
| 7 | **`cp_test.py` lives only in /tmp** — CP experiment not reproducible from repo. | — | Revisiting CP work |
| 8 | **Recompute ON/OFF A/B never run** — recompute proven active by inference (config_identity=True + fits), not by direct comparison. | — | If long-seq headroom ever questioned |
| 9 | **No config audit dump per run** — effective limits only recoverable from logs/env, not a file in OUTPUT_DIR. | trainer.py | Comparing runs |
| 10 | **`recompute_granularity` on provider is a no-op** — must be set on `model.config` (already the case; comment in model_utils.py). Trap for future edits. | model_utils.py:118-120 | Refactors touching load_model |
| 11 | **Per-request multi-GB python allocations stall loopback throughput**: zero-fill runs under the GIL and starves the peer's send loop (measured: 10.7 GB/s -> 1.4 GB/s). Affects logprob_client.py:67 (`np.frombuffer(...).copy()`). Preallocate + reuse buffers for big transfers. | logprob_client.py | Any large-response HTTP client |

## 2026-08-04 - Teacher logprobs bottleneck: HTTP stack, not compute (TCP queue fix validated)

### Symptom
- Run 21 (10 epochs) stopped early: `teacher=135-153s` of every ~330s step (~43%). TIMING breakdown (steps 3-7): gen ~120s, teacher ~135-153s, student ~40-60s, loss_bwd ~13s, optim ~2-3s, wsync ~9s.
- Teacher path: 32 requests/step (4 DP ranks x 8 microsteps), each response = (C,V) fp16, up to 6144x201088x2B = 2.47 GB.

### Investigation timeline (hypotheses tested and killed)
1. **Blamed server compute** (32 serial batch=1 forwards). Challenged: student fwd+bwd is ~5s/microstep on 4-rank FSDP.
2. **Measured single-GPU forward** (fwd_bench.py, 8192 tok, gpt-oss-20b): **0.19s**, peak 58.6 GB, fp32 log_softmax over V=201088 ~0.00s. → compute is NOT the bottleneck.
3. **Benchmarked transports** at real payload (hbench.py, 2.47 GB response):
   - Full HTTP stack (uvicorn+starlette+requests): **0.35-0.5 GB/s** → 6.57s per response
   - Raw loopback TCP: **9.4-10.7 GB/s** → 0.24s per response (20-25x faster)
   - Client `np.frombuffer(resp.content).copy()`: **1.38s** (1.8 GB/s)
4. **Per-request production budget**: compute 0.19s (2%) + HTTP transfer ~5s (75%) + numpy parse 1.4s (18%). The python HTTP stack is the bottleneck, not the forward and not the network.
5. **Git history**: the batch endpoint existed (`8d2659c`) and was reverted (`1b55cae`, "slower due to padding overhead ~15s vs ~10s"). Revert was right (batching didn't help) but the diagnosis was wrong — batching changes compute, not transport.
6. **NCCL transport considered and rejected**: needs 4 comm channels or rank-tagged protocol (vLLM deps); saves transport but the TCP route is 10x less complexity.

### TCP+queue experiments (validated the fix)
Play server: `job queue -> GPU worker (compute) -> chunked send`. Client: 4 threads x 8 sequential requests (mirrors 4 DP ranks x 8 microsteps), worst-case C=6144.

| Variant | Wall (32 x 2.47 GB) | Effective |
|---|---|---|
| HTTP baseline (matches production) | 226.2s | 0.35 GB/s |
| TCP queue, serial worker compute+send | 52.4s | 1.51 GB/s (4.3x) |
| TCP queue, per-request sender threads | 42.1s | 1.88 GB/s (5.4x) |
| TCP queue, send-pool sweep (1/4/8 x 1/4/16MB) | 41-47s | ~1.8 GB/s — pool/chunk size DON'T matter |
| TCP queue, mixed lengths (real step shape) | 22.1s | 1.90 GB/s (6.5-10x) |

### Root cause of residual slowness (why not 10 GB/s)
- `iso_compare.py` (client recv's 1MB chunks, no big allocation): **10.7 GB/s** on py3.9, py3.12, and container — identical in all.
- `iso_send.py` (client allocates `bytearray(2.47GB)` per request): **1.4-1.9 GB/s**.
- **Per-request multi-GB zero-fill (calloc) runs under the GIL and stalls the peer's send loop** — kills the pipeline. This is why play runs capped at ~1.9 GB/s regardless of threads/queues.
- Preallocate+reuse recv buffer: recovered to **3.4-3.5 GB/s**; `recv(1MB)`+discard was fastest (10.7).
- Note: production client `np.frombuffer(resp.content).copy()` (logprob_client.py:67) has the same per-request multi-GB alloc pattern — 1.38s/req measured.
- Also noted: python interpreter version does NOT matter (3.9 host == 3.12 container == ~10.7 GB/s in the clean test).

### Hardware ceiling (final numbers, recv_variants.py + memcpy bench)
Single-threaded memory bandwidth on this box is the real ceiling, NOT the socket:
- `memcpy(2.47GB)`: **1.56 GB/s** (1.59s)
- `bytearray(2.47GB)` zero-fill: **2.66 GB/s** (0.93s) — the per-request poison in play/prod clients
- recv into preallocated cold 2.47GB buffer (recv_into full-len, or recv+slice-assign): **~3.4-3.5 GB/s** (0.70s/req)
- recv 1MB chunks + discard: **10.7 GB/s** (1MB temp stays cache-hot, never touches cold DRAM)
- → realistic TCP transfer target: **~0.7s per 2.47GB response** (vs 6.6s HTTP). 10 GB/s is unreachable for transfers that must land in a big buffer.
- Optional further wins (not planned): parallel recv threads into disjoint buffer regions (memory BW scales with threads), recv directly into a pinned tensor for fast async H2D.

### Decision (validated, pending implementation)
- **TCP + queue replaces HTTP `/logprobs`**; FastAPI stays for `/health` + weight-sync handshake (unchanged NCCL weight sync).
- Protocol: length-prefixed binary, keepalive per rank: `[int32 prompt_len][int32 seq_len][int32 ids]` → `[int32 C][fp16 C*V]`.
- Server: bounded job queue + single GPU worker (serial compute + chunked send); 4 connections = 4 ranks.
- Client: one persistent socket per rank; **preallocated recv buffer reused across microsteps** (no per-request multi-GB alloc), `torch.frombuffer` direct to GPU.
- Expected: teacher 140s -> **~25-35s** (server: 32 x (0.19 compute + 0.70 send) serial ≈ 28s; compute hides under sends); step 330s -> ~220-240s; run 21 ~11h -> ~7.5-8h. Plus client parse 1.4s/req -> ~0.

### Artifacts
`/tmp/opencode/`: `hbench.py` (transport rates), `fwd_bench.py` (forward timing), `play_tcp_server.py`/`play_tcp_server_v3.py` (queue server), `play_tcp_client.py` (concurrent client), `iso_send.py`/`iso_compare.py` (isolation), `recv_variants.py`, `bench_out.txt`.

## 2026-08-04 - TCP logprob path IMPLEMENTED + smoke PASSED (`logs/smoke_tcp_v1.log`)

### Implementation (per-connection handler threads + model_lock around compute only)
- `config.py`: `LOGPROB_TCP_PORT` (default 8011). HTTP 8010 stays for `/health` + weight-sync handshake.
- `logprob_server.py`: extracted `compute_logprobs_fp16()` (fp16 cast on GPU before D2H); TCP listener thread + one handler thread per connection; `model_lock` wraps compute only — 0.7s sends happen outside the lock so other ranks' forwards overlap. Zero-copy send: `sendall(memoryview(lp.numpy()).cast("B"))`. HTTP `/logprobs` kept as debug fallback.
- `logprob_client.py`: `request_teacher_log_probs_tcp()` — drop-in for the HTTP fn; lazy persistent per-rank socket (blocking, fail-fast on server death); ONE preallocated 2.47GB `bytearray` recv buffer reused across all microsteps; `recv_into` full-len loop; `torch.frombuffer(memoryview(buf)[:nbytes])` zero-copy → single H2D.
- `trainer.py`: import + call-site swap only (line ~405). `train_full.sh`: `LOGPROB_TCP_PORT` env passthrough.

### Smoke results (2 steps, same shape as v2: 64-sample, 4-rank FSDP, 8192-total)
| | v2 (HTTP) | TCP |
|---|---|---|
| step 1 teacher | ~140s | **30.4s** (total 177.5s, comp_len=1628, loss=0.2346) |
| step 2 teacher (worst case comp_len=6144, 32×2.47GB) | ~150s | **53.9s** (total 225.0s vs 277.0s, loss=0.2097) |
- Losses same sane range as v2 (rollouts differ — vLLM not deterministic across restarts). Both ckpts saved (411 tensors), "Training complete."
- Worst-case teacher = 1.68s/req (not the ~0.9s estimate): 4 concurrent 2.47GB sends share the ~3.5 GB/s memory-write ceiling, plus client H2D (~0.3-0.5s pageable, fp16→bf16). Matches the known hardware ceiling — further squeezing = pinned recv buffer + async H2D (not done).
- Run 21 projection: step ~330s → ~225s worst case → 120 steps ≈ **6.5-7.5h** (was ~11h).

## 2026-08-04 - Chunked LM head IMPLEMENTED + smoke PASSED (run-22 OOM fix)

### Problem
Run 22 crashed at step 34: cuDNN workspace cudaMalloc OOM at 74-78 GiB/79.17 GiB during backward (recompute re-forward). Fix: remove the ~13 GB student-side transient at the loss+backward peak.

### Final design (chunked_head.py + trainer.py)
- Hook via MCore `output_processor` (gpt_model.py:690, all-kwargs) runs the LM head ONCE on completion hidden states `hidden[prompt_len-1 : prompt_len+C-1]` → (C, V) bf16 (skips the prompt-prefix rows, saves ~25% head flops).
- `ChunkedRowKL` (autograd.Function): loss math chunked per row-chunk (128 rows) over the full vocab with an **analytic backward** `grad_z = p·(A − K_row)/C` — the exact total derivative (a detached-denominator softmax would leave a spurious `+p` error). Retains only per-row scalars — kills the old (C, V) fp32 log-softmax retention (~4.9 GB at C=6144).
- Parity: CPU tests — loss exact (0.0 diff), gradients match the old autograd path to the bf16 rounding floor (~1.5e-3 rel at V=201088). play.py.

### Smoke results (64-sample, 4-rank FSDP, unsloth/gpt-oss-20b-BF16, logs/smoke_chunked.log)
| | smoke_tcp (old path) | chunked head |
|---|---|---|
| step 1 | loss=0.2346, comp_len=1628, 177.5s | **loss=0.1834, comp_len=1377, 149.2s** |
| step 2 (worst case C=6144) | loss=0.2097, 225.0s | **loss=0.1772, 226.4s, NO OOM** |
| trainer peak mem | 74-78 GiB (crash zone) | **58-66 GiB** |
- Losses in the sane range (rollouts differ per run — vLLM nondeterministic); per-rank epoch avgs 0.13-0.28. Both ckpts saved (411 tensors), "Training complete."

### Gotchas discovered (the hard way — 4 dead-ends)
1. **FSDPColumnParallelLinear ALWAYS returns the gathered full-vocab logits** — `runtime_gather_output=False` is ignored (probe-verified). Memory savings must come from the loss side, not from sharding the head GEMM.
2. **Calling the FSDP-wrapped output layer in a loop deadlocks**: each call issues collectives on the default group; inter-rank drift across 48 sync points = collective mismatch (py-spy: stuck in `linear_with_grad_accumulation_and_async_allreduce`; 100%-GPU NCCL spin). Single call per microstep only.
3. **Cross-rank vocab-shard coupling is impossible with per-rank data**: each rank computes its own data's shard columns; other ranks' shards belong to DIFFERENT rows. Any cross-rank loss reduce (NCCL: silent hang; gloo: loud `collective mismatch`) mixes rows. The loss must be full-vocab per rank, computed locally. (The probe validated shard-loss math only when ALL ranks process IDENTICAL data — never true in the trainer.)
4. **Local-max mixing**: a chunked logsumexp whose terms use different per-rank maxes is not shift-invariant (`Σ_r Σ_local exp(z−m_r)` ≠ global) — needs a global max reduce first. Caught by a 4-shard CPU simulation (0.84 loss error). Only bites when shard maxes differ (identical-data tests miss it).
- bf16 inputs: autograd bf16-rounds grads at `.float()` cast boundaries — the analytic fp32 backward is strictly MORE accurate; parity tolerances must use the reference's own rounding floor (~1e-2 rel max).

### Files
- `megatron_trainer/chunked_head.py` (new): ChunkedRowKL + make_kl_processor (ROWS chunked 128, KL_CHUNK 2048)
- `megatron_trainer/trainer.py`: forward_student + compute_kl replaced by the hook; rest unchanged
- `play.py`: parity suite (4 cases, incl. production V=201088)

### Next
- Launch the real run (run-21/22 base) with the chunked head — expected to complete without the cuDNN OOM; verify loss curve matches prior runs (~0.19-0.27).

## 2026-08-04 - Loss 0.18-vs-0.23 investigation: code exonerated, draws explain it

### Trigger
Chunked-head smokes landed at step-1 loss ~0.17-0.18 while four old-path runs clustered at 0.23-0.27 — user rightly challenged the "rollout nondeterminism" hand-wave.

### Historical step-1 losses (all same data, same base checkpoint)
| run | step1 loss | step2 loss |
|---|---|---|
| smoke 8192fix v2 (old path) | 0.2483 (C=1954) | 0.2257 |
| smoke_tcp (old path) | 0.2346 (C=1628) | 0.2097 |
| run 21 (old path) | 0.2505 (C=2485) | — |
| run 22 (old path) | 0.2664 (C=1356) | 0.1999 |
| smoke_chunked (new) | 0.1834 (C=1377) | 0.1772 |
| smoke_chunked2 (new) | 0.1680 (C=1700) | 0.1079 |
| smoke_dual (new, debug log) | 0.1717 (C=1998) | 0.1338 |

Two tight clusters 0.05-0.08 apart — sampling-luck alone strained. Evidence chain:

1. **Replay** (`/tmp/opencode/replay_loss.py`, 1 GPU, real items from smoke_sdft.jsonl): old full-forward path vs the new hook path on identical tensors → loss diff **~1e-6** across items. Same-GPU; the smokes always ran the same GPU layout (vLLM 0-1, trainers 2-5, logprob 6).
2. **Per-item loss sensitivity** (16 items, fixed completion): losses 0.98-2.65, **std 0.56** — an item's loss swings ±0.5 with the completion text.
3. **Live dual-loss log** (temporary `[DUAL]` debug in chunked_head.py, smoke_dual run): for every real sampled microstep (C=552..6144), `loss_new` vs old compute_kl math on the same z/teacher → **diff ≤ 1.2e-7** (most 0.0). The two loss computations are numerically identical on the actual data.
4. **Per-item spread inside one real step**: 0.026 (C=6144) to 0.286 (C=755) — 10x spread, length-correlated (longer completions → lower per-token KL). The step-1 mean over 32 such items is very draw-sensitive.

### Conclusion
- Loss math: old and new identical to 1e-7 on the same completions (live-verified).
- The 0.17-0.18 vs 0.23-0.27 cluster shift = which 32 items land in step 1 (unseeded DataLoader shuffle) × which completions vLLM samples (temp 0.7, unseeded) — NOT the code, NOT GPUs, NOT fp errors (same GPU layout; fp noise ~1e-7).
- Correctness of the chunked head stands on: parity suite (play.py), live dual-loss (1e-7), probe A/B (loss + grads), 3 clean smokes incl. C=6144 worst case at 58-66 GiB peak.

### Pending
- Remove the temporary `[DUAL]` debug block from `megatron_trainer/chunked_head.py` before the real run.
- Launch the real 10-epoch run (run-21/22 base); watch the loss curve vs historical (~0.19-0.27 decaying) and survival past step 34 (run 22's crash point).

---

## 2026-08-05 — Frozen 120b teacher (TEACHER_MODEL_PATH + mxfp4)

### Goal
Separate teacher: `TEACHER_MODEL_PATH` env var → logprob server loads a different
(frozen) model than the student. Target: `openai/gpt-oss-120b` native mxfp4 4-bit
(~62 GB download, fits ONE 80GB GPU — verified: 63.7 GB weights on GPU).

### mxfp4 stack (verified in NeMo 26.06 container)
- transformers 5.8.1 has `Mxfp4Config` quantizer; auto-detected via `quant_method: mxfp4` in config.json.
- Requires `kernels` pip package (kernels-community): **`pip install kernels==0.14.1`** — transformers 5.8.1's
  `hub_kernels` integration crashes on import with kernels ≥0.15 (LayerRepository now REQUIRES revision/version;
  transformers calls it without). 0.14.1 is the latest import-compatible.
- Without `kernels` installed, transformers silently DEQUANTIZES mxfp4 → bf16 (226 GB — OOM).
- torchao 0.17 + triton 3.6 already in container; H100 CC 9.0 ✓. Kernels compile into `~/.triton` (JIT, ~10 min cold).
- **Persistent triton cache**: `/mnt/nvme5n1/rohan_patched_ckpts/triton_cache` mounted at `/root/.triton`
  (train_full.sh + smoke containers) → first forward fast after the first run.

### Memory: 8192-token forward OOM'd at 83.1 GB peak (79.17 usable) — three fixes
1. **Chunked lm log_softmax** (compute_logprobs_fp16): rows in 1024-chunks → fp32 softmax transient 4.9→0.8 GiB.
2. **Chunked exact attention** (monkey-patched `eager_attention_forward` in external-teacher mode only):
   gpt-oss eager attention materializes (H=64, S, S) bf16 scores = 8.6 GiB at S=8192. Blockwise two-pass
   row-max trick (chunks of 1024 query rows; fp32 exp-sums; sink column handled exactly). Verified vs the
   original full-row math: my offline reproduction of the original is BIT-EXACT; chunked differs only by
   bf16-vs-fp32 softmax arithmetic (mean |Δlogprob| ~0.015 — chunked is the MORE accurate one).
   Gotchas hit: GQA repeat_kv needed; value needs NO transpose; `torch.maximum` broadcasts → block maxima
   must be collected+cat'd, not maxed incrementally.
3. `_model_logits` normalization: HF returns (1,S,V) ModelOutput (batch dim) vs bridge's (S,V) tuple —
   squeeze(0) at the single-request call site; batch endpoint keeps 3-D.
4. `.contiguous()` on the fp16 result (mxfp4 kernel outputs can be strided → memoryview.cast("B") fails).

**Result: 8192-token 120b forward = 74.2 GiB peak, 6.5 s** (full 2048+6144 teacher request; bitwise deterministic).

### Code changes
- config.py: `TEACHER_MODEL_PATH` (default "" = current same-model teacher + EMA sync).
- logprob_server.py: external-teacher mode = plain HF `AutoModelForCausalLM.from_pretrained(bf16)`
  (mxfp4 auto), skips `init_distributed_standalone` + `/init_weight_sync` + `/sync_weights` (both return "disabled").
- trainer.py: `init_logprob_weight_engine` + `sync_weights_to_logprob_server` gated on `not TEACHER_MODEL_PATH`.
- train_full.sh: `TEACHER_MODEL_PATH` passthrough + `kernels==0.14.1` install + triton cache mount.
- Teacher stays on the existing LOGPROB_GPU slot (GPU 6) — NO layout change. GPU 7 remains free.

### Smoke evidence (GPU 7, standalone container)
- Model load ~40 s (cache-warm), weights 63.7 GiB.
- seq=64/256/1024: 3.5 s / 1.0 s / 0.4 s (warm).
- 2048+6144: **6.5 s**, peak 74.2 GiB, logprobs finite, mean ≈ -16.6, repeat → bitwise identical.
- `openai/gpt-oss-120b` fully cached at `/mnt/nvme5n1/rohan_patched_ckpts/hf-cache` (~62 GB, mxfp4 shards only).

### Notes / gotchas
- gpt-oss-120b: 36 layers, 128 experts, **64 attention heads**, 8 KV heads (GQA), head_dim 64, same 201088 vocab/tokenizer as 20b.
- HF keeps layernorms + RoPE fp32 by design; attention ends up bf16 anyway (sink cat is bf16 — no fp32 upcast at H=64).
- Sliding-window layers exist; full attention is used (attention_mask=None, matches the old 20b teacher path).
- Teacher latency 6.5 s/request vs 20b's ~0.4 s → per-step time will rise; LOGPROB_BATCH_SIZE batching is the lever if needed.

## 2026-08-06 - Collator: Qwen3-8B path + new-format OLS tool-trajectory support

### Goal
- Render the new OpenAI-style OLS tool-trajectory dataset (data/ols/train_sdft_mini.jsonl) for on-policy SDFT, with Qwen3-8B as the default model. Spec lives in `docs/megatron_trainer/collator.md`.

### Key findings (validated earlier, recorded in spec)
- gpt-oss templates hardcode `tool_calls[0]` (drop calls 2..N); Qwen3 renders all N `<tool_call>` blocks inline.
- Raw `{'role':'tool','tool_results':[...]}` renders an EMPTY `<tool_response>` on Qwen3 (silent loss) → normalize to `content=json.dumps(tool_results)`.
- Full OLS tool defs = 7,214 tok/prompt (17/31 > 14,336); stripped (descriptions dropped) = 2,332 tok → 1/31 prompt, 2/31 conditional > 14,336, 0 > 16,384.
- Dataset shape gotchas: `tool_calls` = `{name, arguments(dict)}` (NOT OpenAI function format); `user_response` has no `content` in 10/31 items and no `value` ever → old golden-answer code would crash; last prompt message is tool (21), user (5), or assistant (5).

### Code changes
- collator.py: `_normalize_messages` tool branch; `_target_text` (hand-format `name(args_json)` + content, handles value/content/None); `_append_hint` (merge into last user msg / append fresh user turn); `_load_tool_defs` (dirname(TRAIN_DATA_PATH)/tool_defs.json, strip descriptions, cached in `TOOL_DEFS`); family guard (`IS_QWEN`, ValueError for non-Qwen + tools/tool_calls/tool_results); `raw_questions` → last user message; `tools=` kwarg on both renders.
- config.py: `IS_QWEN` flag.
- rag_env.py: comment only — `_build_privileged_prompt_from_feedback` assumes last msg is user; OLS last msg is tool → hint would land inside `<tool_response>`; fix = reuse `_append_hint` when online_feedback + OLS runs.

### Verification (play.py, tokenizer only, no training)
- 31/31 render, 0 errors; prompt toks 2,382–14,727 (median 9,541, 1>14,336); conditional 2,490–16,025 (median 9,632, 2>14,336); 0 > 16,384.
- 88/88 tool-call lines in targets; hint placement: 5 merged, 26 appended (clean `<|im_start|>user` turn after tool results).
- Family guard: fires for gpt-oss + OLS shape; not for gpt-oss + old format; old-format enriched regression passes.
- HINDSIGHT_FIELD=user_response needed for OLS runs (no enriched field in that dataset).

## 2026-08-06 - Collator truncation (budget enforcement) + vLLM context fix

### Why
- vLLM launched with --max-model-len 8192 hard-rejects OLS prompts (400: 7,169 input + 1,024 gen = 8,193 > 8,192).
- Trainer-side tokenizer tail-chops oversize prompts silently — which cuts exactly the privileged hint.
- Collator now enforces budgets at render time (message-level), per spec update (docs/megatron_trainer/collator.md).

### Code changes
- collator.py: `_render_tokens` (renders + counts with add_special_tokens=False, matching trainer.py:328); `_truncate_to_budget(messages, tokenizer, budget, protect_last, render_kwargs)` — greedy drop of earliest messages until rendered tokens fit; system (index 0) never dropped; question (last user msg) dropped only after all other messages; protect_last shields the hint; `_assert_protected_fits` (ValueError on config violation); asserts: init (system+tools <= STUDENT_MAX_PROMPT_LEN) + per-example (system <= student budget; system+hint <= TEACHER_MAX_PROMPT_LEN); normalized_messages/raw_questions now from the truncated student trajectory; both paths truncate (no-op within budget).
- train_full.sh:142: --max-model-len 8192 -> 16384 (14,336 + 1,024 = 15,360 < 16,384).

### Verification (play.py, tokenizer only)
- 0/31 prompts > 14,336 after truncation (1 item dropped 2 tool-turn messages: 14,727 -> 13,887); 0/31 conditionals > 15,360 (max 14,966); system 31/31 complete; hint 31/31 complete; questions 0/31 dropped.
- Old-format regression unchanged; family guard unchanged; init + per-example asserts fire with clear ValueError messages.

### Gotcha
- Drop order matters: for OLS the question is message index 1 — naive "drop from front" killed the question first (StopIteration on raw_questions). Fixed: oldest turns first, question last-droppable, hint never.
- Init assert uses STUDENT_MAX_PROMPT_LEN: default 2048 fails on OLS (system+tools = 2,306) — correct; runs must pass STUDENT_MAX_PROMPT_LEN=14336.

## 2026-08-07 - Collator: budget violations drop examples instead of raising

### Why
- The per-example protected-set assert crashed a legacy gpt-oss/analyze_research run (run_5): enriched hint render 2,171 > default TEACHER_MAX_PROMPT_LEN 2048, ValueError at step 0, torchrun SIGTERM'd all ranks.
- Decision: budget violations must NOT raise — drop the offending example at dataset load (never trained on) and log a warning. Only a fully-dropped dataset raises.

### Code changes
- collator.py: deleted `_assert_protected_fits`; added `_hint_for(ex)` (hint build factored out, None for online_feedback), `_drop_reason(ex)` (protected-set renders vs STUDENT_MAX_PROMPT_LEN / TEACHER_MAX_PROMPT_LEN), `filter_dataset(dataset, rank)` (one-time load pass, `dataset.select(valid_indices)`, per-drop + summary warnings on rank 0, raises if all dropped); `__post_init__` system+tools check is now a warning (no crash); `__call__` per-example asserts removed, uses `_hint_for`; loguru import.
- trainer.py: `dataset = collator.filter_dataset(dataset, rank=rank)` inserted after collator construction, before DataLoader — steps_per_epoch auto-adjusts.
- Also fixed latent bug: `_target_text` crashed on explicit `tool_calls: null` (HF datasets normalize missing keys to None); now `user_response.get("tool_calls") or []`.

### Verification (play.py, 15 checks)
- OLS at 14,336/15,360: 0/31 dropped by filter; truncation/completeness/hint-placement unchanged.
- Init budget check warns (no raise); empty filtered dataset raises ValueError; partial drop: 2 over-budget hints dropped (warned), 1 kept; family guard unchanged.

### Note
- The analyze_research run_5 crash is now resolved by the drop filter: over-budget hints are skipped with warnings instead of killing the run. If TEACHER_MAX_PROMPT_LEN=2048 drops too many, raise the budget.

## 2026-08-10 - Importance sampling weighting (vLLM rollout vs training policy)

### Why
- vLLM is the rollout engine; its proposal distribution can drift from the training policy (weight staleness, future T!=1). Reverse-KL gradients estimated from rollout samples are biased unless corrected. Added TIS weighting, same scheme as TRL DistilTrainer (idan-Self-Distillation).

### Math (chunked_head.py)
- Per-token ratio r_t = exp(policy_logp - rollout_logp), clamped at IS_CAP (default 2.0); per-sequence weight = masked mean of r_t over valid tokens; loss *= weight (detached). policy_logp already existed (ChunkedRowKL); rollout_logp is new, captured from vLLM at generation time.

### Code changes
- config.py: IS_WEIGHTING (default on), IS_CAP (default 2.0)
- vllm_utils.py: vllm_generate requests logprobs=1, returns (text, finish_reason, token_logprobs) — 1:1 aligned with output tokens
- env/rag_env.py: stashes completion_log_probs (single call)
- env/api_adapter_env.py: per-segment log-probs; inserted force-close tokens (.\n</think>\n\n) and template artifacts (<|im_end|>) are None (masked)
- trainer.py: rollout_data carries completion_log_probs; tensor built with NaN for masked; length-aligned with warn; passed to make_kl_processor; wandb config gains knobs
- chunked_head.py: compute_is_weight() + loss rescale + is/* metrics (ratio min/mean/max, logp_diff_mean, clip_rate)
- start_vllm_patched.py + smoke_test.sh: vLLM launched with --logprobs-mode processed_logprobs (post-temperature logp — the correct log q for IS; raw mode only equals it at T=1.0)
- train_full.sh: IS_WEIGHTING/IS_CAP passed into container

### Validation (Qwen3-0.6B, H100, vLLM 0.23)
- logprobs=1: token_logprobs aligned 1:1 with generated tokens; matches independent HF forward at T=1.0 in raw mode (mean |diff| 0.02)
- processed mode at T=0.7: returns log_softmax(z/T) (max diff 0.116 vs computed) — correct for IS at T!=1
- CAVEAT: processed mode at T=1.0 inflates logprobs on low-prob tokens vs HF (up to ~0.97 nats; clean completions <= ~0.12). On-policy is/ratio_mean will read < 1 (~0.3-0.9), NOT 1.0 — that's the accepted processed-mode quirk, not drift.
- Gotcha found: vLLM 0.23 gumbel sampling uses a deterministic hash PRNG (tl.rand); ~2-5% of seeds produce degenerate garbage completions ('_______, with___ _ _ _'). Orthogonal to IS; candidates: per-request seeds, use_fp64_gumbel.

## 2026-08-11 - online_feedback: reflection-model feedback as privileged info (first e2e run)

### Why
- `HINDSIGHT_FIELD=online_feedback` was wired in code (collator → RagEnv → reflector) but never run end-to-end. Made it work: infra fixes + richer feedback + golden-chunk support.

### Changes
- train_full.sh: install `anthropic[vertex]` (was missing → import crash on any rag run); pass `REFLECTOR_MODEL/REGION/PROJECT_ID` via `-e` (project id was never reaching the container).
- reflector.py: detailed feedback prompt (~150-250 words: what's right, concrete errors, actionable guidance; max_tokens 1024→2048); bare-JSON parse fallback (was `IndexError` if no ```json fence).
- collator.py: `golden_chunks` batch field (enriched_user_response value/content, '' when absent); `_drop_reason` drops examples with EMPTY golden answer in online_feedback mode ("empty golden answer (user_response) required").
- trainer.py: passes `golden_chunk` into RagEnv.
- env/rag_env.py: privileged prompt = golden chunk (optional) + golden answer (required) + reflection feedback (`ONLINE_FEEDBACK_TEMPLATE` / `NO_CHUNK` variant).
- Docs: ragenv.md (3-part template, detailed feedback), launch_trainer.md (REFLECTOR_* rows, install line, online_feedback contract).

### Validation (Qwen3-0.6B, maas_raft smoke 64, H100)
- Tokenizer-only: 3-part prompt with chunk (951 tok) / 2-part without (393 tok); empty-golden dropped in online_feedback mode but kept in static mode (regression); batch carries golden_chunks.
- Container e2e (3 GPUs, train_full.sh layout): 2 optimizer steps completed, exit 0 — TIMING step=1 total=45.5s (gen=30.1s = 32 concurrent reflector calls), step=2 total=46.1s; vLLM + logprob NCCL groups initialized; no deadlock.
- Live reflector call: FAIL verdict, 209 words, structured multi-paragraph critique vs golden.

### Gotchas discovered
- **Host-vLLM ↔ container-trainer NCCL weight-engine init fails** (500 / hang / "NCCL error: invalid usage") — cross-process-group NCCL with different NCCL builds. ALL processes must live in ONE NeMo container (train_full.sh pattern: vLLM dev 0, trainer dev N, logprob last dev).
- **trainer.log file sink dies after model load** (bridge reconfigures loguru) — stdout is authoritative; debug lines (Reflector: ...) only in file sink pre-bridge. Pre-existing, not from this change.
- ADC project `cloudability-it-gemini` lacks `aiplatform.endpoints.predict` for Anthropic publisher models; **must use `REFLECTOR_PROJECT_ID=itpc-gcp-ai-eng-claude`** (matches ANTHROPIC_VERTEX_PROJECT_ID).
- Empty `REFLECTOR_PROJECT_ID` → `ValueError: Could not resolve project_id` inside container (host venv tolerated it via ADC default).
- OLS tool-format still needs `_append_hint` reuse in rag_env (`_build_privileged_prompt_from_feedback` assumes last msg is user) — for OLS runs only; maas_raft old format unaffected.

### Remaining
- OLS runs: fix last-message-is-tool hint placement; OLS 10/31 empty-golden examples now auto-dropped by filter (may need TEACHER_MAX_PROMPT_LEN bump).
