# Goal: Migrate SDFT Training to Megatron Bridge

## Objective

Reimplement the SDFT training loop (currently in `train_dir/src/`) using
NVIDIA Megatron Bridge instead of HuggingFace transformers primitives.
All new development happens in `megatron_trainer/`.

## SDFT Algorithm Summary

Paper: "Self-Distillation Enables Continual Learning" (Shenfeld et al., 2026, arXiv:2601.19897).

The same model plays two roles per training sample:
- **Student** `P = pi_theta(.|x)`: sees only the query
- **Teacher** `Q = pi_phi(.|x, c)`: sees query + privileged demonstration/context

Training loop per sample:
1. Sample on-policy completion `y ~ P` (via vLLM)
2. Compute teacher log-probs `Q(y_t | y_<t, x, c)` at each token of `y`
3. Compute student logits `P(v | y_<t, x)` over full vocab at each position
4. Loss = reverse KL: `KL(P || Q) = sum_t sum_v P(v) * (log P(v) - log Q(v))`
   (full analytic per-token estimator, not token-level partial)
5. Backprop through student only; teacher is fixed for this step
6. After optimizer step: EMA update teacher weights `phi = alpha*theta + (1-alpha)*phi`

Key properties:
- On-policy (student generates its own rollouts) -> reduces catastrophic forgetting
- Teacher stays close to base model (KL ~0.68 nats vs SFT's ~1.26 nats)
- Single trajectory per prompt is sufficient (no group sampling like GRPO)

## Reference Implementation

The existing working implementation lives in `train_dir/src/`:
- `trainer.py` -- custom training loop (NOT HF Trainer), entry point
- `logprob_server.py` -- teacher model process, NCCL rank 1
- `collator.py` -- SDFTCollator, privileged info injection
- `nccl_comm.py` -- NCCL protocol (weight sync, logprob transfer)
- `vllm_utils.py` -- vLLM HTTP + NCCL weight transfer
- `config.py` -- all hyperparameters (env-var overridable)

Read these files thoroughly before starting. The algorithm, data flow,
and 3-GPU architecture must be preserved exactly.

## Megatron Bridge Reference

Two working reference projects exist on this node:

### CPT project (`~/rawhad/ols-cpt/`)
- `cpt_lora_qwen3.py` -- CPT with LoRA on Qwen3-8B using Megatron Bridge
- `cpt_lora_qwen35.py` -- CPT with LoRA on Qwen3.5-2B (more detailed config)
- `cpt_lora_nemotron.py` -- CPT with LoRA on Nemotron-3-Nano-30B-A3B
- `megatron-bridge/` -- cloned Megatron Bridge repo with source code
- `megatron-bridge/src/megatron/bridge/recipes/qwen/qwen3.py` -- Qwen3 recipe configs

### CPT with full docs (`~/rawhad/ols-troubleshooting-eval/ra/cpt/`)
- `README.md` -- end-to-end CPT pipeline docs (data, container setup, training, export)
- `TRAINING.md` -- detailed Nemotron CPT walkthrough with podman commands
- `cpt_lora_qwen3_8b.py` -- Qwen3-8B training script
- `export_adapter.py` -- LoRA adapter export to HF PEFT format
- `merge_adapter.py` -- merge LoRA adapter into full model

**Read `~/rawhad/ols-troubleshooting-eval/ra/cpt/README.md` first** -- it has the
complete podman workflow, troubleshooting, and all the gotchas we hit.

### Key Megatron Bridge patterns
```python
from megatron.bridge import AutoBridge
from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.peft.lora import LoRA

config = qwen3_8b_pretrain_config()
config.checkpoint.pretrained_checkpoint = "/data/checkpoints/qwen3_8b_megatron"
config.model.tensor_model_parallel_size = 1
config.model.pipeline_model_parallel_size = 1
finetune(config=config, forward_step_func=forward_step)
```

Available Qwen3 configs: `qwen3_8b_pretrain_config()`, `qwen3_8b_sft_config()`, `qwen3_8b_peft_config()`.

### Container requirement (cannot use native install)

Megatron Bridge must run inside the **NeMo container** (`nvcr.io/nvidia/nemo:26.06`).
Native install via `uv sync` fails on this node (CentOS Stream 9, glibc 2.34)
because `nvidia-resiliency-ext` requires glibc 2.39.

Podman run pattern (CDI, not `--gpus`):
```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -e RAYON_NUM_THREADS=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -v ~/rawhad/ols-cpt:/data:z \
  -w /data \
  nvcr.io/nvidia/nemo:26.06 \
  torchrun --nproc_per_node=8 cpt_lora_qwen3.py
```

Critical podman flags:
- `--device nvidia.com/gpu=all` -- NVIDIA CDI (not `--gpus`, which requires nvidia-docker)
- `--ipc=host` -- shared memory for NCCL (mutually exclusive with `--shm-size`)
- `-e RAYON_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false` -- prevents rayon thread pool panic when multiple torchrun workers init HF tokenizer simultaneously
- `-v ...:/data:z` -- `:z` for SELinux relabeling on CentOS
- `-e CUDA_DEVICE_MAX_CONNECTIONS=1` -- required by Megatron for compute/comm overlap

### Checkpoint conversion

HF → Megatron (must run inside container):
```bash
podman run --rm --device nvidia.com/gpu=0 \
  -v ~/rawhad/ols-cpt:/data -w /opt/Megatron-Bridge \
  nvcr.io/nvidia/nemo:26.06 \
  python examples/conversion/convert_checkpoints.py import \
    --hf-model /data/qwen3-8b-hf \
    --megatron-path /data/checkpoints/qwen3_8b_megatron
```

Megatron → HF (full model only, NOT LoRA):
```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model /data/qwen3-8b-hf \
  --megatron-path /data/checkpoints/... \
  --hf-path /data/output-hf
```

LoRA export uses `AutoBridge.export_adapter_ckpt()` -- see
`~/rawhad/ols-troubleshooting-eval/ra/cpt/export_adapter.py`.

## Architecture (must preserve)

- GPU 0: vLLM server (on-policy rollouts, receives full weight replacement)
- GPU 1: Student/Trainer (trains with reverse KL loss)
- GPU 2: Teacher/Logprob server (EMA-updated, provides log-probs)
- Two independent NCCL groups:
  - `torch.distributed` (port 29500): trainer <-> logprob server
  - vLLM `NCCLWeightTransferEngine` (auto port): trainer <-> vLLM

### Container vs host: open design question

The reference SDFT loop runs entirely on the host (3 separate processes, 2 NCCL
groups, vLLM HTTP server). Megatron Bridge requires the NeMo container. Options:

1. **All-in-container**: Run all 3 processes inside one container. Requires
   vLLM to coexist with the container's torch/NCCL stack. The NeMo container
   has its own torch, which may conflict with vLLM's requirements.
2. **Trainer-only in container**: Only the student/trainer runs in the container.
   vLLM and logprob server run on the host. NCCL must cross the container
   boundary (possible with `--ipc=host` and `--network=host`).
3. **Extract model, train on host**: Use the container only for checkpoint
   conversion (HF ↔ Megatron). Load Megatron-format weights manually on the
   host. May not work if Megatron-Core model definitions are needed at runtime.

This must be resolved in Phase 1. Start with option 1 (simplest) and fall back.

## What changes

- Model loading: `AutoModelForCausalLM` -> `AutoBridge.from_hf_pretrained()`
- Model forward: HF backbone -> Megatron-Core model
- Optimizer: `bitsandbytes.AdamW8bit` -> Megatron-Core optimizer (or keep 8bit if needed)
- Checkpointing: must still export HF-format checkpoints for eval compatibility
  - Full-model: `convert_checkpoints.py export` works
  - LoRA: must use `AutoBridge.export_adapter_ckpt()` then merge with `peft`
- Weight sync: parameter names will differ -- need mapping for NCCL broadcast + vLLM transfer
- Runtime: host python venv -> NeMo container (`nvcr.io/nvidia/nemo:26.06`) via podman

## What stays the same

- SDFTCollator (privileged info injection pattern)
- vLLM integration (HTTP API + NCCL weight transfer)
- NCCL protocol (commands, logprob transfer)
- 3-GPU architecture
- Reverse KL loss computation
- EMA teacher update logic
- On-policy sampling via vLLM
- Training data format + paths
- wandb logging

## Training Data

- Train: `/home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/train_maas_sdft.jsonl`
- Test: `/home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl`

Format: JSONL with fields used by SDFTCollator (`messages`, `enriched_user_response`, `user_response`).

Eval script: `/mnt/nvme0n1/rawhad/self_distillation/aligning_lm_from_user_interaction/scripts/eval_maas_sdft.py` (on node `ai-innovation-h100-01-preserve`).

Usage:
```bash
CUDA_VISIBLE_DEVICES=<gpu> .venv_vllm/bin/python scripts/eval_maas_sdft.py \
  --model <model_path_or_hf_id> \
  --test_jsonl <test_data_path> \
  --output_dir <output_dir>
```

Example -- base model:
```bash
CUDA_VISIBLE_DEVICES=0 .venv_vllm/bin/python scripts/eval_maas_sdft.py \
  --model Qwen/Qwen3-8B \
  --test_jsonl /home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl \
  --output_dir eval_results/base_eval_v2
```

Example -- checkpoint:
```bash
CUDA_VISIBLE_DEVICES=7 .venv_vllm/bin/python scripts/eval_maas_sdft.py \
  --model /home/lab/rawhad/self_distillation/rohans_sdft/train_dir/output_run_14/epoch_10 \
  --test_jsonl /home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl \
  --output_dir eval_results/run_14/epoch_10_eval_v2
```

## Acceptance Criteria

Training must produce models with scores comparable to these (run 14, Qwen3-8B):

| Epoch    | no_context (v2) | with_context (v2) |
|----------|-----------------|-------------------|
| Base     | 3%              | 86%               |
| epoch_4  | 19%             | 80%               |
| epoch_5  | 28%             | 86%               |
| epoch_6  | 25%             | 86%               |
| epoch_7  | 32%             | 86%               |
| epoch_10 | 41%             | 83%               |

Training loss trajectory should be comparable to the reference run.

## Verification Steps

1. Model loads successfully via AutoBridge
2. Forward pass produces valid logits
3. Reverse KL loss computes and backpropagates
4. NCCL weight sync works (both groups)
5. Full training loop runs for 1 optimizer step without crash
6. Checkpoints save in HF format (loadable by `transformers`)
7. Multi-epoch training completes
8. Eval scores match reference table above

## Blocked Stop Conditions

- If Megatron Bridge does not support Qwen3-8B: stop and report
- If NCCL weight sync cannot work with Megatron param format: stop and report
- If vLLM cannot receive Megatron-format weights: stop and report alternatives
- If vLLM cannot coexist with NeMo container's torch/NCCL: try option 2 (trainer-only in container)
- If NCCL cannot cross container boundary: stop and report

## Environment

- Node: `ai-innovation-h100-01-preserve` (8x H100 80GB HBM3)
- GPUs 0-2 (or 3-5 for parallel run)
- Host Python 3.12 via `uv` (for vLLM, logprob server)
- NeMo container `nvcr.io/nvidia/nemo:26.06` via podman (for Megatron Bridge)
- Native `uv sync` of megatron-bridge does NOT work (glibc 2.34 < required 2.39)
- vLLM pinned to 0.23 (v0.25+ pulls torchcodec which needs system FFmpeg)
- Model: `Qwen/Qwen3-8B`
- Megatron Bridge source: `~/rawhad/ols-cpt/megatron-bridge/`
- Reference CPT docs: `~/rawhad/ols-troubleshooting-eval/ra/cpt/README.md`

## Hyperparameters (from reference run 14)

- Learning rate: 5e-5
- Batch size: 1 (effective 32 via grad accum)
- Grad accumulation steps: 32
- Epochs: 10
- Max grad norm: 10.0
- EMA alpha: 0.05
- Gen max new tokens: 2048
- Gen temperature: 0.7
- Gen top_p: 0.95
- Optimizer: 8-bit Adam
- LR schedule: constant

## Progress

Track progress below. Update after each phase.

### Phases
- [x] Phase 1: Container setup -- NeMo 26.06 pulled, GPU access verified. Architecture: trainer+logprob in container, vLLM on host.
- [x] Phase 2: Checkpoint conversion -- NOT NEEDED. AutoBridge converts HF→Megatron in-memory at load time (4.5s).
- [x] Phase 3: Model loading spike -- AutoBridge loads Qwen3-8B (8.19B params, 16.38GB), forward (1,32,151936) logits, backward OK.
- [x] Phase 4: Student forward -- model(input_ids, position_ids, attention_mask=None) returns (B,S,V) logits. gradient_accumulation_fusion=False required.
- [x] Phase 5: Weight sync -- NCCL broadcast works between 2 Megatron models (verified 0 diff). export_hf_weights yields 399 HF tensors from 291 Megatron params.
- [x] Phase 6: Full training loop -- END-TO-END WORKING. All-in-container arch (3 GPUs, 1 container). Loss decreasing (0.82→0.31 by step 26).
- [x] Phase 7: Checkpoint export -- WORKING. Manual safetensors export (avoids distributed barrier deadlock). 16GB model.safetensors loadable by HF transformers.
- [x] Phase 8: Eval -- DONE. no_context: 34% (ref: 41%), with_context: 82% (ref: 83%). Comparable to reference.

### Running the full training
```bash
# Full training (10 epochs, grad_accum=32, ~8 hours on 3x H100)
TMPDIR=/mnt/nvme0n1/podman_tmp podman run --rm \
    --device nvidia.com/gpu=3 --device nvidia.com/gpu=4 --device nvidia.com/gpu=5 \
    --ipc=host --network=host \
    --add-host $(hostname):127.0.0.1 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e RAYON_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false \
    -e MASTER_ADDR=127.0.0.1 -e PYTHONPATH=/workspace \
    -e MODEL_NAME=Qwen/Qwen3-8B -e HF_MODEL_PATH=Qwen/Qwen3-8B \
    -e NCCL_MASTER_PORT=29501 -e VLLM_PORT=8001 \
    -e OUTPUT_DIR=/workspace/output_megatron_run \
    -e NUM_EPOCHS=10 -e GRAD_ACCUM_STEPS=32 \
    -e VLLM_SERVER_DEV_MODE=1 -e BNB_CUDA_VERSION=130 \
    -e WANDB_PROJECT=sdft-online -e WANDB_NAME=sdft-megatron-qwen3-8b \
    -v $(pwd):/workspace:z \
    -v ~/.cache/huggingface:/root/.cache/huggingface:z \
    -v /home/lab/rawhad:/home/lab/rawhad:ro \
    -w /workspace \
    nvcr.io/nvidia/nemo:26.06 \
    bash megatron_trainer/smoke_all_in_container.sh
```

### Eval after training
```bash
CUDA_VISIBLE_DEVICES=7 .venv_vllm/bin/python scripts/eval_maas_sdft.py \
  --model /path/to/output_megatron_run/epoch_10 \
  --test_jsonl /home/lab/rawhad/sdg-ki-eval/data/maas_data/rohans_data/test_maas_sdft.jsonl \
  --output_dir eval_results/megatron_run/epoch_10
```

### Key implementation decisions
- `gradient_accumulation_fusion=False` — avoids main_grad requirement (we use PyTorch optimizer, not Megatron's)
- `wrap_with_ddp=False` — no DDP since we have our own training loop
- `--add-host $(hostname):127.0.0.1` — required in container for NCCL hostname resolution
- `TMPDIR=/mnt/nvme0n1/podman_tmp` — root partition too full for container temp files
- `pip install --no-deps vllm==0.23 bitsandbytes` at container startup
- `BNB_CUDA_VERSION=130` — container CUDA 13.2, highest bnb binary is 13.0
- `--enforce-eager` for vLLM (torch.compile incompatible with container's torch 2.12)
- `start_vllm_patched.py` — monkey-patches prometheus _IncludedRouter crash
- All-in-container (3 GPUs, 1 container) — cross-container NCCL fails (version mismatch 2.30 vs 2.28)

### Files created
- `megatron_trainer/__init__.py` — empty
- `megatron_trainer/config.py` — env-var-overridable config (same as reference + HF_MODEL_PATH)
- `megatron_trainer/collator.py` — identical to reference (SDFTCollator)
- `megatron_trainer/model_utils.py` — AutoBridge init, model loading, HF weight export, checkpoint save
- `megatron_trainer/nccl_comm.py` — NCCL protocol (adapted for MCore GPTModel forward API)
- `megatron_trainer/vllm_utils.py` — vLLM HTTP client + NCCL weight sync (Megatron→HF conversion)
- `megatron_trainer/trainer.py` — main training loop
- `megatron_trainer/logprob_server.py` — teacher model NCCL command loop
- `megatron_trainer/run.sh` — launcher (deprecated, cross-container NCCL doesn't work)
- `megatron_trainer/smoke_all_in_container.sh` — smoke test launcher (1 epoch, grad_accum=2)
- `megatron_trainer/train_full.sh` — PRODUCTION launcher (10 epochs, grad_accum=32)
- `megatron_trainer/start_vllm_patched.py` — vLLM launcher with prometheus fix
- `megatron_trainer/test_checkpoint.sh` — checkpoint export validation test
- `megatron_trainer/test_spike.py` — single-GPU validation (load, forward, backward, export)
- `megatron_trainer/test_distributed.py` — 2-rank validation (NCCL broadcast between Megatron models)
