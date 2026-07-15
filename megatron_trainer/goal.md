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

A working Megatron Bridge project exists at `~/rawhad/ols-cpt/` on node `rh-h100-01`:
- `cpt_lora_qwen3.py` -- CPT with LoRA on Qwen3-8B using Megatron Bridge
- `cpt_lora_qwen35.py` -- CPT with LoRA on Qwen3.5-2B (more detailed config)
- `megatron-bridge/` -- cloned Megatron Bridge repo with source code
- `megatron-bridge/src/megatron/bridge/recipes/qwen/qwen3.py` -- Qwen3 recipe configs

Key Megatron Bridge patterns from the reference:
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

## Architecture (must preserve)

- GPU 0: vLLM server (on-policy rollouts, receives full weight replacement)
- GPU 1: Student/Trainer (trains with reverse KL loss)
- GPU 2: Teacher/Logprob server (EMA-updated, provides log-probs)
- Two independent NCCL groups:
  - `torch.distributed` (port 29500): trainer <-> logprob server
  - vLLM `NCCLWeightTransferEngine` (auto port): trainer <-> vLLM

## What changes

- Model loading: `AutoModelForCausalLM` -> `AutoBridge.from_hf_pretrained()`
- Model forward: HF backbone -> Megatron-Core model
- Optimizer: `bitsandbytes.AdamW8bit` -> Megatron-Core optimizer (or keep 8bit if needed)
- Checkpointing: must still export HF-format checkpoints for eval compatibility
- Weight sync: parameter names will differ -- need mapping for NCCL broadcast + vLLM transfer

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

Eval script: `/mnt/nvme0n1/rawhad/self_distillation/aligning_lm_from_user_interaction/scripts/eval_maas_sdft.py` (on node `rh-h100-01`).

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

## Environment

- Node: `rh-h100-01` (H100 cluster)
- GPUs 0-2 (or 3-5 for parallel run)
- Python 3.12 via `uv`
- vLLM pinned to 0.23
- Model: `Qwen/Qwen3-8B`
- Megatron Bridge source: `~/rawhad/ols-cpt/megatron-bridge/`

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
- [ ] Phase 1: Setup -- venv, deps, megatron-bridge install
- [ ] Phase 2: Model loading spike -- AutoBridge loads Qwen3-8B, forward pass works
- [ ] Phase 3: Student forward -- selective lm_head pattern with Megatron model
- [ ] Phase 4: Weight sync -- NCCL broadcast + vLLM transfer with Megatron params
- [ ] Phase 5: Full training loop -- end-to-end SDFT with Megatron Bridge
- [ ] Phase 6: Eval -- run eval, compare to reference scores
