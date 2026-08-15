---
title: "Megatron-Bridge LoRA Training — Deep Research Report"
date: 2026-08-14
tags: [sdft, lora, megatron-bridge, vllm, peft]
sources: 40+ web (docs.nvidia.com, GitHub source, PyPI, vLLM source, community blogs) + local codebase (train_dir/, megatron_trainer/)
---

# Megatron-Bridge LoRA Training — Deep Research Report

> Generated: 2026-08-14 | Sources: 40+ web + 4 local code files | Topic: NeMo Megatron-Bridge LoRA training & vLLM hot-swap, for SDFT issue #11

## TL;DR

- **Megatron-Bridge** (`megatron.bridge`, repo `NVIDIA-NeMo/Megatron-Bridge`) is NVIDIA's Megatron-Core training library and the **de-facto LoRA implementation for the whole MCore ecosystem** — adopted by verl, SkyRL, NeMo-RL, ms-swift, Mind Lab. Old `megatron.core.peft` is **gone** (verified: zero LoRA/PEFT code in Megatron-LM main, all tags, all PyPI `megatron-core` sdists); PEFT now lives in the bridge (`src/megatron/bridge/peft/`).
- **Bridge's recommended LoRA recipe**: `LoRA(target_modules=["linear_qkv","linear_proj","linear_fc1","linear_fc2"], dim=32, alpha=32)` via `finetune(cfg, forward_step_func)`; base frozen, **adapter-only** gradients/optimizer/checkpoints (few MB); adapters exported to **standard HF PEFT format** (`adapter_config.json` + `adapter_model.safetensors`) via `export_adapter.py` — verified safe end-to-end for **Qwen3-8B** (fused `linear_qkv` → q/k/v split math unit-tested, GQA-correct).
- **The hot-swap advantage from issue #11 is real and first-class**: vLLM's `POST /v1/load_lora_adapter` with `"load_inplace": true` was built explicitly for "asynchronous reinforcement learning setups, where adapters are continuously updated and swapped in without interrupting ongoing inference." Qwen3-8B is a supported LoRA base (`SupportsLoRA`), with a dedicated `test_load_inplace_offline_reload` test.
- **Two critical caveats for on-policy SDFT**: (1) in-place swap has **no per-request versioning** — an in-flight rollout silently switches to new weights mid-sequence; use the existing `/pause_generation` dev endpoint as a drain barrier. (2) Runtime LoRA updating requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, which vLLM labels **dev-only / production-unsafe** — acceptable here (isolated cluster, trusted trainer).
- **Bridge LoRA can train directly from a local HF directory** as `pretrained_checkpoint` (no Megatron conversion needed) — a 1-GPU Qwen3-8B LoRA recipe ships in-repo (`qwen3_8b_peft_1gpu_h100_bf16_config`, dim=8/alpha=16, attention-only), peak ~25–35 GB on H100-80GB.

---

## Overview

**Megatron-Bridge** is a PyTorch-native library inside the NeMo Framework that provides pretraining, SFT, and PEFT (LoRA/CanonicalLoRA/DoRA) for popular language/VLM/audio/multimodal models, plus bidirectional HF↔Megatron-Core checkpoint conversion. It is the current Megatron path in NVIDIA's 2026 repo re-org: `NVIDIA-NeMo/Automodel` (PyTorch DTensor/FSDP2, HF day-0) and `NVIDIA-NeMo/Megatron-Bridge` (Megatron-Core parallelism) are the two training backends; RL lives in `NVIDIA-NeMo/RL` and consumes both.

For SDFT (issue #11), the bridge matters for two reasons: (1) the repo's `megatron_trainer/` already runs inside the `nemo:26.06` container which **ships Megatron-Bridge 0.5.0** (verified software matrix), and already uses `AutoBridge` for model loading/conversion — so the LoRA path is available in the current environment with zero new infra; (2) the bridge's adapter export lands in exactly the format vLLM's LoRA hot-swap consumes.

---

## Key Findings

### 1. Bridge LoRA implementation (how it works)

- **Native PEFT stack on MCore primitives**: `src/megatron/bridge/peft/` (lora.py, lora_layers.py, canonical_lora.py, dora.py, lora_merge.py, multi_lora.py). Wraps `ColumnParallelLinear`/`RowParallelLinear`/TE linears with `ParallelLinearAdapter` + `LoRALinear`. HF `peft` is a hard dependency but used **only for HF-format export/verification**, never for training.
- **Frozen base confirmed in source**: `PEFT.__call__` → `freeze_model()` sets every base param `requires_grad=False`, then injects adapters (pre-wrap hook, before distributed wrapping). Only adapter params get gradients, master params, and optimizer state.
- **Two variants**: performant `LoRA` (default; ONE adapter on fused `linear_qkv` / fused `linear_fc1`) vs `CanonicalLoRA` (per-projection `linear_q/k/v`, `linear_fc1_up/gate`); mathematically equivalent when A-matrices tied. `DoRA` adds magnitude vector, `alpha` default 64 (vs 32 for LoRA).
- **Default `LoRA` fields**: `target_modules=["linear_qkv","linear_proj","linear_fc1","linear_fc2"]` (wildcards OK, `exclude_modules` supported), `dim=32` (rank), `alpha=32`, `dropout=0.0`, `dropout_position="pre"`, `lora_A_init="xavier"`, `lora_B_init="zero"`. MoE-only: `share_expert_adapters=True`, `normalize_moe_lora=False`, router excluded by default.
- **Adapter-only checkpointing**: `apply_peft_adapter_filter_to_state_dict` keeps only adapter keys (+ adapter-scoped optimizer state). `iter_N/` is a few MB in Megatron `torch_dist` format. Resume = adapter ckpt (`checkpoint.load`) + same frozen base (`pretrained_checkpoint`); `ckpt_step` selects the adapter iter only.
- **DoRA exports with `use_dora:true` — vLLM rejects DoRA** (`vLLM does not yet support DoRA.`). LoRA-only path unaffected.

### 2. The adapter → vLLM path (verified end-to-end, dense Qwen3-8B)

- **Mapping is implemented + unit-tested**: `Qwen3Bridge` registry maps mcore `linear_qkv` → HF `q_proj/k_proj/v_proj` (`QKVMapping`), `linear_fc1` → `gate_proj/up_proj` (`GatedMLPMapping`), `linear_proj`→`o_proj`, `linear_fc2`→`down_proj`. Export splits ONE fused qkv adapter into THREE per-projection adapters: **lora_A replicated identically across q/k/v**, lora_B sliced per projection via GQA-interleaved row indexing (`split_qkv_weights`; Qwen3-8B: 32 heads / 8 kv / head_dim 128 — verified). FC1 = plain dim-0 chunk at TP=1.
- **Exported `adapter_config.json`** is standard LoraConfig: `r←dim`, `lora_alpha←alpha`, `target_modules` inferred from weight names (HF names), `task_type:"CAUSAL_LM"`, `use_dora:false`. vLLM's `PEFTHelper` silently filters unknown keys; module matching is suffix-based with `packed_modules_mapping` (`qkv_proj←[q,k,v]`, `gate_up_proj←[gate,up]` for Qwen3 in vLLM).
- **Export paths**: `examples/conversion/adapter/export_adapter.py` (CPU mode, no GPU needed, float32 materialization by design) or the **`also_save_hf_checkpoint=True` sidecar** which writes `iter_N/hf/adapter_model.safetensors` + `adapter_config.json` automatically at every save — effectively replacing the manual export step. Logit-level verification via `verify_adapter.py` (top-k match vs Megatron merged weights).
- **Known safe-path conditions**: keep rank ≤ vLLM `max_lora_rank` (default 64; allowed: 1,8,16,32,64,128,256,320,512); exclude MTP (`--exclude-adapter-base-prefix mtp.layers` — vLLM Qwen3 has no `mtp.*`); PP=1 for export (open bug #5585 for PP>1 HF export).

### 3. vLLM hot-swap mechanics (the issue #11 payoff)

- **Endpoint** (gated by `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, else 404): `POST /v1/load_lora_adapter {"lora_name": "...", "lora_path": "...", "load_inplace": true}` → `200 Success: LoRA adapter '...' added successfully.` Same name keeps the same `lora_int_id` → **same preallocated GPU slot**; swap = load-new-first (CPU) → remove-old → overwrite slot. **Zero new GPU allocation per swap**; failed loads keep the old adapter serving.
- **Selection**: OpenAI API has no `lora_request` field; requests select the adapter via `"model": "<lora_name>"` (exact match; unknown → 404, never base fallback). `max_loras=1` (default) is correct for a single-policy loop.
- **⚠️ No versioning**: a request running during a swap silently continues with new weights from its next decode step. For on-policy consistency use the drain barrier already available: `/pause_generation` → push adapter → `/resume_generation` (dev mode already enabled in `start_vllm.sh`).
- **Ops**: GPU slot memory committed at startup from `--max-lora-rank` × `--max-loras` (set rank to the training rank, not higher); do NOT use `unload`+`load` instead of `load_inplace` (unload is frontend-only, in-flight requests resurrect the old path); rolling restart never needed for weight swaps; `VLLM_ALLOW_RUNTIME_LORA_UPDATING` logs a dev-only warning (isolated cluster OK).

### 4. What NVIDIA recommends (practices + hyperparameters)

- **Workflow**: (a) base = converted Megatron ckpt OR local HF dir (no conversion needed for LoRA); (b) `finetune()` with `cfg.peft=LoRA(...)`; (c) export adapter (or `also_save_hf_checkpoint`); (d) serve base + adapter.
- **LR**: PEFT recipes use `finetune_lr=1e-4` (vs 5e-6 for full SFT — ~20× higher), cosine decay, warmup 10–50 iters, wd 0.1 (qwen3 recipe), betas (0.9, 0.95).
- **Rank/alpha**: no official scaling table; de-facto guidance: 8B dense → `dim=8, alpha=16` (both the Qwen3-8B and Llama3-8B recipes), attention-only for Qwen3-8B (`linear_qkv, linear_proj`); defaults `32/32`; verl RL guidance: rank 32–128 for 0.5B–32B, LR ≈ 10× full-FT; Mind Lab Kimi K2 (1.04T MoE): rank 128.
- **Parallelism**: LoRA recipes are explicitly LOWER parallelism than full SFT — Qwen3-8B LoRA = TP1/PP1 on a single H100; recipes disable CUDA graphs for PEFT; packed sequences force `micro_batch_size=1`.
- **MoE**: `share_expert_adapters`, `normalize_moe_lora` (expert rank = dim//topk), router control; grouped-expert grad-sync bug (#5229, fixed Aug 2026). DeepSeek MLA needs explicit targets + has open bugs (#5261/#5294).
- **DoRA**: documented as "consistently outperforms LoRA" with TP/PP support; alpha default 64.

---

## Current vs Recommended (SDFT)

| Aspect | Current (repo) | Recommended (bridge LoRA path) | Evidence | Impact |
|---|---|---|---|---|
| Trainable params | Full 8B model, FSDP (`TorchFullyShardedDataParallel`), full AdamW | Adapter only (~7M params with r=8–32); base frozen | peft/base.py freeze_model; qwen3 recipe | ~10–20× less optimizer memory; no bitsandbytes needed |
| Weight sync to vLLM | Full 16 GB via NCCL weight transfer + pause/resume every optimizer step | Adapter dir (MBs) via `POST /v1/load_lora_adapter load_inplace`; no pause needed (drain barrier optional) | vLLM worker_manager add_adapter; docs/features/lora | ~1000× less data; zero engine downtime |
| Weight sync to logprob server | Full-weight EMA via NCCL broadcast | Adapter-only EMA (small tensors, same NCCL group); merge/in-place update into frozen base | bridge adapter-only ckpts | ~1000× less bandwidth; EMA semantics shift to adapter-only |
| Checkpoints | Full model per `SAVE_EVERY` | Adapter-only Megatron ckpt + optional `iter_N/hf/` HF-PEFT sidecar | checkpointing.py filter | MBs instead of GBs |
| vLLM serving format | Full weights, NCCL | Base model + HF PEFT adapter, hot-swappable | export_adapter.py → vLLM PEFTHelper | multi-policy/multi-adapter serving becomes possible |
| Rollout policy consistency | pause/resume = hard sync | drain barrier via existing `/pause_generation` (dev mode) | vllm serve/dev endpoints | same consistency, no weight transfer |
| Infra | custom trainer glue (train_dir + megatron_trainer) | in-repo recipes (`qwen3_8b_peft_1gpu_h100_bf16_config`) inside the already-used nemo:26.06 container (ships bridge 0.5.0) | software-versions matrix; megatron_trainer/README.md | no new containers; less custom code |

**EMA note (decision for the plan)**: current code EMAs full weights (`phi = (1-α)φ + αθ`). With LoRA, base is constant, so EMA applies to adapter params only. Since base never changes, `φ_full → base + EMA(Δ)` — mathematically consistent with the current behavior only if the base starts identical (it does: same frozen base both sides). Applying EMA to the merged delta instead of raw A/B is a modeling choice; the adapter-only EMA is the clean option.

---

## Alternatives considered (and why ruled out)

- **HF PEFT on the HF model (train_dir/ student path)**: works today, but no fused-QKV/TE kernels, no MCore parallelism, weaker MoE support, and — critically — **Mamba-2/hybrid coverage gaps** (peft#2274) if models evolve. Not ruled out for the HF train_dir; ruled out as the megatron path.
- **NeMo AutoModel (DTensor/FSDP2, HF day-0)**: saves adapters natively in HF format, serves via vLLM `LoRARequest` directly — attractive, but a **second** stack to adopt; the repo already runs the Megatron path (nemo:26.06 container, AutoBridge usage). Bridge keeps one stack. (Official positioning: bridge = "Megatron-based pretraining library"; AutoModel = "PyTorch DTensor-based pretraining library" — no official "use X" statement.)
- **MCore native LoRA (`megatron.core.peft` / `FusedLinearWithFusedLora`)**: **eliminated — does not exist anymore** in Megatron-LM main, any tag, or any PyPI `megatron-core` release (0.7.0–0.18.2 checked). Treat as historical API.
- **Full-weight SFT + NCCL sync (status quo)**: keeps the 16 GB sync and pause/resume cost issue #11 exists to fix; only viable when LoRA capacity is insufficient (not yet demonstrated for SDFT).

---

## Practical Guide

### A. Bridge LoRA training (1 GPU, Qwen3-8B, inside nemo:26.06)

```bash
# 0. Base: local HF dir works as pretrained_checkpoint (no conversion)
# 1. Train (recipe exists: qwen3_8b_peft_1gpu_h100_bf16_config, alias qwen3_8b_peft_config)
#    defaults: dim=8 alpha=16, targets linear_qkv+linear_proj, lr=1e-4 cosine, TP1/PP1,
#    GBS=32 MBS=1 seq=2048, bf16, distributed optimizer, ~25-35GB H100
```

```python
from megatron.bridge.recipes.qwen import qwen3_8b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.peft.lora import LoRA

cfg = qwen3_8b_peft_config("lora")                    # or "dora"
cfg.checkpoint.pretrained_checkpoint = "/local/path/to/Qwen3-8B"   # local HF dir OK
cfg.checkpoint.save = "/results/peft_ckpts"
cfg.checkpoint.load = None                            # fresh run (defaults auto-resume!)
cfg.peft = LoRA(target_modules=["linear_qkv","linear_proj","linear_fc1","linear_fc2"],
                dim=32, alpha=32, dropout=0.0)
# replace cfg.dataset (recipe default = SQuAD)
finetune(cfg, forward_step_func)
# launch: python train.py   (or torchrun --nproc_per_node=1)
```

### B. Export adapter (every save, or sidecar)

```bash
# Option 1: one-shot export (CPU-only, no GPU)
uv run python examples/conversion/adapter/export_adapter.py \
    --hf-model-path Qwen/Qwen3-8B \
    --lora-checkpoint /results/peft_ckpts/iter_0000100 \
    --output /results/adapters/step_100 \
    --exclude-adapter-base-prefix mtp.layers

# Option 2 (zero post-processing): set also_save_hf_checkpoint=True in the config
# → every save writes iter_N/hf/adapter_config.json + adapter_model.safetensors

# Gate: verify logits match Megatron (optional but recommended once)
uv run python examples/conversion/adapter/verify_adapter.py \
    --hf-model-id Qwen/Qwen3-8B --hf-adapter-path /results/adapters/step_100
```

### C. vLLM serving with hot-swap (the SDFT loop)

```bash
# start_vllm.sh additions:
#   --enable-lora --max-lora-rank 32 --max-loras 1 --max-cpu-loras 2
#   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

```python
# per optimizer step:
#   1. export adapter → adapters/step_{N}  (or read iter_N/hf sidecar)
#   2. optional drain barrier: POST /pause_generation ; wait for in-flight
#   3. POST /v1/load_lora_adapter {"lora_name": "sdft-policy",
#                                   "lora_path": "/abs/adapters/step_{N}",
#                                   "load_inplace": true}
#   4. POST /resume_generation
# rollout requests always use "model": "sdft-policy"
```

### D. Logprob server (teacher) with LoRA

Same frozen base + adapter in-place update via the existing NCCL group (adapter tensors only, MBs), replacing `broadcast_weights_ema` over full weights; EMA applied to adapter params. No merge math needed if both sides hold the same frozen base and update adapters in-place.

---

## Gotchas & Pitfalls

1. **In-flight rollouts mix policy versions** on swap (no versioning in vLLM) — drain via `/pause_generation` for on-policy consistency.
2. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING` is dev-only** — vLLM logs a warning; bind server to non-public interface; `lora_path` is host file access (trusted trainer only).
3. **Do not use unload+load instead of load_inplace** — unload is frontend-only; in-flight requests re-load the old path.
4. **`max_lora_rank` must equal the training rank** (memory scales linearly; allowed 1,8,16,32,64,128,256,320,512); rank above → 500 on load.
5. **Failed swap keeps old adapter serving** (load-before-remove) — retry next step; log the 500 body.
6. **`checkpoint.load` refuses HF dirs**; resume = adapter ckpt + same frozen base; `ckpt_step` only picks the adapter iter.
7. **Auto-resume hazard**: `_peft_common` defaults `save == load` → set `load=None` for fresh runs.
8. **`pretrained_checkpoint` accepts local HF dir but NOT remote HF IDs**; `also_save_hf_checkpoint` requires an HF source (auto-resolved) and is incompatible with `fsdp_dtensor` ckpt format.
9. **DoRA adapters can't be served by vLLM** (rejected at load). LoRA only for the serving path.
10. **Export precision**: CPU export materializes float32 by design (bf16 merge → ~1e-3 weight errors); adapter tensors keep training dtype on disk.
11. **`lora_A` triplicated across q/k/v** in exported safetensors (~3× attention adapter size on disk) — expected, don't "fix" it.
12. **MoE/MLA bugs if models evolve**: DeepSeek MLA targets + bugs (#5261/#5294); grouped-expert grad sync (#5229, fixed); Mamba-2 targets needed for Nemotron hybrid (`in_proj`/`out_proj`).
13. **Community lesson**: more LoRA coverage ≠ better results (100% coverage degraded RAFT F1 — recipe epochs overfit; tune `train_iters`).
14. **`finetune()` is `@experimental_fn`** — API may change without notice; pinned container (nemo:26.06 = bridge 0.5.0) mitigates.
15. **First training iter can take ~15 min** (graph capture + MoE warmup) at scale — don't kill the job.

## Hardware / Environment Considerations

- **nemo:26.06 container** (already used by `megatron_trainer/`): Megatron-Bridge 0.5.0, Megatron-Core 0.18.0, TE 2.16, PyTorch 2.12, Transformers 5.8.1, vLLM 0.20.1 (in-container; repo pins its own vLLM 0.23 via `pip install --no-deps`).
- **Python ≥3.12 required** (3.10 dropped at bridge 0.4.0).
- **1 GPU suffices** for Qwen3-8B LoRA (~25–35 GB peak, no recompute/offload); distributed optimizer is memory-neutral at DP=1.
- Megatron-Core is a pinned git submodule (`scripts/switch_mcore.sh main|dev`).
- vLLM LoRA memory: `max_lora_rank` × `max_loras` slot buffers committed at startup; hot swaps cost zero additional GPU memory.
- vLLM LoRA works with quantized bases (GPTQ/AWQ, bnb via plugin, FP8 kernels) — future headroom if the 8B base ever needs quantization.

## Next Steps: Implementation (mapping to issue #11)

1. **Verify environment**: confirm `megatron.bridge` importable inside the nemo:26.06 container + vLLM 0.23's `LoadLoRAAdapterRequest` has `load_inplace` (added ~v0.8.4; check installed source per AGENTS.md style).
2. **Phase A — bridge LoRA training**: replace full-weight FSDP trainer with `finetune()` + `LoRA` config (recipe base, custom dataset/collator → SDFT JSONL); enable `also_save_hf_checkpoint`.
3. **Phase B — vLLM hot-swap**: add `--enable-lora --max-lora-rank` to `start_vllm.sh` (or `start_vllm_patched.py`); new `push_lora()` in `vllm_utils.py` (save → load_inplace → optional pause/resume barrier); rollout `model="sdft-policy"`.
4. **Phase C — teacher adapter sync**: logprob server loads frozen base + adapter; NCCL sync filtered to adapter params; EMA on adapter params.
5. **Phase D — validation**: 0.6B smoke → 8B parity run (loss curve vs full-weight baseline), sync-time + rollout-downtime measurements, `verify_adapter.py` gate.

## Sources

1. https://docs.nvidia.com/nemo/megatron-bridge/latest/ (+ training/peft.html, training/checkpointing.html, training/data-preparation.html, models/qwen/qwen.html, nemo2-migration-guide.html, releases/software-versions.html)
2. https://github.com/NVIDIA-NeMo/Megatron-Bridge — src/megatron/bridge/peft/{lora.py,lora_layers.py,canonical_lora.py,dora.py,lora_merge.py,utils.py,base.py}; training/{finetune.py,checkpointing.py,config.py}; recipes/qwen/h100/qwen3.py; recipes/common.py; examples/conversion/adapter/{export_adapter.py,verify_adapter.py,README.md}; examples/peft/merge_lora.py; models/conversion/{auto_bridge.py,peft_bridge.py,param_mapping.py}; models/qwen/qwen3_bridge.py; docs/training/peft.md; tutorials/recipes/llama/01_quickstart_finetune.py
3. Issues/PRs: #5261, #5294, #5229, #4565, #4847, #4246, #5585, #5483, #5589, #5396, #5376, #5484
4. https://pypi.org/project/megatron-bridge/ (0.5.1) ; https://pypi.org/project/megatron-core/ (0.7.0–0.18.2 inspected, no PEFT)
5. https://github.com/NVIDIA/Megatron-LM (main @ 2a75ac12 + tags: no peft/lora anywhere)
6. https://docs.vllm.ai/en/stable/features/lora/ + vllm source: entrypoints/serve/lora/{api_router.py,protocol.py}, entrypoints/openai/models/serving.py, entrypoints/serve/engine/serving.py, lora/{worker_manager.py,model_manager.py,lora_model.py,peft_helper.py,utils.py,request.py}, config/lora.py, envs.py, model_executor/models/qwen3.py, tests/lora/test_qwen3_with_multi_loras.py
7. https://github.com/NVIDIA-NeMo/Automodel (PeftConfig, finetune.mdx vLLM deploy)
8. verl: https://raw.githubusercontent.com/volcengine/verl/main/docs/advance/ppo_lora.rst + examples/tuning/lora/run_qwen3_30b_a3b_megatron.sh
9. Classmethod walkthrough: https://dev.classmethod.jp/en/articles/nemotron-9b-megatron-bridge-brev/ (+ himorishige/dgx-spark-blog)
10. Mind Lab trillion-param GRPO-LoRA: https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus
11. Nemotron-3 Ultra cookbook: NVIDIA-NeMo/Nemotron usage-cookbook/Nemotron-3-Ultra/lora-text2sql/nemo-megatron-bridge
12. ms-swift: https://github.com/modelscope/ms-swift/pull/6714 ; peft: https://github.com/huggingface/peft/issues/2274
13. https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
14. Local: megatron_trainer/{README.md,model_utils.py,train_full.sh}, train_dir/{start_vllm.sh,src/{vllm_utils.py,trainer.py,logprob_server.py,config.py}}
