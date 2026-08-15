# LoRA Support — SDFT Megatron Trainer (`TRAIN_MODE=lora`)

> Implements issue #11 (feat: LoRA Support). Design grounded in
> `docs/RESEARCH_megatron_bridge_lora.md` (source-verified: Megatron-Bridge
> native PEFT, bridge→HF-adapter export, vLLM hot-swap semantics).

## Role

Adds a **switchable LoRA training mode** to `megatron_trainer/`. The existing
full fine-tuning path (`TRAIN_MODE=full`, default) is untouched; `TRAIN_MODE=lora`
trains only Megatron-Bridge LoRA adapters on a frozen base and serves the
policy in vLLM via **hot-swappable adapters** (`POST /v1/load_lora_adapter` with
`load_inplace: true`) instead of per-step full-weight NCCL transfer.

Motivations (from research):

- Full-weight sync is ~16 GB (Qwen3-8B) per optimizer step per sink, with
  vLLM pause/resume downtime; adapters are MBs and swap atomically with **zero
  vLLM downtime** (no pause needed for the engine — see drain barrier below).
- Adapter-only optimizer states (~7M params at dim=32 vs 8B) — no more
  optimizer memory pressure at 8B scale.
- Adapters land in standard HF PEFT format (`adapter_config.json` +
  `adapter_model.safetensors`) — the exact format vLLM's LoRA loader consumes.

## Design decisions

1. **LoRA engine = Megatron-Bridge native PEFT** (`megatron.bridge.peft.lora.LoRA`),
   applied inside the existing custom trainer loop — the **verl pattern**
   (bridge as connector, custom RL loop), NOT a swap to bridge's `finetune()`.
   Rationale: SDFT's loop is on-policy RL (vLLM rollout → teacher logprobs →
   reverse KL), not SFT; `finetune()` would force a loop rewrite.
2. **Mode switch via one env var** — `TRAIN_MODE=full|lora` (default `full`).
   Every subsystem branches at exactly one point (see table below); the
   rollout/teacher-logprob/loss core is mode-agnostic.
3. **No FSDP in LoRA mode.** Trainer ranks hold a replicated frozen base and
   all-reduce the (tiny, ~14 MB bf16 at dim=32) adapter grads via plain
   `dist.all_reduce` on flattened grads. Rationale: bridge LoRA + MCore FSDP is
   unverified upstream (#4246); adapter grads are small enough that FSDP's
   machinery buys nothing. Memory: full base per rank (~16 GB + activations,
   ~25–35 GB at seq 2048 per research) still fits H100-80GB. FSDP-LoRA is
   future work.
4. **DoRA is out of scope** — vLLM rejects DoRA adapters at load
   (`vLLM does not yet support DoRA.`). LoRA only.

## Mode-branch table

| Subsystem | `TRAIN_MODE=full` (existing) | `TRAIN_MODE=lora` (new) |
|---|---|---|
| Model init (`model_utils.py`) | `load_model()` + FSDP wrap (`register_fsdp_module_mappings`, `TorchFullyShardedDataParallel`) | `load_model()` + `LoRA(...)` transform, base frozen; no FSDP wrap |
| Optimizer / grad clip (`trainer.py`) | all params, AdamW (as-is) | trainable params only (`requires_grad` filter) — one shared line works for both |
| Grad sync across ranks | FSDP `no_sync` / `finish_grad_sync` | `dist.all_reduce` of flattened adapter grads at step end |
| vLLM sync (`vllm_utils.py`) | NCCL weight transfer + pause/resume (as-is) | `push_lora_adapter()`: export adapter dir → drain barrier → `load_inplace` hot-swap |
| vLLM generation | `"model": MODEL_NAME` (as-is) | `"model": LORA_ADAPTER_NAME` (adapter selection by model name) |
| Teacher sync (`logprob_server.py`) | full-weight NCCL broadcast + EMA (as-is) | adapter-params-only broadcast + EMA on adapters |
| Checkpoints (`trainer.py`) | `save_hf_checkpoint` full model (as-is) | adapter-only HF PEFT dir at `SAVE_EVERY` |
| vLLM server launch | current flags (as-is) | adds `--enable-lora --max-lora-rank --max-loras`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` |

## New config contract (`config.py`)

| Var | Default | Notes |
|---|---|---|
| `TRAIN_MODE` | `"full"` | `"full"` \| `"lora"`; anything else raises |
| `LORA_DIM` | `32` | adapter rank; **must be in vLLM `MaxLoRARanks`** (1, 8, 16, 32, 64, 128, 256, 320, 512) and equal the vLLM `--max-lora-rank` |
| `LORA_ALPHA` | `32` | scaling; vLLM uses `alpha / r` |
| `LORA_DROPOUT` | `0.0` | |
| `LORA_TARGET_MODULES` | `"linear_qkv,linear_proj,linear_fc1,linear_fc2"` | comma-separated mcore names (bridge names, not HF names) |
| `LORA_ADAPTER_NAME` | `"sdft-policy"` | vLLM `lora_name`; must match the `"model"` field of rollout requests exactly |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `True` (set in launch) | required for `/v1/load_lora_adapter`; dev-only per vLLM (isolated cluster OK) |

Bridge recipe reference (research, `qwen3_8b_peft_1gpu_h100_bf16_config`):
dim=8/alpha=16, attention-only targets, lr=1e-4 cosine — treat as a starting
point, not law; our custom loop keeps `LEARNING_RATE` as the knob.

## Model loading (`model_utils.py`)

- `load_model()` unchanged; then, in `lora` mode, apply the bridge PEFT
  transform before the return:
  `LoRA(target_modules=LORA_TARGET_MODULES, dim=LORA_DIM, alpha=LORA_ALPHA,
  dropout=LORA_DROPOUT)` — freezes the base (`requires_grad=False` everywhere),
  marks adapter params trainable.
- LoRA mode: **skip** `register_fsdp_module_mappings` / FSDP wrap.
- Activation recompute settings on `model.config` stay as-is (long-completion
  backward still needs them).

## Trainer changes (`trainer.py`)

- Optimizer: built on trainable params only (`[p for p in model.parameters()
  if p.requires_grad]`); `clip_grad_norm_` over the same set.
- Grad sync (LoRA mode): before the optimizer step, flatten + `dist.all_reduce`
  adapter grads across trainer ranks (replaces `finish_grad_sync()`); norm
  accounting unchanged (existing all-reduced norm² pattern).
- Checkpointing (LoRA mode): at `SAVE_EVERY` / epoch end, export the adapter
  via the bridge (`AutoBridge.save_hf_adapter(model=..., path=..., peft_config=...,
  base_model_name_or_path=HF_MODEL_PATH)`) to `OUTPUT_DIR/step_{N}` →
  `step_{N}/adapter_config.json` + `adapter_model.safetensors`. Same dir
  layout as full mode; the vLLM push consumes these dirs directly.
- wandb: add `train_mode`, `lora_dim`, `lora_alpha` to run config; log
  `adapter/swap_latency_ms` and `adapter/push_bytes` per sync.

## Weight sync — vLLM (`vllm_utils.py`)

New `push_lora_adapter(model, adapter_dir, rank)` (LoRA mode only), per
instance in `VLLM_BASE_URLS`:

1. Export adapter dir (or reuse the checkpoint dir from the current step).
2. **Drain barrier**: `POST /pause_generation` → wait for in-flight rollouts
   → push → `POST /resume_generation`. Required because vLLM has **no
   per-request adapter versioning** — a request running across the swap
   silently continues with new weights mid-sequence (on-policy inconsistency).
3. `POST /v1/load_lora_adapter` with body
   `{"lora_name": LORA_ADAPTER_NAME, "lora_path": "<abs path>", "load_inplace": true}`
   → expect `200 Success: LoRA adapter '...' added successfully.`
4. Non-200 → log the raw body (rank-too-high / module-mismatch surface as
   500); the old adapter keeps serving (load-before-remove), retry next step.
   **Never** use `unload` + `load` instead of `load_inplace` (unload is
   frontend-only; in-flight requests resurrect the old path).

Generation: `vllm_generate` sends `"model": LORA_ADAPTER_NAME` in LoRA mode
(unknown adapter name → 404, never a silent base fallback).

The full-mode NCCL path (`sync_weights_to_vllm`, weight engines) is left
intact and simply not called in LoRA mode.

## Teacher sync (`logprob_server.py`)

- LoRA mode server load: `load_model(HF_MODEL_PATH)` + same `LoRA(...)`
  transform (identical config — parameter order must match the trainer's).
- `/sync_weights` handler: broadcast **adapter params only** in order
  (`[p for p in model.parameters() if p.requires_grad]`), `lerp_` with
  `EMA_ALPHA` in-place. Base is frozen and identical on both sides.
- EMA semantics: EMA now applies to adapter params only (base constant).
  Consistent with the current full-weight EMA since the base never changes
  (see research report "EMA note").
- External frozen teacher path (`TEACHER_MODEL_PATH` set) unchanged.

## vLLM server launch (`train_full.sh` / `start_vllm_patched.py`)

In LoRA mode append to the vLLM launch:

```
--enable-lora --max-lora-rank ${LORA_DIM} --max-loras 1 --max-cpu-loras 2
```

plus env `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Keep `VLLM_SERVER_DEV_MODE=1`
(the `/pause_generation` drain barrier and weight-transfer endpoints are dev
gated). `--max-lora-rank` must **equal** `LORA_DIM` exactly (GPU slot buffers
are preallocated from it; memory scales linearly — don't oversize).

## Hard invariants

- `TRAIN_MODE ∈ {"full", "lora"}` — raised at config import otherwise.
- `LORA_DIM ∈ {1, 8, 16, 32, 64, 128, 256, 320, 512}` and
  `LORA_DIM == vLLM --max-lora-rank`.
- LoRA mode rollout requests always carry `"model": LORA_ADAPTER_NAME`;
  a 404 means the adapter isn't loaded — fail fast, don't fall back to base.
- Adapter push happens **between rollout waves** under the pause/resume drain
  barrier; never mid-wave.
- Trainer and logprob server must use the **same** `LoRA` config
  (rank/targets) — parameter order of the NCCL adapter sync depends on it.
- Only rank 0 performs adapter export + HTTP push; export is not an FSDP
  collective in LoRA mode (no sharding), so non-zero ranks do nothing.

## Known gotchas

- **No adapter versioning in vLLM** — mid-sequence policy switch if the drain
  barrier is skipped (see above).
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` is flagged dev-only by vLLM
  (startup warning); acceptable on the isolated cluster, keep the server off
  public interfaces.
- Exported `adapter_model.safetensors` triplicates `lora_A` across
  q/k/v (fused `linear_qkv` → per-projection split) — expected, ~3× attention
  adapter size on disk, don't "fix" it.
- `lora_path` must be an absolute path on the vLLM host.
- DoRA (`peft_scheme="dora"`) is NOT servable by vLLM — LoRA mode only.
- Export materializes in float32 by design (bf16 merges give ~1e-3 weight
  errors); adapter tensors keep training dtype on disk.
- Adapter rank validation happens at load: rank > `--max-lora-rank` → 500 on
  push. Keep `LORA_DIM` and the server flag in lockstep.

## Validation plan

1. **Smoke (0.6B/1B)**: LoRA-mode run end-to-end on 2 GPUs (vLLM + trainer) +
   logprob server; verify loss decreases, adapter pushes succeed each step,
   rollout uses the pushed adapter (log `model` in requests).
2. **`verify_adapter.py` gate (one-time)**: exported adapter logit-parity vs
   the Megatron-side merged weights (GQA split sanity for Qwen3-8B).
3. **Parity run (8B)**: `TRAIN_MODE=full` vs `TRAIN_MODE=lora` on identical
   SDFT data + seeds — loss curves, `pass_rate`, completion lengths.
4. **Metrics**: sync payload size (16 GB → MBs), per-step sync wall time,
   vLLM rollout downtime per sync (~0 with hot-swap), adapter swap latency.

## References

- `docs/RESEARCH_megatron_bridge_lora.md` — full research report (bridge LoRA
  recipe, adapter export path, vLLM hot-swap internals, hyperparameter table).
- Bridge docs: `docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html`
- vLLM LoRA: `docs.vllm.ai/en/stable/features/lora/`
