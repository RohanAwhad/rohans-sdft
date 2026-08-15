# Goal — Issue #22: LoRA for gpt-oss (GQA dim mismatch in vLLM adapter loader)

## Goal
Make `TRAIN_MODE=lora` work with `MODEL_NAME=unsloth/gpt-oss-20b-BF16` end-to-end, the same way it works with Qwen3-8B.

## Objective
1. Reproduce the issue on node 12 (`rh-h100-12`) on branch `ra/autoresearch-loop` (v0.3.0 base).
2. RCA the failure chain: adapter export succeeds → `POST /v1/load_lora_adapter` 500s (`tensor a (2880) vs b (5760) at non-singleton dimension 2`) → adapter never loads → rollout 404 (`The model 'sdft-policy' does not exist`) → rollout-producer thread crashes.
3. Fix the export/load path so vLLM's gpt-oss LoRA loader accepts the adapter (GQA layout: 64 heads, 8 kv heads, hidden 2880).
4. Update design docs at `docs/megatron_trainer/` to reflect the fix (root cause, constraints, gotchas).
5. Raise a PR with the tested solution.

## End State (Definition of Done — all must hold)
- [x] `TRAIN_MODE=lora` + `MODEL_NAME=unsloth/gpt-oss-20b-BF16` run trains for **20 consecutive optimizer steps without error** on node 12 — **25/25 steps, `Training complete.`** (trainer.log, 2026-08-15)
- [x] Adapter export → vLLM `load_lora_adapter` push succeeds (no 500), verified in logs — **26/26 pushes `Success: LoRA adapter 'sdft-policy' added successfully.`**
- [x] Rollout requests hit the adapter (no 404 `model 'sdft-policy' does not exist`), verified in logs — **0 vLLM 4xx/5xx, 400 completions served**
- [x] Rollout-producer thread does not crash across the 20 steps
- [x] Root cause documented (RCA writeup in the PR + devlogs)
- [x] `docs/megatron_trainer/*` updated with the fix and any new invariants
- [x] PR raised with the solution, tested on node 12 (test evidence in PR body) — **PR #23**

### RCA summary
Bridge `save_hf_adapter` exports MoE expert LoRA (gpt-oss) with `lora_A`/`lora_B` swapped and transposed (A carries 2×intermediate=5760, B carries hidden=2880). vLLM's `_stack_moe_lora_weights` expects PEFT layout `lora_A (E×r, hidden)` / `lora_B (2N, E×r)` → `copy_` dim-2 mismatch → 500 → adapter never loads → rollout 404 → producer crash. Fix: `_fix_fused_expert_gate_up_adapter_layout` swap+transpose post-processing in `save_hf_adapter_checkpoint` (both gate_up AND square down_proj). Verified: value parity max|Δ|=0.00e+00, 25-step clean run.

## Out of scope
- Qwen3-8B LoRA regression (must keep working, but no new work)
- FSDP-LoRA, DoRA, adapter versioning in vLLM
- Hyperparameter sweeps (existing GOAL.md phase is separate)

## Repro reference
- Branch: `ra/autoresearch-loop` (rebased on v0.3.0)
- Config: `TRAIN_MODE=lora`, `MODEL_NAME=unsloth/gpt-oss-20b-BF16`
- Error: `The size of tensor a (2880) must match the size of tensor b (5760) at non-singleton dimension 2`
