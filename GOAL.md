# GRPO Loss Implementation

## Objective

Implement `LOSS_TYPE=grpo` end-to-end in `megatron_trainer` from
`docs/megatron_trainer/grpo.md`, while keeping the default `LOSS_TYPE=sdft`
path behavior unchanged.

## I/O Contract

- Input prompt: one collated dataset item.
- Rollout group: `GRPO_GROUPS` completions from that same prompt, each with
  completion text, rollout token log-probs, finish reason, binary reward,
  group id, and policy version.
- `GRAD_ACCUM_STEPS` continues to mean trained completions per optimizer step.
  Prompts per step are `GRAD_ACCUM_STEPS / GRPO_GROUPS`.
- Groups are never split across ranks. Required invariant:
  `GRAD_ACCUM_STEPS % (world_size * GRPO_GROUPS) == 0`.
- Output: one optimizer update from the DAPO token-level GRPO objective plus
  optional K3++ reference KL, followed by the existing vLLM weight sync.
- Persistent state: model, optimizer, scheduler, checkpoints, and logs.
  Group rewards/advantages and rollout-policy metadata are transient payloads.

## Phase 1: Config and Launch Contract

- [x] Add and validate every GRPO environment variable in the design spec.
- [x] Keep `LOSS_TYPE=sdft` and all current SDFT defaults unchanged.
- [x] Use GRPO LR, warmup, and grad-clip overrides only in GRPO mode.
- [x] Pass GRPO variables through `train_full.sh` and smoke launchers.
- [x] Require a reward-producing env configuration in GRPO mode.

## Phase 2: Grouped Rollouts

- [x] Generate G independent completions per prompt; seeded runs use distinct,
  deterministic seeds within each group.
- [x] Carry reward, finish reason, group identity, rollout log-probs, and policy
  version in sync and async payloads.
- [x] Compute `mean`, `zscore`, and `median`/MAD advantages at the group boundary.
- [x] Mask length-truncated completions without applying a punitive reward.
- [x] Implement optional degenerate-group filtering with a bounded resample cap.
- [x] Preserve whole groups through rank slicing and async queue pulls.

## Phase 3: GRPO Loss

- [x] Add `make_grpo_processor` using the existing one-call LM-head hook.
- [x] Compute selected-token policy log-probs with gradients in row chunks.
- [x] Implement asymmetric PPO clipping and DAPO token-level normalization.
- [x] Implement sequence-level vLLM IS truncation/masking.
- [x] Apply GPG non-degenerate-group rescaling.
- [x] Add optional K3++ reference KL with log-ratio clamp and special-token mask.
- [x] Emit pass-rate, zero-std, entropy, length, clip, advantage, KL, and
  sampling-mismatch metrics.

## Phase 4: Trainer Integration

- [x] Branch locally inside `_train_sample`; leave the SDFT branch intact.
- [x] Skip all teacher log-prob work in GRPO mode when `GRPO_KL_COEF=0`.
- [x] Keep EMA/frozen reference behavior when `GRPO_KL_COEF>0`.
- [x] Normalize gradients over active completion tokens and prompt groups.
- [x] Reject non-finite GRPO loss/gradients visibly.
- [x] Keep checkpointing and vLLM weight synchronization unchanged.

## Phase 5: Local Verification

- [x] Hand-check binary group advantages, including degenerate groups.
- [x] Prove detached-old-logp GRPO gradients equal plain policy-gradient
  gradients on toy logits.
- [x] Hand-check active clipping and sequence IS for mismatched vLLM log-probs.
- [x] Hand-check K3++, special-token masking, and the +/-20 clamp.
- [x] Verify sync and async group-to-rank assignments.
- [x] Run SDFT regression coverage and syntax/type checks.

## Phase 6: Node 12 Verification

- [x] Run a 2-trainer smoke with `G=8`, `GRAD_ACCUM_STEPS=16`, and debug logs.
- [x] Cite `logs/trainer.log` line numbers proving finite loss, finite non-zero
  gradients for mixed groups, optimizer progress, GRPO metrics, and weight sync.
- [x] Prove no NaN, OOM, collective mismatch, or deadlock in the smoke logs.
- [x] Run two optimizer steps and report observed pass-rate movement;
  distinguish mechanical correctness from model-quality conclusions.
- [x] Record commands, effective config, results, and gotchas in `devlogs.md`.

## Done

The goal is complete only when local numeric/dataflow tests pass, the existing
SDFT path still passes its regression checks, and node-12 log evidence proves a
real GRPO optimizer run works end-to-end.
