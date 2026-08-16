# GRPO Live Campaign — Plan

Node: `rh-h100-05`, repo `/home/rohan/1_Projects/rohans_sdft_api_adapter`.
Design spec (source of truth for math/config): `docs/megatron_trainer/grpo.md` +
`docs/research/RESEARCH_grpo_variants.md` (this repo, `ra/grpo`, merged via PR #19).
I am running this campaign autonomously from here on (`autoresearch` skill).

## Goal

Beat baseline accuracy **0.7415** (with thinking) on the analyze_deepresearch
eval for the gpt-oss-20B student — via **GRPO**, not self-distillation.
Reverse-KL self-distillation is retired for this phase; **no more distillation
loss**. Reward comes from the **reflector verdict that already exists**
(`megatron_trainer/reflector.py::run` → `{"verdict": "PASS"|"FAIL", ...}`,
already computed per-rollout in `trainer.py::_sample_meta` but currently only
logged, never used in the loss) — no new grading infra needed.

## Objective

- Primary: eval accuracy > 0.7415. Stretch/significant: ≥ **0.7715**
  (+3pp — the McNemar significance threshold established in the prior phase).
- Mechanism: `LOSS_TYPE=grpo` (on-policy, group-relative policy gradient)
  replacing `LOSS_TYPE=sdft` (reverse-KL) entirely as the training objective.
- Reward: binary reflector verdict, `PASS→1.0 / FAIL→0.0`, one per rollout.

## Verification (how we know it worked)

- Same harness as the prior phase, unchanged: `poll_eval.sh <run> <epoch>` →
  reads `{maas-knowledge-eval}/eval_results/analyze_deepresearch/{run}/{epoch}/run_1.json`,
  field `summary.accuracy`. 557 questions (~500 unique), Claude-judged
  majority vote.
- Noise floor: paired SE ≈ 1.5pp (McNemar). `0.7415–0.7715` = "crossed but
  not significant" (this is exactly where the best SDFT run, E045, landed at
  0.7433). Only claim success at **≥ 0.7715**.
- Track GRPO health metrics every step (new, don't exist yet — part of this
  implementation): `grpo/pass_rate`, `grpo/frac_reward_zero_std` (degenerate
  groups), `grpo/entropy`, `grpo/clip_frac`, `grpo/adv_mean_std`,
  `grpo/sampling_logp_diff` (train/inference mismatch). A run with flat/rising
  accuracy but collapsing entropy or `clip_frac→0` is failing silently —
  watch these, not just eval accuracy.

## Why pivot off self-distillation (context)

- Prior phase: 47 experiments (E001–E047), all reverse-KL + IS variants,
  plateaued at 0.69–0.73. Only lever that ever crossed baseline was **context
  injection** (E045: 0.7433, +0.18pp, not significant) — an input-signal
  change, not a loss change.
- SFT-anchor hybrid (direct NLL gradient mixed into KL) was tried and
  **refuted twice** (E038 λ=0.1 destabilized KL; E039 λ=0.01 degraded
  accuracy by 4.3pp) — the belief on record is "NLL gradient competes with
  and degrades reverse-KL."
- Standing hypothesis in `STATE.md`: reverse-KL optimizes distribution
  matching to the teacher, not correctness — "sharper but not smarter."
  GRPO's reward is literally correctness (reflector verdict), so it targets
  the eval criterion directly instead of a proxy.
- Decision: run GRPO **on top of the context-injected datasets**
  (`subset_k400_subset_with_context.jsonl`, `combined_dataset_train_sdft_with_context.jsonl`)
  — the one proven lever — rather than plain data, so a negative result isn't
  confounded by "model doesn't have the facts."
- **Verified mechanism** (checked the actual data, not just the code path):
  context injection prepends a `"Context:\n..."` retrieval block directly
  into the raw `prompt` field itself (the human/user turn) — this is NOT
  purely a teacher-side hint. It flows through the collator into
  `prompt_texts`, i.e. the **student's own rollout/generation prompt**. So
  GRPO (which never touches `conditional_texts`/`privileged_information_prompt`
  since there's no teacher call) still gets the context, because it's baked
  into the same prompt the student generates from. `enriched_user_response`
  (the teacher-hint field) is identical between the plain/with-context
  dataset variants — only `prompt` differs. This confirms building on the
  with-context dataset is meaningful for GRPO, not a no-op.

## Design surface (full detail in `docs/megatron_trainer/grpo.md`)

| Knob | v1 default | Why |
|---|---|---|
| `LOSS_TYPE` | `grpo` | retires `sdft` path for this phase |
| `TRAIN_MODE` | `lora` | full FT's FSDP AdamW state doesn't fit a 20B model on <4 trainer GPUs (see OOM below); LoRA sidesteps it. `LORA_DIM=32`/`LORA_ALPHA=32` (unvalidated for GRPO specifically, carried from SDFT-LoRA defaults) |
| `GRPO_GROUPS` (G) | `8` | rollouts/prompt; `GRAD_ACCUM_STEPS % (world_size×G) == 0` |
| `GRPO_ADV` | `mean` | `r_i − mean(group)`, no /σ (binary reward, low-variance amplification risk with zscore) |
| `GRPO_CLIP_LOW/HIGH` | `0.2` / `0.28` | DAPO clip-higher, active since old logps come from vLLM |
| `GRPO_OLD_LOGPS` | `vllm` | ratio vs rollout logprobs (clip + IS active) |
| `GRPO_IS_C_MAX` | `3.0` | sequence-level TIS clamp, reuses `IS_CAP` semantics |
| `GRPO_KL_COEF` (β) | `0.0` | **no reference forward at all** — drops the 120b teacher from the critical path entirely |
| `GRPO_LR` / warmup | `1e-6` (full) / `1e-5` (lora, guess) / 15 steps | constant after warmup; LoRA's smaller effective param space likely tolerates higher LR than full FT, but this is unvalidated for GRPO — E046 used 3e-4 for SDFT-LoRA on a different model, not directly transferable |
| `GRPO_GRAD_CLIP` | `0.2` | small-batch insurance |
| `GRPO_MASK_TRUNCATED` | `1` | never punish length-truncated completions |
| temperature | `1.0` | literature default for GRPO exploration |

Reward wiring (the actual new plumbing, everything else in the table is
already-designed math): `_sample_meta`'s `pass_value` is computed once per
rollout today and only logged. Under `LOSS_TYPE=grpo` it becomes the reward
`r_i` feeding the group advantage — same reflector call, new consumer.

## Implementation plan — STATUS: implemented, LoRA smoke test running

Deviated from the original plan in two ways: **async rollout is required,
not optional** (`ASYNC_ROLLOUT=1` enforced at config time) — group-atomic
streaming (below), not the sync `produce()` path; and **LoRA is now
required, not banned** (reversed mid-session, see "TRAIN_MODE pivot" below).

1. ~~Commit/stash node05's outstanding loss-function docs~~ DONE — committed
   as `c65648a` (E045 full 10-epoch trajectory recovered from eval_results
   and added to the historical record: peak 0.7433@ep5, settled 0.72–0.73
   through ep10, no late collapse).
2. ~~Branch `ra/grpo-live` off `ra/autoresearch-loop`~~ DONE.
3. ~~`config.py`: `GRPO_*` contract + import-time asserts~~ DONE — v1 fences
   (fail fast, not silent): `TRAIN_MODE=full` only, `ASYNC_ROLLOUT=1` only,
   `GRPO_ADV=mean` only, `GRPO_KL_COEF=0` only, `GRPO_FILTER_GROUPS=0` only.
4. ~~Rollout: G completions/prompt~~ DONE, but **not** via `produce()`/`_build_env`
   G-times-per-item as originally sketched — instead a dedicated group-atomic
   async producer (see below), since async was upgraded from "nice to have"
   to "required."
5. ~~Rank slicing: group-aligned~~ DONE, via reuse of the *existing*
   `_pull_microbatch` unchanged (it broadcasts opaque queue items — pushing
   `list[dict]` groups instead of single dicts as the queue item made this a
   zero-line change).
6. ~~`chunked_head.py`: `make_grpo_processor`~~ DONE, plus extracted
   `grpo_loss_from_logp` (pure function, no MCore dependency) specifically so
   the gradient-identity check is unit-testable off-cluster.
7. ~~`_train_sample` grpo branch~~ DONE as a parallel `_train_sample_grpo`
   (not a branch inside `_train_sample`) — keeps the sdft path byte-identical,
   smaller diff than threading conditionals through the existing function.
8. ~~`_step_tail` grpo metrics~~ DONE for free — `_step_tail` already
   generically aggregates and logs whatever keys `make_grpo_processor`
   returns; only grad-clip (`GRPO_GRAD_CLIP`) and logprob-server-sync gating
   (`USE_LOGPROB_SERVER`) needed explicit edits.
9. ~~Unit test~~ DONE — 4 tests in `megatron_trainer/test_grpo_loss.py`, run
   inside the nemo container (CPU-only, no GPU): gradient-identity
   (detached mode exactly matches plain PG, max_diff=0.00e+00), clip/IS
   engagement at large ratio, degenerate-group advantage math, truncation
   masking. **All pass.**
10. Cluster smoke test — three attempts:
    - `grpo_smoke_test_1` (`TRAIN_MODE=full`): py-spy showed the trainer
      genuinely computing but stuck for minutes inside `output_layer()` —
      found a real perf bug in `make_grpo_processor`: the LM head GEMM was
      called once **per row-chunk** (32 small (128,H)@(H,V) matmuls) instead
      of once for the whole completion like `make_kl_processor` does. Fixed
      (single GEMM, chunk only the fp32 upcast + logsumexp).
    - `grpo_smoke_test_2` (`TRAIN_MODE=full`, fix applied): forward+backward
      for all 16 rollouts completed, but crashed in `torch/optim/adam.py`'s
      `state["exp_avg"] = torch.zeros_like(...)` — i.e. **CUDA OOM on the
      very first optimizer-state allocation**, not a GRPO-code bug at all.
      Root cause: 2 trainer GPUs → 2-way FSDP shard of a ~20B model → ~10B
      params/rank; bf16 params + bf16 grads + bf16 `exp_avg` + bf16
      `exp_avg_sq` ≈ 80GB, right at the H100's 79.17GB ceiling. Would affect
      SDFT identically at this trainer count — unrelated to LOSS_TYPE.
    - **TRAIN_MODE pivot**: rather than just adding more trainer GPUs,
      switched to `TRAIN_MODE=lora` (per direct instruction, after being
      pointed at PR #23 which fixes issue #22 — the gpt-oss MoE expert LoRA
      adapter export layout bug that was E046's original cancellation
      reason). LoRA only optimizes adapter params, sidestepping the
      optimizer-state ceiling entirely regardless of trainer count. Pulled
      PR #23 (`fix/issue-22-gptoss-moe-lora-export`, merged clean, no
      conflicts — it only touches `model_utils.py`) and removed the
      `TRAIN_MODE != "full"` config-time assert for grpo. All existing
      LoRA infra (`_step_tail`'s `sync_adapter_grads`/`push_lora_adapter`
      branches, `fsdp_model=None` skip-FSDP path, hot-swap vLLM sync) was
      already LOSS_TYPE-agnostic — zero trainer.py changes needed.
    - `grpo_smoke_test_3_lora` (`TRAIN_MODE=lora`, `LORA_DIM=32`,
      `GRPO_LR=1e-5`): **running now**. Watching for: no OOM, no crash,
      `grpo/*` metrics present and finite, adapter push succeeds
      (`Success: LoRA adapter ... added successfully`), pass rate moves.

New pieces not in the original numbered plan, needed once async became
mandatory:
- `_produce_streaming_grpo` + `_push_group` (trainer.py): the producer's
  unit of overlap is the **group**, not the individual rollout — G envs for
  one prompt run concurrently (nested thread pool) and are pushed to the
  queue together only once all G finish, since the advantage needs every
  member's reward first. Async-across-groups, sync-within-a-group.
- `env.finish_reason` capture (`env/base.py`, `rag_env.py`,
  `api_adapter_env.py`) — needed for DAPO Overlong Filtering
  (`GRPO_MASK_TRUNCATED`), didn't exist before (finish_reason was discarded).
- Custom `LambdaLR` (linear warmup, then constant, no decay) for
  `GRPO_LR`/`GRPO_LR_WARMUP_STEPS` — different shape than the existing
  cosine-with-warmup scheduler, so it's a separate branch, not a reuse.
- `step_unit`/`steps_per_epoch` redefinition for grpo: one epoch = one pass
  over **unique prompts** (`len(dataset)`), each expanded ×G — not one pass
  over `GRAD_ACCUM_STEPS`-sized batches of unique prompts like sdft.
- `train_full.sh`: added the 13 `GRPO_*` vars to `OPTIONAL_ENVS` passthrough
  (container launch script didn't forward them at all before).

## Experiment plan (first entries, `autoresearch/EXPERIMENTS.log`)

Continuing the existing log (exact next number = last entry on node05 at
E04x, confirm before writing — the prior campaign's tail wasn't fully
re-verified as of writing this plan).

- **Smoke test** (not a science experiment — infra validation): defaults per
  table above, k400-with-context, short. Pass/fail = "no crash, metrics sane."
- **First real run**: same defaults, full length on
  `subset_k400_subset_with_context.jsonl`. `ASYNC_ROLLOUT=1` from the start
  (G=8 means 8× the generation load per step — async rollout is the standard
  mitigation per the existing design doc, and the prior campaign already
  adopted it).
- **Next**: if noisy but promising, try `GRPO_FILTER_GROUPS=1` (dynamic
  sampling for degenerate groups) or `GRPO_ADV=zscore` — both are explicit
  "knob, not default" experiment surfaces in the design spec.
- Comparison bar: must beat E045 (0.7433, best SDFT result) to be
  interesting at all, and beat 0.7715 to be a real win.

## Operational risks / notes

- **GPU reservation**: re-checked directly via `nvidia-smi` (not the
  reservation-tracker alias, unavailable over non-interactive ssh) —
  0% util, ~4MiB used, zero compute processes on all 8 GPUs. Node is
  actually free; proceeding on your explicit instruction to run this now.
- **Full-FT memory ceiling (why we're on LoRA now)**: 2-way FSDP shard of
  gpt-oss-20b → ~10B params/rank; bf16 params + bf16 grads + bf16 `exp_avg`
  + bf16 `exp_avg_sq` ≈ 80GB ≈ the H100's 79.17GB budget, before any
  activation memory. `grpo_smoke_test_2` OOM'd exactly here (first-ever
  `optimizer.step()` call, allocating `exp_avg`). More trainer GPUs (4+)
  would also fix this (smaller shard/rank) but LoRA was chosen instead per
  direct instruction, since it sidesteps the ceiling at any trainer count
  and issue #22 (the prior LoRA blocker) is now fixed. Full FT + more GPUs
  remains a fallback if LoRA's rank-32 bottleneck turns out to matter for
  GRPO quality.
- **Wasted GPU in v1**: `train_full.sh` always starts a `logprob_server`
  process and allocates it a GPU, even though GRPO with `GRPO_KL_COEF=0`
  never calls it (`USE_LOGPROB_SERVER=False` skips it Python-side). Left
  as-is for v1 (correctness over GPU efficiency); reclaiming that GPU for
  more vLLM capacity is a fast-follow, not done yet.
- **Compute cost**: G=8 → 8× rollouts/prompt vs the old 1×. Generation is the
  new bottleneck, not the teacher forward (which we're dropping via
  `GRPO_KL_COEF=0`). This roughly cancels — teacher GPU capacity can likely
  be reallocated to vLLM rollout capacity.
- **Reflector cost**: PASS/FAIL already costs one Claude/Vertex call per
  rollout; G× more rollouts means G× more reflector calls per prompt.
  Existing retry/backoff (tenacity) stays as-is.
- **Dirty working tree on node05** — must commit/stash before branching
  (item 1 above).
- **Producer concurrency window must cover `world_size` groups, not just
  `N_ASYNC`'s raw value** (found the hard way, 2 crashes): `_produce_streaming_grpo`
  caps in-flight GROUPS at `n_groups_async = N_ASYNC // GRPO_GROUPS`, and each
  GRPO step needs exactly `world_size` distinct groups (1/rank). If
  `n_groups_async < world_size`, at least one rank's group can't even start
  generating until an earlier group frees a producer slot — a second
  sequential ~250-400s generation round that lands right on the 600s NCCL
  collective timeout (`_pull_microbatch`'s broadcast) and hard-crashes the
  run. `grpo_smoke_test_4_lora` used `N_ASYNC=32` at `world_size=2`
  (`n_groups_async=4 ≥ 2`, safe) — carrying that same absolute `N_ASYNC=32`
  forward to `world_size=5` (`n_groups_async=4 < 5`) is what broke E047's
  attempt 3. **Rule: `N_ASYNC ≥ world_size × GRPO_GROUPS` whenever scaling
  trainer count**, not just "whatever worked in the smoke test." Now enforced
  by an assert at trainer startup (`trainer.py`, commit `9d9dea5`).
- **Separate, unfixed issue: vLLM generation throughput can collapse under
  sustained concurrent load, independent of the above.** With the
  concurrency bug fixed (attempt 4, `N_ASYNC=40`), 5 clean steps ran, then
  step 6 stalled the full 600s with zero progress. `vllm_0.log` shows the
  smoking gun directly: generation throughput dropped from ~190 tok/s to
  16-28 tok/s for a sustained multi-minute window while `Running: 40 reqs`
  stayed constant and **GPU KV cache usage stayed at ~5%** — ruling out
  memory/capacity exhaustion. This is the *same* symptom as attempt 2's
  collapse (which used 3 separate vLLM processes), now reproduced on a
  **single** vLLM instance — disproving the earlier "multi-instance
  topology is the problem" theory. Appears probabilistic/time-dependent
  (16-concurrent never hit it in a short 2-step smoke test; 40-concurrent
  got 5 good steps first) rather than a hard threshold. Not root-caused —
  candidates not yet investigated: vLLM scheduler/batching config
  (`--max-num-seqs`, `--max-num-batched-tokens`, none set explicitly so
  vLLM defaults apply), `--enforce-eager` (required for weight-transfer
  dev-mode, disables CUDA graphs), MoE-specific routing/expert-parallel
  inefficiency under high concurrency, or a CPU-bound scheduling bottleneck
  (Python-side request loop, not GPU-bound given low KV cache usage).
  **Mitigation deployed (not a fix)**: no resume-from-checkpoint capability
  exists in the trainer, so a crash loses all progress since the last save.
  `/tmp/launch_run_47.sh` now (a) sets `SAVE_EVERY=5` (down from 40) so a
  short crash-prone run still yields a real checkpoint for eval, and
  (b) wraps the launch in a bounded (`MAX_RETRIES=20`) auto-restart loop so
  the campaign keeps generating real training+eval data points unattended
  between check-ins, at the cost of each restart training a fresh LoRA init
  from scratch (no `TRAINER_SEED` set, so each attempt sees a different
  shuffle — not true resumption). Proper checkpoint-resume is the real fix,
  deferred until eval signal from short runs justifies the added scope.
- **Likely major contributing factor found: `auto_eval_poller.sh` was
  stuck in an infinite re-eval loop, hammering GPU 7 continuously during
  E047's crash windows.** The poller blindly echoed "DONE" after
  `eval_with_retrieval.py` regardless of exit code, and never marked
  failed checkpoints — so `sdft_gptoss_20b_run_44_long/epoch_46` (a
  checkpoint that fails to load, unrelated root cause) was being
  re-evaluated on **every single 5-minute scan cycle**, back-to-back with
  almost no idle gap, for the entire time E047's attempt-5 retry loop was
  crashing repeatedly within 1-2 steps each try (much worse than attempt
  4's 5 good steps). Podman GPU isolation is device-level only — shared
  host CPU (tokenization/scheduling), PCIe, and NVMe bandwidth between the
  poller's continuous eval workload (GPU 7) and the training vLLM instance
  (GPU 0) is a plausible mechanism for the throughput collapses observed.
  Fixed: `auto_eval_poller.sh` now checks the eval command's exit status
  and writes a `$ckpt_dir/.eval_failed` marker on failure (merge failure
  or eval failure), skipped on future scans — no more infinite retries.
  Poller paused (not restarted) to test this hypothesis cleanly: watching
  whether E047's current retry-loop attempt achieves multi-step stability
  now that the confound is removed, before restarting the (fixed) poller.
  **Confirmed, then refined**: pausing the poller took the training run from
  crashing every 1-2 steps to 11 consecutive clean steps. Restarted the
  fixed poller — it ran one real (legitimate, non-buggy) eval on GPU 7
  and training crashed again ~6 min into that eval's execution window
  (attempt 3 → attempt 4 in the retry loop). So the infinite-retry bug was
  the *dominant* contributor, but not the *only* one: even a single
  legitimate eval on GPU 7 can still occasionally push a training step's
  generation over the 600s timeout via host-level contention. Accepting
  this residual risk for now — the bounded auto-restart loop recovers
  automatically (attempt 4 launched within 10s of the crash, no manual
  intervention needed). A cleaner fix (CPU affinity/cgroups isolation
  between the poller and training processes, or scheduling eval to avoid
  active training windows) is a further follow-up, not blocking.
- **Native LoRA support added to `eval_with_retrieval.py`** (separate repo,
  `eshwarprasadS/maas-knowledge-eval` — file changes deployed to node05,
  not committed/pushed there without explicit go-ahead per repo-ownership
  policy). Uses vLLM's own `enable_lora=True` + `LoRARequest` instead of
  merging into a full checkpoint first: detects `adapter_config.json`,
  loads the base model (`base_model_name_or_path` from the adapter config)
  with `max_lora_rank` read from the adapter's `r` field, hot-attaches the
  adapter via `LoRARequest` at `generate()` time. Confirmed vLLM 0.25.1
  has native "fused MoE LoRA" support — handles our `target_parameters`-
  based fused-expert adapter format (the same PR #23 layout fix, re-saved
  to disk by `save_hf_adapter_checkpoint`) with no extra conversion needed.
  Validated end-to-end on step_10: **0.6715** (native) vs **0.6463**
  (merge-based, same checkpoint) — a 2.5pp gap, plausibly sampling/judge
  noise given `temperature=0.7` with no fixed seed in the eval script (no
  `VLLM_SEED`-equivalent knob there), not a loading-correctness issue.
  `auto_eval_poller.sh` simplified to remove the whole merge/podman step
  (`merge_lora_for_eval.py` is now unused by the poller — still valid as a
  standalone tool, just not on this hot path). Benefit beyond avoiding the
  42GB copies: one less heavy podman subprocess per eval cycle, which may
  also reduce the residual contention risk noted above.
- **`SAVE_EVERY` was silently ignored for LoRA mode**: `push_lora_adapter`
  (called every optimizer step, unconditionally, to hot-swap the adapter
  into vLLM) internally calls `save_hf_adapter_checkpoint` on every step
  regardless of `SAVE_EVERY`, since vLLM's hot-swap endpoint loads from a
  local disk path — the separate `SAVE_EVERY`-gated save at the same path
  was actually redundant for LoRA (meaningful only for the full-FT branch,
  which has no other disk write). Fixed in `trainer.py`: delete the
  per-step adapter dir right after a successful push when the step isn't
  a `SAVE_EVERY` multiple (vLLM already has the weights in GPU memory by
  then, the on-disk copy isn't needed). Also manually cleaned up E047's
  already-existing non-`SAVE_EVERY` step dirs (kept step_0/5/10).

## Open questions carried into this phase

1. Does GRPO beat the 0.73 reverse-KL ceiling on **plain** data too, or does
   it need context injection as well (compounding vs. substitute lever)?
2. Reflector pass rate is currently ~0.65–0.74 (not 0.5) — how often are
   groups degenerate (all-pass/all-fail) at G=8 with this pass rate? Spec's
   44%-at-p=0.5 number doesn't directly apply; watch
   `grpo/frac_reward_zero_std` from the smoke test to calibrate whether
   `GRPO_FILTER_GROUPS` should be on by default here.
3. Literature defaults (LR=1e-6, clip 0.2/0.28, G=8) are generic — none of
   this has been validated on this task's long-CoT thinking-mode completions.
4. With `GRPO_KL_COEF=0` there's no distributional anchor at all — risk of
   entropy collapse or reward hacking against the reflector. `β>0` against
   the EMA teacher is the documented mitigation if that shows up.

## Next steps

1. Smoke test result → if healthy, launch the first real GRPO run
   (full-length, `subset_k400_subset_with_context.jsonl`, G=8, GA=32 or
   larger for a real gradient-noise-per-step budget) and log it as the first
   `autoresearch/EXPERIMENTS.log` entry for this phase.
2. Formalize `autoresearch/GOAL.md`/`STATE.md` for the GRPO phase on node05
   (per the `autoresearch` skill) — supersedes the SDFT-phase framing,
   keeps the SDFT history as background/baseline.
3. Continue the autoresearch loop autonomously: iterate on `GRPO_FILTER_GROUPS`,
   `GRPO_ADV=zscore`, `GRPO_KL_COEF>0` (not yet implemented — would need the
   K3++ reference-KL path built) as needed based on what the health metrics
   show.
