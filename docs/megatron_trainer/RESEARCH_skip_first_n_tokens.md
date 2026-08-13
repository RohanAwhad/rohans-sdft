# Token Skip Mask (skip first N completion tokens) — Deep Research Report

> Generated: 2026-08-12 | Sources: 12 web + 8 code
> Topic: issue #7 — "Token Skip Mask - skips first N completion tokens from loss. Avoids noisy early-token KL."

## TL;DR

- **"Noisy early-token KL" is folklore, not an established claim.** No mainstream RLHF/KD trainer (TRL PPO/GRPO/DPO, verl, OpenRLHF, Megatron) masks the first completion token. The SDFT paper's actual rationale for skipping the first tokens is **artifact suppression** (student echoing demo-preface phrases like "Based on the text…"), and the paper calls it "fundamentally a heuristic fix."
- **The OPD evidence points the opposite direction**: teacher signal is *strongest* at early positions (+0.37 advantage at 1K prefix → +0.02 at 16K, arXiv:2604.13016 §6.1); instability/entropy spikes originate at the **suffix** and propagate backward. Noise is at the end of the rollout, not the start.
- **The feature exists in exactly one lineage**: idanshen/Self-Distillation (`num_loss_tokens_to_skip`, `main.py` passes 3; config default 0), its forks, and TRL's experimental SDFT trainer (default 0). Tinker Cookbook hard-codes 3 "matching the reference implementation." No framework outside SDFT implements it; no paper reports numbers for it.
- **In OUR setup the fix is mode-dependent**: with `STUDENT_THINKING=1` (and all api_adapter runs) the first N tokens are `<think>` + early reasoning — the highest-value teacher-signal region. Skipping there targets the signal, not noise. With `STUDENT_THINKING=0` the first tokens are the preface/content region the paper's fix targets, but also the sharpest direct-answer signal. Either way a prefix skip is an ablation, not a default.
- **First-token distributions are degenerate** (near-deterministic template tokens) → their KL contribution is ≈0, so small N is *harmless* but also *nearly pointless*; the literature's honest reading: implement it as a flag, default 0, and only enable if eval artifacts are observed.
- **Implementation is well-understood**: the mask must live inside `ChunkedRowKL.forward` (the Function returns a scalar row-sum; the backward sees only scalar `grad_kl`, so a per-row mask cannot be injected at the processor level). IS weighting couples via NaN-ing the skipped prefix (already supported by `compute_is_weight`'s NaN handling); metrics must be masked for A/B comparability.

## Overview

`megatron_trainer/chunked_head.py` computes a full-vocab reverse-KL per completion row: `loss = kl_sum / C` (row-mean over all C completion tokens, chunked_head.py:170). Issue #7 proposes excluding the first N completion rows from the loss on the grounds that early-token KL is noisy.

This report audits (1) what the SDFT lineage actually says and does about skipping first tokens, (2) what the broader RLHF/KD/OPD literature says about early-position loss masking, (3) whether the artifact mechanism applies to our privileged-info teacher setup, and (4) exactly how the mask would be implemented in `ChunkedRowKL` with the autograd constraints verified against the code.

## Key Findings

### 1. The SDFT lineage: rationale is artifact suppression, not noise

SDFT paper (arXiv:2601.19897v2, §5 "Learned Artifacts") — verbatim:

> "A subtle failure mode of our approach is that the student can inherit spurious linguistic patterns from the teacher. Because the teacher is conditioned on demonstrations or text passages, it may produce responses prefaced with phrases like 'Based on the text…' or 'Following the example…' The student, although receiving no such context, sometimes nevertheless reproduces these markers, having learned them as part of the teacher's output distribution. Empirically, we find that masking the loss over the first few tokens during training effectively suppresses these artifacts without harming downstream accuracy. While this workaround is effective in practice, it is fundamentally a heuristic fix."

Implementation facts:
- idanshen/Self-Distillation: `distil_config.py:612-619` default **0**, help text "initial tokens of the response, which may be less predictable"; `main.py:129` passes **3**; mask construction `distil_trainer.py:1594-1606` (`completion_mask * (token_positions >= N)`); normalization `((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()` at `:1687`. Attention mask untouched — "We need to keep the original for attention."
- TRL experimental SDFT (`trl/experimental/sdft/sdft_trainer.py:946-950`): same mask, default 0, comment "to suppress teacher-prompt artifacts."
- Tinker Cookbook (`tinker_cookbook/distillation/sdft.py:296, 399-400, 559`): `skip_first_n_tokens=3` hard-wired in both forward-KL and reverse-KL datum builders, "reference skips 3"; raw sum, no renormalization. Not applied in the deprecated topk=0 path.

The paper never says early-token KL is noisy; it gives no N; there is no ablation table. The repo help-text "less predictable" is the only phrasing close to the issue's premise.

### 2. No one else does this, and the OPD literature says the noise is at the suffix

Framework audit (all verified in raw source):
- TRL PPO v0.7.1→v0.24, GRPO, DPO: first completion token always in loss/advantages (`ppo_trainer.py:955-974` shift logic keeps it; GRPO `completion_mask` includes it).
- verl: `response_mask` includes EOS; GAE carries through masked positions (`core_algos.py:216-266`); no first-token skip. First-token value-baseline is a known confusion class (verl issue #1960, unanswered).
- OpenRLHF, DeepSpeed-Chat, Megatron-LM (pretrain + KD `cached_logits_loss.py`), litgpt, axolotl, unsloth: no positional first-token skip anywhere.

Counter-evidence to the "early = noisy" premise:
- **Rethinking OPD** (arXiv:2604.13016 §6.1): teacher advantage falls monotonically from **+0.37 at 1K prefix to +0.02 at 16K**; "high entropy first appears at the end of the response and progressively propagates toward earlier tokens." Early positions = most reliable supervision; suffix = noise.
- **MiniLLM** (arXiv:2306.08543): "the error in the front tokens accumulates along the whole sentence"; per-token importance-weight variance accumulates with position → early positions are the lowest-variance part of the estimator.
- **First-token distributions are degenerate**: near-deterministic template mass (e.g., "Sure", "Here", "Based on") — near-zero KL contribution. Masking is harmless but removes almost nothing.
- **Smaug/DPOP** (arXiv:2402.13228) treats under-incentivized early tokens in DPO as a *failure to fix*, not a feature.
- **Revisiting OPD** (arXiv:2603.25562): their masking win is **special-token** masking (+4.3 math score, tokenizer-mismatch), not positional prefix masking; their noise analysis locates "over-continuation" filler in mid/suffix.

### 3. Does the artifact mechanism apply to OUR teacher setup? Partially, and mode-dependent

Our teacher IS conditioned on privileged information — docs + golden answer (collator.py:51-62), or chunk + golden + reflection feedback (rag_env.py:15-27), or adapter history + env feedback (api_adapter_env.py:72-80). Same family as the paper's demonstration-conditioned teacher, so preface-echo risk is structurally plausible ("Based on the documentation…", "The correct answer is…").

Two nuances:
1. Our hint is an explicit *instruction* + labeled data, not an in-context *demonstration* — the paper's "Following the example"-type echo is weaker.
2. **The artifact and the knowledge signal are entangled**: "The correct answer is…" is both a preface and the knowledge transfer we want.

What "skip first N" actually cuts in our rollouts (Qwen3-8B, verified against tokenizer_config.json + code):
- **STUDENT_THINKING=1** (and all api_adapter, which hardcodes thinking): first tokens are `<think>`(151667) + early reasoning. This is the region where teacher advantage is largest per 2604.13016. The paper-style prefaces live *after* `</think>` — a prefix skip misses the artifact and hits the signal.
- **STUDENT_THINKING=0** (Qwen3 default, empty think block pre-seeded in the prompt): completions start directly with content ("Based on…", "Here is…", "The answer…"). The paper's fix maps cleanly, but this is also the sharpest direct-answer region for non-CoT models.
- api_adapter verdict tokens (`<|VERDICT_START|>`, PASS/FAIL) are deep in the completion, not at the start.

No artifact handling exists anywhere in our codebase today (grep-verified); completions are stored raw (`rag_env.py:61-63`, `skip_special_tokens=False` in vllm_utils.py:69), so template tokens like `<think>` are trained on.

### 4. Implementation: the autograd constraint is real, and the mask goes inside the Function

Verified against chunked_head.py:
- `ChunkedRowKL.forward` returns the row-**SUM** `kl_row.sum()` (chunked_head.py:100); backward receives only a scalar `grad_kl` and applies `g = grad_kl * p * (A - kl_row)` uniformly per row (chunked_head.py:109).
- Because `grad_kl` is scalar, a **per-row** mask cannot be injected at the processor level — a chunk-aligned per-chunk scalar would work, but a prefix mask straddles row chunks (row_chunk=128), and the divisor must change from C to C_eff or unmasked rows get mis-scaled gradients. Correct fix: pass `row_mask` as a 4th arg into `ChunkedRowKL.apply`, mask the row-sum in forward, multiply the backward by `row_mask.unsqueeze(1)`.
- Masked backward math (verified): `dL/dz_{r,u} = (w/C_eff) · mask_r · p_{r,u}(A_{r,u} − kl_row_r)`. Masked rows get exactly zero gradient; unmasked rows keep the exact total-derivative formula; the per-row `-kl_row` constant has no cross-row coupling (each row is an independent logsumexp).
- IS-weighting coupling: NaN-ing `rollout_log_probs[:N]` excludes skipped rows from the IS mean for free (`valid = ~torch.isnan(...)`, chunked_head.py:46); the length-mismatch ValueError compares `size(0)` only — no interaction.
- Metrics (`sdpo/signal_mean` etc., chunked_head.py:183-199) are means over all C rows; unmasked metrics break A/B comparability. `sdpo/len_signal_mean` divides by literal C (:188).
- Guards: C is variable per microstep; `skip_first_n ≥ C` → warn + skip the microstep in trainer.py (next to the existing empty-completion skip at trainer.py:340-342), and `C_eff = max(C − N, 1)` defensively.
- Dead code found: `KL_CHUNK=2048`/`ROW_CHUNK=128` constants (chunked_head.py:24-25) unreferenced; `loss_mask` kwarg (chunked_head.py:143) accepted but never used; stale docstring at chunked_head.py:16-19 ("1/C is inside this Function" — it's in the processor at :170).

### 5. What the literature actually recommends for "noisy token KL"

If the goal is noise suppression in token-level KL, the evidence-backed knobs are (all stronger than a positional prefix skip):
- **Entropy-based masking** (TRL GRPO `top_entropy_quantile`) — masks high-entropy positions; matches the suffix-noise finding.
- **Special-token masking** (2603.25562: +4.3 score) — mask tokenizer-mismatch tokens (relevant for our gpt-oss channel tokens).
- **Length control** (2604.13016: 3K-7K optimal response length; our GEN_MAX_NEW_TOKENS=6144 sits in that range).
- **Teacher top-K support matching** (2603.25562, +19.8% — see issue #5 research).

## Practical Guide

Recommendation (default N=0, ablation if artifacts appear):

1. Implement `SKIP_FIRST_N` (env, default `"0"`) → config.py → trainer.py → `make_kl_processor(skip_first_n=...)`:
   - `row_mask` built once per processor (`torch.arange(C) >= N`, fp32); passed as 4th arg to `ChunkedRowKL.apply`.
   - forward: `kl_masked = kl_row * row_mask`; return `kl_masked.sum()`; save `row_mask`.
   - backward: `g = grad_kl * p * (A - kl_row.unsqueeze(1)) * row_mask.unsqueeze(1)`.
   - processor: `C_eff = max(C - skip_first_n, 1)`; `loss = kl_sum / C_eff`; NaN `rollout_log_probs[:N]` before `compute_is_weight`; mask metrics (or log both masked + full).
   - trainer: `if len(completion_ids) <= SKIP_FIRST_N: warn + continue`; wandb config gains `skip_first_n`; train_full.sh passthrough line.
2. Don't skip by default. If preface artifacts are observed on eval outputs ("Based on the documentation…" from a non-thinking run):
   - `STUDENT_THINKING=0`: ablation N∈{8,16} — removes the opening phrase region.
   - `STUDENT_THINKING=1`: don't use a prefix skip; if artifacts exist, they're post-`</think>` — consider masking that specific region, or entropy masking.
3. Numerical test (net-new; no chunked_head tests exist today): masked rows → exactly zero grad; unmasked rows match `grad_kl · p · (A − kl_row)` vs finite differences; non-multiple-of-128 N; `N ≥ C` guard.

## Gotchas & Pitfalls

1. **The issue's premise is wrong as stated** — the paper's reason is artifact suppression; no source supports "early-token KL is noisy." The noise evidence points to the suffix. Justify any enablement with observed eval artifacts, not noise.
2. **Thinking mode reverses the tradeoff** — skipping first N in thinking mode removes `<think>` + early reasoning = the region with the strongest teacher signal. Worst place to mask.
3. **Mask must be inside the autograd Function** — processor-level masking either can't express a straddling prefix mask or silently mis-scales gradients (C vs C_eff). Don't mask `loss` post-hoc.
4. **IS weight and metrics must be masked in sync** — unmasked IS weight or `signal_mean` mixes skipped rows in and breaks A/B comparisons (`len_signal_mean` divides by C literally).
5. **C ≤ N is a live edge case** — C varies per microstep; guard with warn+continue (trainer) and `max(C−N,1)` (processor). Skipping microsteps amplifies the pre-existing `is_final`/no_sync index issue (trainer.py:423) — perf-only, `finish_grad_sync()` keeps it correct.
6. **Raw completions include template tokens** — `<think>`, `<|im_end|>` are trained on today (`skip_special_tokens=False`); any skip mask should only remove plain content tokens, never structural tokens or EOS (EOS carries stop-signal).
7. **N=3 is de facto, not principled** — it comes from one launch script (idanshen main.py); config default is 0 everywhere. There is no study of N anywhere.

## Hardware / Environment Considerations

- No hardware impact: the head GEMM runs once regardless (chunked_head.py:156-157); the teacher logprob server still computes the full (C,V) tensor even if the first N rows are loss-masked.
- C is variable per microstep (up to GEN_MAX_NEW_TOKENS=6144); row_chunk=128.
- api_adapter completions end with `<|im_end|>` and contain force-close `</think>` inserts with None log-probs (already NaN-masked in IS) — a prefix skip doesn't interact with those.

## Sources

1. SDFT paper — arXiv:2601.19897v2 §5 "Learned Artifacts", §3 — https://arxiv.org/html/2601.19897v2
2. idanshen/Self-Distillation — distil_trainer.py:1594-1606, 1687; distil_config.py:612-619; main.py:129 — https://github.com/idanshen/Self-Distillation
3. TRL experimental SDFT — trl/experimental/sdft/sdft_trainer.py:946-950, 1012-1013; sdft_config.py:425-428 — https://github.com/huggingface/trl
4. Tinker Cookbook — tinker_cookbook/distillation/sdft.py:296, 328-329, 399-400, 559 — https://github.com/thinking-machines-lab/tinker-cookbook
5. Rethinking OPD — arXiv:2604.13016 §6.1, App. B.1/D.1 — https://arxiv.org/html/2604.13016v2
6. Revisiting OPD — arXiv:2603.25562 (special-token masking, noise analysis) — https://arxiv.org/html/2603.25562v2
7. MiniLLM — arXiv:2306.08543 (front-token error accumulation) — https://arxiv.org/abs/2306.08543
8. Smaug/DPOP — arXiv:2402.13228 (first-token incentive analysis) — https://arxiv.org/abs/2402.13228
9. TRL PPO v0.7.1/v0.24, GRPO, DPO — ppo_trainer.py:955-974; grpo_trainer.py:1910-1914, 3194-3225 — https://github.com/huggingface/trl
10. verl — torch_functional.py:349-369 (response_mask); core_algos.py:216-266 (GAE); issue #1960 — https://github.com/verl-project/verl
11. OpenRLHF — models/actor.py:294; models/utils.py:121-122 — https://github.com/OpenRLHF/OpenRLHF
12. Megatron-LM — gpt_dataset.py:246-252; cached_logits_loss.py:816-876 — https://github.com/NVIDIA/Megatron-LM
13. This repo — chunked_head.py:24-25, 28-67, 70-110, 113-203; trainer.py:337-343, 347-368, 372-381, 402-427; collator.py:51-62, 100-131, 314-389; rag_env.py:15-27, 61-63, 86-106; api_adapter_env.py:72-80, 306-358; vllm_utils.py:69; config.py:29-35, 72-73; train_full.sh:114-115
