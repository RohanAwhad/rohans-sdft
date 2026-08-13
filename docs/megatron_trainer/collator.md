# Collator — On-Policy SDFT Prompt Construction

## Role

The collator is the only component that renders model inputs. For each dataset example it produces two prompt variants:
- **Student prompt (x)** — question only, sent to vLLM for generation
- **Teacher prompt (x, o)** — question + privileged info, scored by the (frozen) teacher

Completions are generated on-the-fly by vLLM, so the collator never touches targets.

## Output contract (unchanged)

```python
{
    "prompt_texts": list[str],        # student context
    "conditional_texts": list[str],   # teacher context (x + privileged o)
    "raw_questions": list[str],       # last user message content (logging/reflector only)
    "golden_answers": list[str],      # plain-text target (logging/reflector only)
    "normalized_messages": list[list[dict]],  # truncated normalized trajectory (logging/reflector only)
}
```

Only `prompt_texts` / `conditional_texts` participate in training.

## Current behavior (gpt-oss — kept as-is, flag-gated)

- `_normalize_messages`: converts WildChat `from/value` → `role/content`
- `IS_GPT_OSS` flag (derived from `MODEL_NAME`) gates `_append_channel`: appends an explicit channel start to the generation prompt so gpt-oss picks a deterministic channel — `<|channel|>analysis<|message|>` when `STUDENT_THINKING=1` (thinking channel, model self-switches to final when done), `<|channel|>final<|message|>` otherwise (answer-only)
- `HINDSIGHT_TEMPLATES`: `user_response`, `enriched_user_response`, `online_feedback` (latter defers privileged text to the env after rollout)
- Renders via `apply_chat_template(add_generation_prompt=True, enable_thinking=STUDENT_THINKING)` with gpt-oss's harmony template

This path is **not removed** — both paths coexist, selected by `IS_GPT_OSS`.

## Added: Qwen path (Qwen3-8B, default)

### Qwen3 specifics (validated)

- **Native multi-call tool calling** — all N `<tool_call>` blocks render inline in one assistant message (gpt-oss's template hardcodes `tool_calls[0]`, silently dropping calls 2..N; Qwen3 needs no splitting/custom rendering)
- **Generation prompt** — `enable_thinking` driven by `STUDENT_THINKING` (default `False`): `False` renders `<|im_start|>assistant\n<think>\n\n</think>\n\n` (official template quirk, identical string at train and serve time); `True` renders a bare `<|im_start|>assistant\n` and the model thinks natively. Applies to **both** student and teacher renders (reverse-KL requires both sides to match).
- **Tool defs ARE included in the prompt** — the model must know its tool inventory to call the right tool with the right arguments. Passed via `tools=` to `apply_chat_template`, which renders them as a `<tools>` JSON block in the system message. Loaded from `tool_defs.json` next to the train dataset (`dirname(TRAIN_DATA_PATH)`); if absent, `tools=None` and the block is skipped.
- **Stripped tool defs** — description fields are dropped (top-level + per-property), keeping names, parameter types, enums, defaults, required. Full OLS defs cost ~7,214 tokens; stripped cost **~2,332 tokens** (validated). Rationale: descriptions in the OLS defs are long (28 tools), and the token budget matters more — the trajectory's tool calls demonstrate correct usage.

### New-format dataset support (OLS, OpenAI-style)

1. **`_normalize_messages` extension** — tool messages `{'role': 'tool', 'tool_results': [...]}` → `{'role': 'tool', 'content': json.dumps(tool_results)}` (role stays `tool`). Required: without it Qwen3 renders an **empty** `<tool_response>` and the results silently vanish (256 vs 3,603 tokens for item 1, validated).
2. **`raw_questions` fix** — `clean_prompt[-1]` may be a tool message in the new format; use the last `role == "user"` message.
3. **Target text helper** — hand-formatted (no template render, no tokenizer round-trip): one `name(args_json)` line per tool call + `content` (newline-joined). Avoids the template's default-system boilerplate and special-token stripping artifacts (validated: template render glues words at stripped markers and prepends ~80 tokens of identity text).
4. **`golden_answers`** — same helper output; **`normalized_messages`** — the normalized trajectory.
5. **Privileged hint placement** — the correct-answer hint is appended as a **new `user` message** at the end of the trajectory when the last message is not a user message (new format: last message is a `tool` message → renders as its own clean `<|im_start|>user` block right before the generation prompt, after all tool results); when the last message **is** a user message, the hint merges into its content (old WildChat format — unchanged behavior).

### Prompt truncation (budget enforcement)

- **Why**: vLLM hard-rejects prompts over the server context length (HTTP 400, prompt + gen > `--max-model-len`), and the trainer-side tokenizer silently tail-chops oversize prompts (`truncation=True` in `trainer.py`) — which cuts exactly the privileged hint. The collator enforces the budgets at render time instead, so the rendered strings always fit.
- **Budgets**: `prompt_texts` → `STUDENT_MAX_PROMPT_LEN`; `conditional_texts` → `TEACHER_MAX_PROMPT_LEN`. Measured on the **rendered** text (post-template, via the tokenizer with `add_special_tokens=False`) — the same tokenization the trainer and vLLM see.
- **Mechanism**: message-level only. Drop the **earliest** messages (after the system message) until the rendered token count fits. No partial-message cuts, no token-level slicing of rendered text (would risk breaking special tokens or the generation prompt). Overshoot on this dataset is small (≤ ~700 tokens), so 1–2 message drops on the few oversize items; no-op for the rest.
- **Protected — never dropped or truncated**:
  - **System message** — always complete (includes the `<tools>` block when tool defs are present).
  - **Privileged hint** (conditional only) — appended *after* body truncation, then a second pass protects it; the hint is always complete.
  - Everything else is droppable — **including the last user message (the question)**, dropped only if truly needed.
- **Invariant (drop, don't raise)**: max lengths must always be greater than the system-prompt render and the system + hindsight render — `render(system + tools) ≤ STUDENT_MAX_PROMPT_LEN` and `render(system + tools + hint) ≤ TEACHER_MAX_PROMPT_LEN`. Truncation therefore always fits by construction. A violation does **not** raise: the offending example is **dropped from the dataset** at load time (one-time startup filter before the DataLoader is built) and **never trained on**, with a `logger.warning` per drop plus a summary. No example is ever rendered over budget; examples that fit are guaranteed to be truncatable to budget. If the filter leaves the dataset empty, training cannot proceed — log an error and raise.
- Applies to **both paths** (Qwen + gpt-oss, and old + new format) — it is a no-op for old-format data, whose prompts are far below budget.

### Validation evidence (data/ols/train_sdft_mini.jsonl, 31 items)

| Check | Result |
|---|---|
| Render errors (with normalization + stripped tools) | 0/31 |
| Prompt tokens (x, with stripped tools) | 2,382–14,727, median 9,541 — 1/31 > 14,336, 0/31 > 16,384 |
| Conditional tokens (x,o, with stripped tools) | 2,490–16,024, median 9,627 — 2/31 > 14,336, 0/31 > 16,384 |
| Target tokens | 20–1,288, none empty; all 88 target tool calls preserved |
| Multi-call assistant messages | up to 9 calls/message render fully |
| Tool results | `json.dumps`'d list renders inside `<tool_response>` |
| Privileged append (new user turn when last msg isn't user; merge when it is) | renders cleanly, positioned after tool results, before generation prompt |
| Truncation: prompts > 14,336 | 0/31 after truncation (1 item drops tool-result messages) |
| Truncation: conditionals > 15,360 | 0/31 after truncation (2 items drop messages) |
| System message complete after truncation | 31/31 |
| Privileged hint complete after truncation | 31/31 |
| Protected-set drop filter (system ≤ 14,336; system+hint ≤ 15,360) | 0/31 dropped, 0 warnings at OLS budgets |
| Reference: full (unstripped) tools block | +7,214 tokens/prompt → 17/31 > 14,336, 8/31 > 16,384 (rejected) |

**Budget note:** stripped defs fit the budgets (`STUDENT_MAX_PROMPT_LEN=14336`, `TEACHER_MAX_PROMPT_LEN=15360`) — 1/31 prompts and 2/31 conditionals slightly exceed and are handled by collator truncation (message-level drop, system + hint protected). None exceed 16,384 pre-truncation. The server side must be launched with `--max-model-len ≥ budget + GEN_MAX_NEW_TOKENS` (e.g. 16,384 for 14,336 + 1,024), or vLLM hard-rejects (400).

## Constraints

- **Thinking is config-driven, per family.** `STUDENT_THINKING` (default `"0"`) selects the rendering for both Qwen and gpt-oss and applies to **both** student and teacher prompts (they must stay in sync for the reverse-KL). Qwen: `enable_thinking=STUDENT_THINKING`. gpt-oss: channel suffix `analysis` (thinking) vs `final` (answer-only). Only Qwen and gpt-oss families are validated with `STUDENT_THINKING=1` — `config.py` raises at import otherwise.
- **New-format paths are Qwen-only.** Tool defs (`tools != None`), `tool_calls`, and `tool_results` are only validated for Qwen-family models. If any of these are present and the model is not Qwen, **raise an error** stating this path is not validated — gpt-oss currently fails silently or with obscure jinja errors on this shape (drops calls 2..N, crashes on `content|tojson`).
- **Protected-set budgets are enforced by dropping, never by raising.** If `render(system + tools) > STUDENT_MAX_PROMPT_LEN` or `render(system + tools + hint) > TEACHER_MAX_PROMPT_LEN`, the example is dropped at dataset load with a warning and not trained on; a fully-dropped dataset is an error. This keeps runs alive on datasets with a few oversized hints (e.g. legacy WildChat/enriched data where the hint alone can exceed a small `TEACHER_MAX_PROMPT_LEN`).

## Proposed changes to `megatron_trainer/collator.py`

- Keep gpt-oss path untouched (behind `IS_GPT_OSS`); `_append_channel` selects the channel suffix from `STUDENT_THINKING`
- `_render_tokens` passes `enable_thinking=STUDENT_THINKING` (was hardcoded `False`)
- Extend `_normalize_messages` with the tool branch (`tool_results` → `content=json.dumps(...)`)
- Add `_target_text(user_response) -> str` (hand-format)
- `raw_questions` → last user message (not `clean_prompt[-1]`)
- Load `tool_defs.json` from `dirname(TRAIN_DATA_PATH)` (absent → `None`), strip descriptions, pass `tools=` on both renders (Qwen path)
- Add family guard: `tools is not None` OR any `tool_calls`/`tool_results` present AND model not Qwen → `ValueError` ("path not validated for this model family")
- Privileged hint placement: merge into last message's content if `role == "user"`, else append a new `user` message (used for `conditional_texts`)
- Add `_truncate_to_budget(messages, budget, protect_last=False) -> list`: message-level drop from the front (after system) until the rendered token count fits; greedy re-render per drop; `protect_last=True` for conditional (hint never dropped); applies to both paths, no-op within budget
- Replace the init/per-example budget asserts with a **drop filter**: `filter_dataset(dataset, rank)` runs once at load, `_drop_reason(ex) -> str | None` checks the protected-set renders against `STUDENT_MAX_PROMPT_LEN` / `TEACHER_MAX_PROMPT_LEN`, drops violating examples via `dataset.select(valid_indices)` and `logger.warning`s each drop (rank 0) plus a summary; empty filtered dataset → `logger.error` + raise
- Factor hint building into `_hint_for(ex) -> str | None` (None for `online_feedback`), reused by the drop filter and `__call__`
- `__post_init__` system+tools check → `logger.warning` (no crash); `normalized_messages` returns the **truncated** trajectory (what was actually rendered)
