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
    "normalized_messages": list[list[dict]],  # normalized trajectory (logging/reflector only)
}
```

Only `prompt_texts` / `conditional_texts` participate in training.

## Current behavior (gpt-oss — kept as-is, flag-gated)

- `_normalize_messages`: converts WildChat `from/value` → `role/content`
- `IS_GPT_OSS` flag (derived from `MODEL_NAME`) gates `_append_analysis_channel`: appends `<|channel|>analysis<|message|>` to the generation prompt to force the gpt-oss model into its thinking channel
- `HINDSIGHT_TEMPLATES`: `user_response`, `enriched_user_response`, `online_feedback` (latter defers privileged text to the env after rollout)
- Renders via `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` with gpt-oss's harmony template

This path is **not removed** — both paths coexist, selected by `IS_GPT_OSS`.

## Added: Qwen path (Qwen3-8B, default)

### Qwen3 specifics (validated)

- **Native multi-call tool calling** — all N `<tool_call>` blocks render inline in one assistant message (gpt-oss's template hardcodes `tool_calls[0]`, silently dropping calls 2..N; Qwen3 needs no splitting/custom rendering)
- **Generation prompt** with `enable_thinking=False` renders `<|im_start|>assistant\n<think>\n\n</think>\n\n` — official template quirk, identical string at train and serve time
- **Tool defs ARE included in the prompt** — the model must know its tool inventory to call the right tool with the right arguments. Passed via `tools=` to `apply_chat_template`, which renders them as a `<tools>` JSON block in the system message. Loaded from `tool_defs.json` next to the train dataset (`dirname(TRAIN_DATA_PATH)`); if absent, `tools=None` and the block is skipped.
- **Stripped tool defs** — description fields are dropped (top-level + per-property), keeping names, parameter types, enums, defaults, required. Full OLS defs cost ~7,214 tokens; stripped cost **~2,332 tokens** (validated). Rationale: descriptions in the OLS defs are long (28 tools), and the token budget matters more — the trajectory's tool calls demonstrate correct usage.

### New-format dataset support (OLS, OpenAI-style)

1. **`_normalize_messages` extension** — tool messages `{'role': 'tool', 'tool_results': [...]}` → `{'role': 'tool', 'content': json.dumps(tool_results)}` (role stays `tool`). Required: without it Qwen3 renders an **empty** `<tool_response>` and the results silently vanish (256 vs 3,603 tokens for item 1, validated).
2. **`raw_questions` fix** — `clean_prompt[-1]` may be a tool message in the new format; use the last `role == "user"` message.
3. **Target text helper** — hand-formatted (no template render, no tokenizer round-trip): one `name(args_json)` line per tool call + `content` (newline-joined). Avoids the template's default-system boilerplate and special-token stripping artifacts (validated: template render glues words at stripped markers and prepends ~80 tokens of identity text).
4. **`golden_answers`** — same helper output; **`normalized_messages`** — the normalized trajectory.
5. **Privileged hint placement** — the correct-answer hint is appended as a **new `user` message** at the end of the trajectory when the last message is not a user message (new format: last message is a `tool` message → renders as its own clean `<|im_start|>user` block right before the generation prompt, after all tool results); when the last message **is** a user message, the hint merges into its content (old WildChat format — unchanged behavior).

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
| Reference: full (unstripped) tools block | +7,214 tokens/prompt → 17/31 > 14,336, 8/31 > 16,384 (rejected) |

**Budget note:** stripped defs fit the current limits (`STUDENT_MAX_PROMPT_LEN=14336`, `MAX_TOTAL_LEN=16384`) with a small tail — 1/31 prompts and 2/31 conditionals slightly exceed 14,336, none exceed 16,384. Options: skip/truncate the few long items, or nudge limits to ~15k/17k pending teacher forward-pass validation.

## Constraints

- **No thinking hardcoded, except for oss.** The gpt-oss path may force the analysis channel (`_append_analysis_channel`); the Qwen path must not hardcode thinking — `enable_thinking` stays config-driven (default `False`).
- **New-format paths are Qwen-only.** Tool defs (`tools != None`), `tool_calls`, and `tool_results` are only validated for Qwen-family models. If any of these are present and the model is not Qwen, **raise an error** stating this path is not validated — gpt-oss currently fails silently or with obscure jinja errors on this shape (drops calls 2..N, crashes on `content|tojson`).

## Proposed changes to `megatron_trainer/collator.py`

- Keep gpt-oss path untouched (behind `IS_GPT_OSS`)
- Extend `_normalize_messages` with the tool branch (`tool_results` → `content=json.dumps(...)`)
- Add `_target_text(user_response) -> str` (hand-format)
- `raw_questions` → last user message (not `clean_prompt[-1]`)
- Load `tool_defs.json` from `dirname(TRAIN_DATA_PATH)` (absent → `None`), strip descriptions, pass `tools=` on both renders (Qwen path)
- Add family guard: `tools is not None` OR any `tool_calls`/`tool_results` present AND model not Qwen → `ValueError` ("path not validated for this model family")
- Privileged hint placement: merge into last message's content if `role == "user"`, else append a new `user` message (used for `conditional_texts`)
