# ApiAdapterEnv (`env/api_adapter_env.py`)

`ENV_TYPE=api_adapter` — multi-turn loop where the adapter (student, via vLLM)
vets an external API LLM's response with PASS/FAIL verdict + feedback; on FAIL
the API model regenerates. Training target: the adapter's last response
(verdict + feedback). Implements `BaseEnv` (see `base.md`).

## Constructor inputs

`ApiAdapterEnv(prompt_text, vllm_base_url, raw_question, golden_answer,
tokenizer, api_model=API_MODEL, max_adapter_turns=MAX_ADAPTER_TURNS,
success_cache=None)` (`env/api_adapter_env.py:95`):

- `success_cache` — rank-0 dict of previously successful adapter responses,
  keyed by `raw_question`; re-injected as hindsight on later failing rollouts.

## Rollout loop (`env/api_adapter_env.py:141`)

1. Call the API model with the user's question (`api_history`).
2. Build/append the adapter conversation (`ADAPTER_SYSTEM_PROMPT` + the API
   response wrapped in `<|USER_REQUEST_START|>`/`<|LLM_RESPONSE_START|>` blocks).
3. Loop:
   - Adapter responds (see below).
   - Parse verdict/feedback; unparseable → episode FAIL, stop.
   - `PASS` → return the API response (training target).
   - `FAIL` → append the feedback as a user turn, API regenerates;
     `turns_remaining -= 1`; stop when `MAX_ADAPTER_TURNS` exhausted.

## Adapter calls — thinking budget (`env/api_adapter_env.py:186`)

Two-phase generation via `apply_chat_template(... enable_thinking=True)`:
- **Phase 1**: `max_tokens=THINKING_BUDGET`. If `finish_reason != "length"`,
  done.
- **Phase 2** (truncated thinking): force-close `</think>` (append if missing),
  continue with `max_tokens=GEN_MAX_NEW_TOKENS − THINKING_BUDGET`, concatenate.

Each vLLM call returns its per-token log-probs (`logprobs=1`); the env tracks
`_rollout_segments` — `(text_segment, logprobs | None)` per call
(`env/api_adapter_env.py:196-229`). Segments whose text was **never sampled**
get `None` log-probs: the inserted force-close `.\n</think>\n\n` (phase 2) and
chat-template artifacts. These feed the importance-sampling weight (see
`chunked_head.md`).

## Parsing & evaluation

- `parse_adapter_response` (`env/api_adapter_env.py:240`): extracts
  `<|VERDICT_START|>` / `<|FEEDBACK_START|>` blocks (regex).
- `parse_model_answer` (`env/api_adapter_env.py:252`): extracts `\boxed{...}`
  from the API response; missing → FAIL.
- `evaluate` (`env/api_adapter_env.py:265`): exact string match of the boxed
  answer vs `golden_answer`.

## Training attributes (`generate_training_attrs`, `env/api_adapter_env.py:293`)

Builds the training triple by slicing the chat-template render:

- `full_text` = entire conversation, no generation prompt.
- `prompt_text` = everything up to and including `<|im_start|>assistant\n`
  (`apply_chat_template(... add_generation_prompt=True, enable_thinking=True)`).
- `completion_text` = `full_text[len(prompt_text):].rstrip("\n")` — starts at
  the actual content (including `<think>` if present) and includes `<|im_end|>`.
- `completion_log_probs` (`env/api_adapter_env.py:321-339`) — per-token rollout
  log-probs aligned to `completion_text` tokens: each segment's log-probs are
  sliced to that segment's token count; never-sampled tokens (inserted
  force-close text, `<|im_end|>`, stripped trailing `\n`) are `None`; the list
  is truncated/padded with `None` to exactly `completion_text`'s token count.
- `privileged_information_prompt` = prompt + `HINDSIGHT_TEMPLATE` (LLM response
  + env feedback) appended to the last user message. On failure with a cached
  successful response available, the cache text is appended as
  "=== CORRECT RESPONSE FROM ANOTHER ROLLOUT ===" (`env/api_adapter_env.py:347`).

Asserts the last `adapter_history` entry is `assistant`
(`env/api_adapter_env.py:301`).

## API calls / retry

External API via `litellm.completion(model=API_MODEL, ...)` with tenacity retry
(3 attempts, exp jitter) (`env/api_adapter_env.py:176`). `litellm.suppress_debug_info`
is set.

## Known gotchas

- The adapter history's last message must be `assistant` (asserted in
  `generate_training_attrs`).
- The cached correct response is injected only when the current rollout failed
  AND the question is in `success_cache`.
