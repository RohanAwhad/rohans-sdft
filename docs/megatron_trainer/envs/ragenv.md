# RagEnv (`env/rag_env.py`)

`ENV_TYPE=rag` — single-turn vLLM generation, optionally graded by the
reflector. Implements `BaseEnv` (see `base.md`).

## Constructor inputs

`RagEnv(prompt_text, vllm_base_url, privileged_information_prompt, raw_question,
golden_answer, normalized_messages, tokenizer, use_reflector=False,
golden_chunk="")` (`env/rag_env.py:23`):

- `privileged_information_prompt` — seeded from the collator's
  `conditional_texts` (static hint); in `online_feedback` mode it's **replaced**
  by the feedback-built prompt after generation.
- `raw_question` / `golden_answer` / `golden_chunk` / `normalized_messages` —
  fed to the reflector and the feedback template. `golden_chunk`
  (enriched_user_response value/content) is optional — the template's chunk
  section is omitted when empty.
- `use_reflector` — set by the trainer when `HINDSIGHT_FIELD == "online_feedback"`
  (`trainer.py:227`).

## `run()` (`env/rag_env.py:49`)

1. `vllm_generate(prompt_text, base_url)` → `(completion_text, finish_reason,
   token_logprobs)` — the per-token log-probs (requested via `logprobs=1`,
   requires the server's `--logprobs-mode processed_logprobs`) are stashed on
   `completion_log_probs` (`env/rag_env.py:50-52`).
2. If `use_reflector`: call `reflector.run(raw_question, golden_answer,
   completion_text)` → `{verdict, feedback}`, then rebuild the privileged
   prompt from the feedback (`_build_privileged_prompt_from_feedback`,
   `env/rag_env.py:60`).

## Privileged-prompt rebuild (`env/rag_env.py:60`)

Appends the feedback block to the last message, then renders via
`apply_chat_template(... add_generation_prompt=True,
enable_thinking=STUDENT_THINKING)` — the rebuild must match the collator's
render, so thinking is flag-gated here too.

Privileged content = **golden chunk (optional) + golden answer (required) +
reflection feedback**:

- `ONLINE_FEEDBACK_TEMPLATE` (chunk present): "Relevant documentation: … /
  Correct solution: … / The following is feedback from your earlier attempt: …"
- `ONLINE_FEEDBACK_NO_CHUNK_TEMPLATE` (no chunk): the golden answer + feedback
  sections only.

Examples with an **empty golden answer** are dropped at dataset load by the
collator (`filter_dataset`, `collator.py`) — reflection requires a golden to
grade against; a missing chunk is tolerated and ignored.

## Reflector (`reflector.py:56`)

`reflector.run(question, golden_answer, model_response) -> {verdict, feedback}`
via `AnthropicVertex` (`REFLECTOR_MODEL/REGION/PROJECT_ID`), strict JSON output,
tenacity retry (3 attempts, exp backoff) on `APIError`/`APIConnectionError`/
`JSONDecodeError`. Feedback is **detailed** (multi-paragraph critique +
actionable guidance, ~150-250 words; `max_tokens=2048`), not one-liner.

## Outputs

- `completion_text`, `privileged_information_prompt`
- `completion_log_probs: list[float] | None` — per-token rollout log-probs from
  vLLM (1:1 aligned with the generated tokens; the trainer re-encodes the text
  and asserts alignment); `None` when the server omitted logprobs
- `reflector_result: {verdict, feedback} | None` — used by the trainer for
  `reflector/pass_rate`.

## Known gotchas

- `_build_privileged_prompt_from_feedback` assumes the last message is a **user**
  message (`env/rag_env.py:62-66`) — with the new OLS format the last message is
  a `tool` message, so online-feedback hints would land inside the tool
  response. Fix: reuse `collator._append_hint`.
- The rebuild deep-copies `normalized_messages`, so the collator's messages are
  never mutated.
