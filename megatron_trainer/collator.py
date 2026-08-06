"""On-policy SDFT collator: produces prompt_texts and conditional_texts.

Completions are generated on-the-fly by vLLM, so the collator only prepares
the two prompt variants (with and without privileged information).
"""

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase

from megatron_trainer.config import (
    IS_GPT_OSS,
    IS_QWEN,
    MODEL_NAME,
    STUDENT_MAX_PROMPT_LEN,
    TEACHER_MAX_PROMPT_LEN,
    TRAIN_DATA_PATH,
)


ANALYSIS_CHANNEL_SUFFIX = "<|channel|>analysis<|message|>"

TOOL_SHAPE_KEYS = ("tool_calls", "tool_results")


def _append_analysis_channel(text: str) -> str:
    """Force gpt-oss generation into the analysis (thinking) channel.

    The gpt-oss chat template ends the generation prompt on a bare
    '<|start|>assistant' and the model picks a channel on its own; an explicit
    analysis channel start makes it deterministic. Trailing whitespace is
    stripped so the suffix attaches directly to the assistant start token.
    """
    if not text.rstrip().endswith(ANALYSIS_CHANNEL_SUFFIX):
        return text.rstrip() + ANALYSIS_CHANNEL_SUFFIX
    return text


HINDSIGHT_TEMPLATES = {
    "user_response": (
        "The following is the correct answer. "
        "Use this to guide your response: {o}"
    ),
    "enriched_user_response": (
        "The following is the relevant documentation and the correct answer. "
        "Use this to guide your response:\n\n"
        "Documentation:\n{doc}\n\n"
        "Answer:\n{answer}"
    ),
}


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert 'from/value' (WildChat) and 'tool_results' formats to 'role/content'.

    Tool messages without normalization render an empty <tool_response> on Qwen3
    (results silently vanish), so they are json-dumped into content.
    """
    normalized = []
    for msg in messages:
        if "tool_results" in msg and "content" not in msg:
            normalized.append({"role": "tool", "content": json.dumps(msg["tool_results"])})
        elif "value" in msg and "content" not in msg:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            original_role = msg.get("from", "user")
            new_role = role_map.get(original_role, original_role)
            normalized.append({"role": new_role, "content": msg["value"]})
        else:
            normalized.append(msg)
    return normalized


def _target_text(user_response: Dict[str, Any]) -> str:
    """Hand-format the reference answer: one 'name(args_json)' line per tool
    call plus the final content (newline-joined). No template render, no
    tokenizer round-trip (avoids template boilerplate and special-token
    stripping artifacts).
    """
    parts = []
    for tc in user_response.get("tool_calls", []):
        parts.append(f"{tc['name']}({json.dumps(tc['arguments'])})")
    content = (user_response.get("content") or user_response.get("value") or "").strip()
    if content:
        parts.append(content)
    return "\n".join(parts)


def _append_hint(messages: List[Dict[str, Any]], hint: str) -> List[Dict[str, Any]]:
    """Append the privileged hint to a copy of the trajectory.

    Merges into the last message's content when it is a user message (old
    WildChat format); otherwise appends a fresh user message (new format: the
    last message is a tool message, so the hint renders as its own user turn
    right before the generation prompt, after all tool results).
    """
    history = copy.deepcopy(messages)
    if history and history[-1]["role"] == "user":
        history[-1]["content"] += "\n\n" + hint
    else:
        history.append({"role": "user", "content": hint})
    return history


def _render_tokens(
    messages: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    render_kwargs: Dict[str, Any],
) -> tuple[int, str]:
    """Render a trajectory and count tokens exactly like the trainer does
    (add_special_tokens=False, truncation applied by the caller's budget).
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        **render_kwargs,
    )
    return len(tokenizer.encode(text, add_special_tokens=False)), text


def _truncate_to_budget(
    messages: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    budget: int,
    protect_last: bool,
    render_kwargs: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], str]:
    """Drop the earliest messages (after the system message) until the rendered
    token count fits the budget. Mutates the passed list.

    Message-level only — no partial-message cuts, no token-level slicing of the
    rendered text. The system message (index 0) is never dropped; with
    protect_last=True the last message (the privileged hint) is never dropped
    either. Drop order: oldest turns first, and the last user message (the
    question) is dropped only when every other message is gone. The protected
    set is asserted to fit the budget upstream, so the loop always terminates
    in budget (or with only the protected set left).
    """
    while True:
        n_tokens, text = _render_tokens(messages, tokenizer, render_kwargs)
        if n_tokens <= budget:
            return messages, text
        protected_last = len(messages) - 1 if protect_last else None
        last_user = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"),
            default=None,
        )
        candidates = [
            i for i in range(1, len(messages))
            if i != protected_last and i != last_user
        ]
        if not candidates:
            candidates = [
                i for i in range(1, len(messages)) if i != protected_last
            ]
        if not candidates:
            return messages, text
        del messages[min(candidates)]


def _load_tool_defs() -> Optional[List[Dict[str, Any]]]:
    """Load tool defs from tool_defs.json next to the train dataset.

    Descriptions are stripped (top-level + per-property): full OLS defs cost
    ~7,214 tokens/prompt, stripped ~2,332 (validated). Absent file -> None,
    meaning no <tools> block is rendered.
    """
    path = os.path.join(os.path.dirname(TRAIN_DATA_PATH), "tool_defs.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tool_defs = json.load(f)
    for d in tool_defs:
        fn = d.get("function", d)
        fn.pop("description", None)
        for prop in fn.get("parameters", {}).get("properties", {}).values():
            prop.pop("description", None)
    return tool_defs


TOOL_DEFS = _load_tool_defs()


def _assert_protected_fits(
    messages: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    budget: int,
    render_kwargs: Dict[str, Any],
    label: str,
) -> None:
    """Assert a protected-only render fits the budget (config invariant)."""
    n_tokens, _ = _render_tokens(messages, tokenizer, render_kwargs)
    if n_tokens > budget:
        raise ValueError(
            f"Protected-set render ({label}) is {n_tokens} tokens, exceeding "
            f"budget {budget}; max lengths must exceed system/hint renders "
            "(config error — raise the max length)"
        )


@dataclass
class SDFTCollator:
    """Collator for on-policy SDFT.

    Returns:
        prompt_texts: list[str]       — student context (question only)
        conditional_texts: list[str]   — teacher context (question + privileged info)
    """

    tokenizer: PreTrainedTokenizerBase
    hindsight_field: str = "enriched_user_response"

    def __post_init__(self) -> None:
        render_kwargs = {"tools": TOOL_DEFS} if TOOL_DEFS is not None else {}
        _assert_protected_fits(
            [{"role": "system", "content": ""}],
            self.tokenizer,
            STUDENT_MAX_PROMPT_LEN,
            render_kwargs,
            "system + tools (init)",
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_texts: list[str] = []
        conditional_texts: list[str] = []
        raw_questions: list[str] = []
        golden_answers: list[str] = []
        normalized_messages: list[list[dict[str, str]]] = []

        render_kwargs = {"tools": TOOL_DEFS} if TOOL_DEFS is not None else {}

        for ex in examples:
            clean_prompt = _normalize_messages(ex["prompt"])

            # Family guard: tools / tool_calls / tool_results are only validated
            # for Qwen. gpt-oss fails silently on this shape (drops tool calls
            # 2..N, crashes on content|tojson), so fail loudly instead.
            if not IS_QWEN and (
                TOOL_DEFS is not None
                or any(
                    any(k in m for k in TOOL_SHAPE_KEYS) for m in ex["prompt"]
                )
            ):
                raise ValueError(
                    "New-format path (tools/tool_calls/tool_results) is not "
                    f"validated for model family {MODEL_NAME}; Qwen only"
                )

            golden_answers.append(_target_text(ex["user_response"]))

            # Per-example protected-set assert (system always complete).
            if clean_prompt and clean_prompt[0]["role"] == "system":
                _assert_protected_fits(
                    clean_prompt[:1],
                    self.tokenizer,
                    STUDENT_MAX_PROMPT_LEN,
                    render_kwargs,
                    "system + tools",
                )

            # --- Student prompt (x) ---
            student_prompt, p_text = _truncate_to_budget(
                copy.deepcopy(clean_prompt),
                self.tokenizer,
                STUDENT_MAX_PROMPT_LEN,
                protect_last=False,
                render_kwargs=render_kwargs,
            )
            if IS_GPT_OSS:
                p_text = _append_analysis_channel(p_text)
            prompt_texts.append(p_text)

            # Raw data for env: truncated trajectory, last question, golden answer
            normalized_messages.append(student_prompt)
            raw_questions.append(
                next(m["content"] for m in reversed(student_prompt) if m["role"] == "user")
            )

            # --- Teacher prompt (x, o) — append privileged info ---
            # online_feedback: conditional_text is built dynamically by the env
            # after rollout, so we skip it here.
            if self.hindsight_field == "online_feedback":
                conditional_texts.append(None)
            else:
                template = HINDSIGHT_TEMPLATES[self.hindsight_field]

                if self.hindsight_field == "enriched_user_response":
                    doc_data = ex["enriched_user_response"]
                    doc = (doc_data.get("value") or doc_data.get("content") or "").strip()
                    answer_data = ex["user_response"]
                    answer = (answer_data.get("value") or answer_data.get("content") or "").strip()
                    hint = template.format(doc=doc, answer=answer)
                else:
                    o = _target_text(ex[self.hindsight_field])
                    hint = template.format(o=o)

                # Per-example protected-set assert (system + hint always complete).
                assert_msgs: list[dict] = []
                if clean_prompt and clean_prompt[0]["role"] == "system":
                    assert_msgs.append(clean_prompt[0])
                assert_msgs.append({"role": "user", "content": hint})
                _assert_protected_fits(
                    assert_msgs,
                    self.tokenizer,
                    TEACHER_MAX_PROMPT_LEN,
                    render_kwargs,
                    "system + tools + hint",
                )

                conditional_history = _append_hint(clean_prompt, hint)
                _, xo_text = _truncate_to_budget(
                    conditional_history,
                    self.tokenizer,
                    TEACHER_MAX_PROMPT_LEN,
                    protect_last=True,
                    render_kwargs=render_kwargs,
                )
                if IS_GPT_OSS:
                    xo_text = _append_analysis_channel(xo_text)
                conditional_texts.append(xo_text)

        return {
            "prompt_texts": prompt_texts,
            "conditional_texts": conditional_texts,
            "raw_questions": raw_questions,
            "golden_answers": golden_answers,
            "normalized_messages": normalized_messages,
        }
