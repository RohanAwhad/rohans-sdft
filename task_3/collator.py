"""Adapter collator for SDFT on synthetic algebra dataset.

Pre-formatted prompts (MaaS format) — no chat template needed.
Student prompt: system + user messages as plain text.
Teacher prompt: same + user_response appended as privileged info.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert 'from/value' (WildChat) format to 'role/content' (standard)."""
    normalized = []
    for msg in messages:
        if "value" in msg and "content" not in msg:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            original_role = msg.get("from", "user")
            new_role = role_map.get(original_role, original_role)
            normalized.append({"role": new_role, "content": msg["value"]})
        else:
            normalized.append(msg)
    return normalized


@dataclass
class AdapterCollator:
    """Collator for adapter-style SDFT.

    Prompts are pre-formatted text (no chat template). The dataset
    provides 'prompt' (list of messages) and 'user_response' (privileged info).

    Returns:
        prompt_texts: list[str]       — student context (raw text)
        conditional_texts: list[str]   — teacher context (with mapping + answer)
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_texts: list[str] = []
        conditional_texts: list[str] = []

        for ex in examples:
            messages = _normalize_messages(ex["prompt"])
            system_msg = messages[0]["content"] if messages[0].get("role") == "system" else ""
            user_msg = messages[-1]["content"]

            # Student: system + user as plain text
            student = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
            prompt_texts.append(student)

            # Teacher: system + user + privileged info (mapping + answer)
            hint = (
                ex["user_response"].get("value")
                or ex["user_response"].get("content")
            ).strip()
            teacher_user = f"{user_msg}\n\n{hint}"
            teacher = f"{system_msg}\n\n{teacher_user}" if system_msg else teacher_user
            conditional_texts.append(teacher)

        return {
            "prompt_texts": prompt_texts,
            "conditional_texts": conditional_texts,
        }
