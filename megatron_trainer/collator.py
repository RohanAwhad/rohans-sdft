"""On-policy SDFT collator: produces prompt_texts and conditional_texts.

Completions are generated on-the-fly by vLLM, so the collator only prepares
the two prompt variants (with and without privileged information).
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase


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
class SDFTCollator:
    """Collator for on-policy SDFT.

    Returns:
        prompt_texts: list[str]       — student context (question only)
        conditional_texts: list[str]   — teacher context (question + privileged info)
    """

    tokenizer: PreTrainedTokenizerBase
    hindsight_field: str = "enriched_user_response"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_texts: list[str] = []
        conditional_texts: list[str] = []
        raw_questions: list[str] = []
        golden_answers: list[str] = []
        normalized_messages: list[list[dict[str, str]]] = []

        for ex in examples:
            clean_prompt = _normalize_messages(ex["prompt"])

            # Raw data for env: full messages, last question, golden answer
            normalized_messages.append(clean_prompt)
            raw_questions.append(clean_prompt[-1]["content"])
            answer_data = ex["user_response"]
            golden_answers.append(
                (answer_data.get("value") or answer_data.get("content")).strip()
            )

            # --- Student prompt (x) ---
            p_text = self.tokenizer.apply_chat_template(
                clean_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_texts.append(p_text)

            # --- Teacher prompt (x, o) — append privileged info ---
            # online_feedback: conditional_text is built dynamically by the env
            # after rollout, so we skip it here.
            if self.hindsight_field == "online_feedback":
                conditional_texts.append(None)
            else:
                template = HINDSIGHT_TEMPLATES[self.hindsight_field]
                conditional_history = copy.deepcopy(clean_prompt)

                if self.hindsight_field == "enriched_user_response":
                    doc_data = ex["enriched_user_response"]
                    doc = (doc_data.get("value") or doc_data.get("content")).strip()
                    answer_data = ex["user_response"]
                    answer = (answer_data.get("value") or answer_data.get("content")).strip()
                    conditional_history[-1]["content"] += "\n\n" + template.format(
                        doc=doc, answer=answer
                    )
                else:
                    hindsight_data = ex[self.hindsight_field]
                    o = (hindsight_data.get("value") or hindsight_data.get("content")).strip()
                    conditional_history[-1]["content"] += "\n\n" + template.format(o=o)

                xo_text = self.tokenizer.apply_chat_template(
                    conditional_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                conditional_texts.append(xo_text)

        return {
            "prompt_texts": prompt_texts,
            "conditional_texts": conditional_texts,
            "raw_questions": raw_questions,
            "golden_answers": golden_answers,
            "normalized_messages": normalized_messages,
        }
