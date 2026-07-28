"""IFBench collator for on-policy SDFT.

Prepares prompt_texts and constraint metadata for IFBenchApiAdapterEnv.
Each example comes from allenai/IF_multi_constraints_upto5 with fields:
    - messages: list[dict] — chat messages ending with the user request
    - ground_truth: str — Python repr of constraint metadata
    - constraint: str — human-readable constraint description
    - constraint_type: str — constraint category
"""

import ast
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase


@dataclass
class IFBenchCollator:
    """Collator for IFBench SDFT.

    Returns:
        prompt_texts: list[str] — tokenized prompts for vLLM generation
        raw_questions: list[str] — original user request text
        instruction_id_lists: list[list[str]] — per-example constraint IDs
        kwargs_lists: list[list[dict]] — per-example constraint kwargs
        constraint_descriptions: list[str] — human-readable constraint text
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts: list[str] = []
        raw_questions: list[str] = []
        instruction_id_lists: list[list[str]] = []
        kwargs_lists: list[list[dict]] = []
        constraint_descriptions: list[str] = []

        for ex in examples:
            messages = ex["messages"]
            # Last message is the user request
            raw_question = messages[-1]["content"]
            raw_questions.append(raw_question)

            # Parse ground_truth: Python repr string → list of dicts
            # Each dict has "instruction_id_list" and "kwargs"
            parsed = ast.literal_eval(ex["ground_truth"])
            # parsed is a list with one element containing the constraint metadata
            constraint_meta = parsed[0]
            instruction_id_lists.append(constraint_meta["instruction_id_list"])
            kwargs_lists.append(constraint_meta["kwargs"])

            # Human-readable constraint description
            constraint_descriptions.append(ex.get("constraint", ""))

            # Build prompt text from messages
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_texts.append(prompt_text)

        return {
            "prompt_texts": prompt_texts,
            "raw_questions": raw_questions,
            "instruction_id_lists": instruction_id_lists,
            "kwargs_lists": kwargs_lists,
            "constraint_descriptions": constraint_descriptions,
        }
