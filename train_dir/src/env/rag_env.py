"""RAG rollout environment for SDFT training.

Generates a completion via vLLM, optionally grades it via reflector,
and produces a ready-to-use privileged_information_prompt for the teacher.
"""

import copy

from src import reflector
from src.env.base import BaseEnv
from src.vllm_utils import vllm_generate


ONLINE_FEEDBACK_TEMPLATE = (
    "Correct solution:\n{golden_answer}\n\n"
    "The following is feedback from your earlier attempt:\n{feedback}"
)


class RagEnv(BaseEnv):
    """RAG-based rollout: vLLM generation + optional reflector feedback."""

    def __init__(
        self,
        prompt_text: str,
        vllm_base_url: str,
        privileged_information_prompt: str | None,
        raw_question: str,
        golden_answer: str,
        normalized_messages: list[dict],
        tokenizer,
        use_reflector: bool = False,
    ):
        self.prompt_text = prompt_text
        self.vllm_base_url = vllm_base_url
        self.raw_question = raw_question
        self.golden_answer = golden_answer
        self.normalized_messages = normalized_messages
        self.tokenizer = tokenizer
        self.use_reflector = use_reflector

        # outputs (populated by run())
        self.completion_text: str | None = None
        self.privileged_information_prompt: str | None = privileged_information_prompt
        self.reflector_result: dict[str, str] | None = None

    def run(self) -> None:
        self.completion_text, _ = vllm_generate(self.prompt_text, base_url=self.vllm_base_url)
        self.completion_text += self.tokenizer.eos_token

        if self.use_reflector:
            self.reflector_result = reflector.run(
                self.raw_question, self.golden_answer, self.completion_text,
            )
            self._build_privileged_prompt_from_feedback()

    def _build_privileged_prompt_from_feedback(self) -> None:
        cond_history = copy.deepcopy(self.normalized_messages)
        cond_history[-1]["content"] += "\n\n" + ONLINE_FEEDBACK_TEMPLATE.format(
            feedback=self.reflector_result["feedback"],
            golden_answer=self.golden_answer,
        )
        self.privileged_information_prompt = self.tokenizer.apply_chat_template(
            cond_history, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
