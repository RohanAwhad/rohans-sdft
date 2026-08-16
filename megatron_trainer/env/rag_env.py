"""RAG rollout environment for SDFT training.

Generates a completion via vLLM, optionally grades it via reflector,
and produces a ready-to-use privileged_information_prompt for the teacher.
"""

import copy

from megatron_trainer import reflector
from megatron_trainer.config import STUDENT_THINKING
from megatron_trainer.env.base import BaseEnv
from megatron_trainer.vllm_utils import vllm_generate


ONLINE_FEEDBACK_TEMPLATE = (
    "Relevant documentation:\n{chunk}\n\n"
    "Correct solution:\n{golden_answer}\n\n"
    "The following is feedback from your earlier attempt:\n{feedback}"
)

ONLINE_FEEDBACK_NO_CHUNK_TEMPLATE = (
    "Correct solution:\n{golden_answer}\n\n"
    "The following is feedback from your earlier attempt:\n{feedback}"
)

FALLBACK_TEMPLATE = "Relevant documentation:\n{chunk}\n\nCorrect solution:\n{golden_answer}"
FALLBACK_NO_CHUNK_TEMPLATE = "Correct solution:\n{golden_answer}"


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
        golden_chunk: str = "",
        seed_offset: int = 0,
        reward_only: bool = False,
    ):
        self.prompt_text = prompt_text
        self.vllm_base_url = vllm_base_url
        self.raw_question = raw_question
        self.golden_answer = golden_answer
        self.golden_chunk = golden_chunk
        self.normalized_messages = normalized_messages
        self.tokenizer = tokenizer
        self.use_reflector = use_reflector
        self.seed_offset = seed_offset
        self.reward_only = reward_only

        # outputs (populated by run())
        self.completion_text: str | None = None
        self.completion_log_probs: list[float] | None = None
        self.finish_reason: str | None = None
        self.privileged_information_prompt: str | None = privileged_information_prompt
        self.reflector_result: dict[str, str] | None = None

    def run(self) -> None:
        text, finish_reason, logprobs = vllm_generate(
            self.prompt_text,
            base_url=self.vllm_base_url,
            seed_offset=self.seed_offset,
        )
        self.completion_text = text
        self.completion_log_probs = logprobs
        self.finish_reason = finish_reason

        if self.use_reflector:
            self.reflector_result = reflector.run(
                self.raw_question,
                self.golden_answer,
                self.completion_text,
                reward_only=self.reward_only,
            )
            if self.reflector_result is not None:
                self._build_privileged_prompt_from_feedback()
            else:
                self._build_privileged_prompt_fallback()

    def _build_privileged_prompt_fallback(self) -> None:
        cond_history = copy.deepcopy(self.normalized_messages)
        template = FALLBACK_TEMPLATE if self.golden_chunk else FALLBACK_NO_CHUNK_TEMPLATE
        cond_history[-1]["content"] += "\n\n" + template.format(
            chunk=self.golden_chunk,
            golden_answer=self.golden_answer,
        )
        self.privileged_information_prompt = self.tokenizer.apply_chat_template(
            cond_history, tokenize=False, add_generation_prompt=True,
            enable_thinking=STUDENT_THINKING,
        )

    def _build_privileged_prompt_from_feedback(self) -> None:
        cond_history = copy.deepcopy(self.normalized_messages)
        # NOTE: assumes the last message is a user message. With the new OLS
        # format the last message is a `tool` message, so this hint would land
        # inside the <tool_response> body instead of as a fresh user turn
        # before the generation prompt. Fix (when online_feedback + OLS runs):
        # reuse collator._append_hint.
        template = (
            ONLINE_FEEDBACK_TEMPLATE
            if self.golden_chunk
            else ONLINE_FEEDBACK_NO_CHUNK_TEMPLATE
        )
        cond_history[-1]["content"] += "\n\n" + template.format(
            chunk=self.golden_chunk,
            feedback=self.reflector_result["feedback"],
            golden_answer=self.golden_answer,
        )
        self.privileged_information_prompt = self.tokenizer.apply_chat_template(
            cond_history, tokenize=False, add_generation_prompt=True,
            enable_thinking=STUDENT_THINKING,
        )
