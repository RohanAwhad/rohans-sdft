"""API-Adapter rollout environment for SDFT training.

The adapter (student) sits between a user and an external API LLM.
It vets API responses with PASS/FAIL verdicts and feedback.
On FAIL, feedback is sent back to the API model to regenerate.

Training target: the adapter's last response (verdict + feedback).
"""

import copy
import re

import litellm
litellm.suppress_debug_info = True
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from megatron_trainer.env.base import BaseEnv
from megatron_trainer.vllm_utils import vllm_generate
from megatron_trainer.config import API_MODEL, GEN_MAX_NEW_TOKENS, MAX_ADAPTER_TURNS, THINKING_BUDGET


ADAPTER_SYSTEM_PROMPT = """\
You are a personalized user assistant that sits between an LLM and the user.
User directly requests the LLM for some task, and LLM comes back with a response.
Your job is to vet the llm response to see if it is correct and ready to be seen by the user.
Or does the response need to be edited.

You will fulfill this job based on the parametric knowledge that you have that is very specific to the task at hand.
LLM because it is a general LLM used by the world, may or may not know about these preferences.

So when you vet it, and see that something needs to be improved, you give that feedback to LLM by verbalizing the internal knowledge.

The input to you for this task will be:

```
<|USER_REQUEST_START|>
...
<|USER_REQUEST_END|>

<|LLM_RESPONSE_START|>
...
<|LLM_RESPONSE_END|>
```

And the output expected from you is:

```
<|VERDICT_START|>
PASS/FAIL
<|VERDICT_END|>
<|FEEDBACK_START|>
... keep it 1-3 lines ...
<|FEEDBACK_END|>
```

And when the verdict is FAIL, and the LLM generates a new response, that will be attached as a new message turn like this:

```
<|LLM_RESPONSE_START|>
...
<|LLM_RESPONSE_END|>
```

---
In the current setup, you have to verify mathematical operations. Addition, subtraction, multiplication and division.
And α, β, θ, and γ are encrypted operations which each represent one of the ops.
User will not provide which symbol represents which op.
""".strip()


HINDSIGHT_TEMPLATE = (
    "=== PRIVILEGED INFORMATION ===\n"
    "Treat the below information as hindsight from the env for the action that you are about to take.\n"
    "Note: α, β, θ, and γ are encrypted operations which each represent one of addition, multiplication, subtraction, or division.\n"
    "---\n"
    "LLM Response: {llm_response}\n"
    "---\n"
    "Env Feedback: {feedback}"
)


_VERDICT_RE = re.compile(
    r"<\|VERDICT_START\|>\s*(.*?)\s*<\|VERDICT_END\|>", re.DOTALL
)
_FEEDBACK_RE = re.compile(
    r"<\|FEEDBACK_START\|>\s*(.*?)\s*<\|FEEDBACK_END\|>", re.DOTALL
)
_BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)


class ApiAdapterEnv(BaseEnv):
    """API-Adapter rollout: adapter (student via vLLM) vets an external API model."""

    def __init__(
        self,
        prompt_text: str,
        vllm_base_url: str,
        raw_question: str,
        golden_answer: str,
        tokenizer,
        api_model: str = API_MODEL,
        max_adapter_turns: int = MAX_ADAPTER_TURNS,
        success_cache: dict[str, str] | None = None,
    ):
        self.prompt_text = prompt_text
        self.vllm_base_url = vllm_base_url
        self.raw_question = raw_question
        self.golden_answer = golden_answer
        self.tokenizer = tokenizer
        self.api_model = api_model
        self.max_adapter_turns = max_adapter_turns
        self.success_cache = success_cache

        # state (populated during rollout)
        self.adapter_history: list[dict] = []
        self.api_history: list[dict] = []

        # outputs (populated by run())
        self.completion_text: str | None = None
        self.privileged_information_prompt: str | None = None
        self.episode_result: bool | None = None
        self.verdict: bool = False
        self.feedback: str = ""

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        model_response = self.rollout(self.raw_question)
        model_answer = self.parse_model_answer(model_response) if model_response else None
        if model_answer is not None: self.evaluate(model_answer, self.golden_answer)
        self.episode_result = self.verdict
        self.generate_training_attrs()

    # ------------------------------------------------------------------
    # Rollout (follows .llm.md pseudocode)
    # ------------------------------------------------------------------

    def rollout(self, user_message: str) -> str | None:
        self.adapter_history = []
        self.api_history = [{"role": "user", "content": user_message}]
        turns_remaining = self.max_adapter_turns

        api_response = self.call_api(self.api_history)
        self.api_history.append({"role": "assistant", "content": api_response})

        self.build_adapter_history(api_response, user_message)
        while True:
            adapter_response = self.call_adapter()
            self.adapter_history.append({"role": "assistant", "content": adapter_response})
            verdict, feedback = self.parse_adapter_response(adapter_response)
            if not verdict:
                self.verdict = False
                self.feedback = f"Parse failed: could not parse adapter response"
                return None
            if verdict.strip().upper() == "PASS":
                return api_response

            # regenerate using api
            self.api_history.append({"role": "user", "content": feedback})
            api_response = self.call_api(self.api_history)
            self.api_history.append({"role": "assistant", "content": api_response})

            turns_remaining -= 1
            if turns_remaining == 0: break
            self.build_adapter_history(api_response, user_message=None)

        return api_response

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=10))
    def call_api(self, messages: list[dict]) -> str:
        """Call external API model via litellm."""
        response = litellm.completion(model=self.api_model, messages=messages)
        return response.choices[0].message.content

    def call_adapter(self) -> str:
        """Call adapter (student) via vLLM with thinking budget enforcement.

        Phase 1: generate with max_tokens=THINKING_BUDGET.
        Phase 2: if thinking was truncated (finish_reason=="length"),
                 force-close </think> and continue with remaining budget.
        """
        prompt_text = self.tokenizer.apply_chat_template(
            self.adapter_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # Phase 1: thinking-budgeted generation
        text, finish_reason = vllm_generate(
            prompt_text, base_url=self.vllm_base_url, max_tokens=THINKING_BUDGET,
        )

        if finish_reason != "length":
            return text

        # Phase 2: force-close thinking, generate the actual answer
        truncated_thinking = text
        if "</think>" not in truncated_thinking:
            truncated_thinking = truncated_thinking.rstrip() + ".\n</think>\n\n"

        continued_prompt = prompt_text + truncated_thinking
        answer_text, _ = vllm_generate(
            continued_prompt,
            base_url=self.vllm_base_url,
            max_tokens=GEN_MAX_NEW_TOKENS - THINKING_BUDGET,
        )
        return truncated_thinking + answer_text

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def build_adapter_history(self, api_response: str, user_message: str | None) -> None:
        """Build/append to adapter conversation history."""
        if not self.adapter_history:
            self.adapter_history.append({"role": "system", "content": ADAPTER_SYSTEM_PROMPT})

        if user_message is not None:
            content = (
                f"<|USER_REQUEST_START|>\n{user_message}\n<|USER_REQUEST_END|>\n\n"
                f"<|LLM_RESPONSE_START|>\n{api_response}\n<|LLM_RESPONSE_END|>"
            )
        else:
            content = f"<|LLM_RESPONSE_START|>\n{api_response}\n<|LLM_RESPONSE_END|>"

        self.adapter_history.append({"role": "user", "content": content})

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_adapter_response(self, text: str) -> tuple[str, str]:
        """Extract verdict and feedback from adapter output."""
        verdict_match = _VERDICT_RE.search(text)
        feedback_match = _FEEDBACK_RE.search(text)

        verdict = verdict_match.group(1).strip() if verdict_match else ""
        feedback = feedback_match.group(1).strip() if feedback_match else ""

        if not verdict:
            logger.warning(f"Could not parse verdict from adapter response: {text[:200]}")
        return verdict, feedback

    def parse_model_answer(self, response: str) -> str | None:
        """Extract \\boxed{...} answer from API model response."""
        match = _BOXED_RE.search(response)
        if match:
            return match.group(1).strip()
        self.verdict = False
        self.feedback = f"Parse failed: no \\boxed{{}} in API response: {response[:200]}"
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model_answer: str, golden_answer: str) -> bool:
        if str(model_answer) == str(golden_answer):
            self.verdict = True
            self.feedback = f"PASS. Model generated {model_answer}. Correct answer is {golden_answer}"
        else:
            self.verdict = False
            self.feedback = f"FAIL. Model generated {model_answer}. Correct answer was {golden_answer}"
        return self.verdict

    # ------------------------------------------------------------------
    # Training attributes
    # ------------------------------------------------------------------

    def generate_training_attrs(self) -> None:
        """Build completion_text, prompt_text, and privileged_information_prompt.

        Uses string slicing: full_text = prompt_text + completion_text.
        prompt_text ends with <|im_start|>assistant\\n (generation prompt),
        completion_text starts at the actual content (including <think> if present)
        and includes <|im_end|> at the end.
        """
        assert self.adapter_history[-1]["role"] == "assistant", (
            "Last adapter_history entry must be an assistant message"
        )

        # full_text: entire conversation with all turns (no generation prompt)
        full_text = self.tokenizer.apply_chat_template(
            self.adapter_history,
            tokenize=False,
            add_generation_prompt=False,
        )

        # prompt_text: everything up to and including <|im_start|>assistant\n
        self.prompt_text = self.tokenizer.apply_chat_template(
            self.adapter_history[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # completion_text: slice off the prompt prefix, strip trailing \n template artifact
        self.completion_text = full_text[len(self.prompt_text):].rstrip("\n")

        # conditional_text: prompt + hindsight appended to last user message
        hindsight = HINDSIGHT_TEMPLATE.format(llm_response=self.api_history[-1]['content'], feedback=self.feedback)

        # Inject cached successful response when current rollout failed
        if not self.verdict and self.success_cache and self.raw_question in self.success_cache:
            hindsight += (
                "\n\n=== CORRECT RESPONSE FROM ANOTHER ROLLOUT ===\n"
                + self.success_cache[self.raw_question]
            )

        cond_history = copy.deepcopy(self.adapter_history[:-1])
        cond_history[-1]["content"] += "\n\n" + hindsight
        self.privileged_information_prompt = self.tokenizer.apply_chat_template(
            cond_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
