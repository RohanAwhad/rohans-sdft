"""IFBench API-Adapter rollout environment for SDFT training.

The adapter (student) sits between a user and an external API LLM.
It vets API responses against IFBench output constraints with PASS/FAIL verdicts.
On FAIL, feedback is sent back to the API model to regenerate.

Training target: the adapter's last response (verdict + feedback).
Evaluation: programmatic IFBench checkers (loose mode), not boxed answer extraction.
"""

import copy
import re

import litellm
litellm.suppress_debug_info = True
from loguru import logger
from src import reflector
from src.env.base import BaseEnv
from src.env.ifbench.verifier import verify_response
from src.vllm_utils import vllm_generate
from src.config import API_MODEL, GEN_MAX_NEW_TOKENS, MAX_ADAPTER_TURNS, THINKING_BUDGET, llm_retry


IFBENCH_ADAPTER_SYSTEM_PROMPT = """\
You are a personalized user assistant that sits between an LLM and the user.
User requests the LLM for a task with specific output constraints, and LLM responds.
Your job is to vet the LLM response to verify it satisfies ALL constraints in the request.
When you find a constraint violation, give feedback to the LLM specifying which \
constraint was violated and how to fix it.

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
... keep it 1-3 lines, specify which constraint was violated and how to fix it ...
<|FEEDBACK_END|>
```

And when the verdict is FAIL, and the LLM generates a new response, that will be attached as a new message turn like this:

```
<|LLM_RESPONSE_START|>
...
<|LLM_RESPONSE_END|>
```

---
In the current setup, the user request contains specific output format constraints \
(e.g. word count ranges, sentence structure rules, formatting requirements). \
You must verify whether the LLM's response satisfies ALL of these constraints.""".strip()


_VERDICT_RE = re.compile(
    r"<\|VERDICT_START\|>\s*(.*?)\s*<\|VERDICT_END\|>", re.DOTALL
)
_FEEDBACK_RE = re.compile(
    r"<\|FEEDBACK_START\|>\s*(.*?)\s*<\|FEEDBACK_END\|>", re.DOTALL
)


class IFBenchApiAdapterEnv(BaseEnv):
    """IFBench API-Adapter rollout: adapter (student via vLLM) vets API responses against constraints."""

    def __init__(
        self,
        prompt_text: str,
        vllm_base_url: str,
        raw_question: str,
        instruction_id_list: list[str],
        kwargs_list: list[dict],
        constraint_description: str,
        tokenizer,
        api_model: str = API_MODEL,
        max_adapter_turns: int = MAX_ADAPTER_TURNS,
        success_cache: dict[str, str] | None = None,
    ):
        self.prompt_text = prompt_text
        self.vllm_base_url = vllm_base_url
        self.raw_question = raw_question
        self.instruction_id_list = instruction_id_list
        self.kwargs_list = kwargs_list
        self.constraint_description = constraint_description
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
        api_response = self.rollout(self.raw_question)
        if api_response is not None:
            self.evaluate(api_response)
        self.episode_result = self.verdict
        conv_lines = []
        for m in self.adapter_history:
            if m["role"] == "system": continue
            content = re.sub(r"<think>.*?</think>", "", m["content"], flags=re.DOTALL).strip() if m["role"] == "assistant" else m["content"]
            conv_lines.append(f"{m['role'].capitalize()}: {content}")
        self.reflector_feedback = reflector.run_ifbench_adapter(
            raw_question=self.raw_question,
            constraint_description=self.constraint_description,
            verification_results=self.feedback,
            adapter_conversation_history="\n\n".join(conv_lines),
            episode_answer=api_response or "",
            episode_feedback=self.feedback,
            adapter_system_prompt=IFBENCH_ADAPTER_SYSTEM_PROMPT,
        )
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
                self.feedback = "Parse failed: could not parse adapter response"
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

    @llm_retry
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
            enable_thinking=bool(THINKING_BUDGET),
        )

        text, finish_reason = vllm_generate(
            prompt_text, base_url=self.vllm_base_url, max_tokens=THINKING_BUDGET if THINKING_BUDGET else GEN_MAX_NEW_TOKENS,
        )
        if not THINKING_BUDGET: return text

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
            self.adapter_history.append({"role": "system", "content": IFBENCH_ADAPTER_SYSTEM_PROMPT})

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

    # ------------------------------------------------------------------
    # Evaluation (IFBench programmatic verifiers)
    # ------------------------------------------------------------------

    def evaluate(self, api_response: str) -> bool:
        """Evaluate API response against IFBench constraints using programmatic checkers."""
        all_pass, results = verify_response(
            response=api_response,
            instruction_id_list=self.instruction_id_list,
            kwargs_list=self.kwargs_list,
            prompt=self.raw_question,
        )
        self.verdict = all_pass
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        failed_ids = [r["instruction_id"] for r in results if not r["passed"]]
        if all_pass:
            self.feedback = f"PASS. All {total} constraints satisfied."
        else:
            self.feedback = f"FAIL. Constraints: {passed}/{total}. Failed: {', '.join(failed_ids)}"
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

        # conditional_text: prompt + reflector feedback (+ cached correct answer) appended to last user message
        cond_history = copy.deepcopy(self.adapter_history[:-1])
        privilege_text = self.reflector_feedback
        if self.success_cache and self.raw_question in self.success_cache:
            privilege_text += "\n\nCorrect answer from a previous successful attempt:\n" + self.success_cache[self.raw_question]
            logger.debug(f"Cache hit for privilege prompt: {self.raw_question[:80]!r}")
        cond_history[-1]["content"] += "\n\n" + privilege_text
        self.privileged_information_prompt = self.tokenizer.apply_chat_template(
            cond_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
