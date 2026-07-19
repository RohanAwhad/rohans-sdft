# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm",
#     "loguru",
#     "tenacity",
#     "google-cloud-aiplatform>=1.38",
# ]
# ///
"""Test reflector.run_api_adapter + conditional adapter regeneration."""

import copy
import sys
from pathlib import Path

# allow running from project root: python train_dir/play.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litellm
litellm.suppress_debug_info = True

from src import reflector
from src.config import REFLECTOR_MODEL

# Inlined to avoid importing api_adapter_env (which pulls torch via vllm_utils)
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
User will not provide which symbol represents which op."""


SCENARIOS = [
    # --- 1. Plain arithmetic, correct, adapter PASSes ---
    {
        "name": "plain_correct_pass",
        "raw_question": "I need you to calculate 25 - 14 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": ["25 - 14 = 11, so the answer is \\boxed{11}"],
        "adapter_verdicts": ["PASS. The calculation is correct."],
        "episode_feedback": "PASS. Model generated 11. Correct answer is 11",
    },
    # --- 2. Plain arithmetic, incorrect, adapter wrongly PASSes ---
    {
        "name": "plain_incorrect_adapter_passes",
        "raw_question": "I need you to calculate 96 * 27 - 26 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": ["96 * 27 = 2592, then 2592 - 26 = 2566, so the answer is \\boxed{2566}"],
        "adapter_verdicts": ["PASS. The calculation looks correct."],
        "episode_feedback": "FAIL. Model generated 2566. Correct answer was 2566",
    },
    # --- 3. Plain arithmetic, incorrect, adapter catches it ---
    {
        "name": "plain_incorrect_adapter_catches",
        "raw_question": "I need you to calculate 85 * 56 + 69 * 7 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": [
            "85 * 56 = 4760, 69 * 7 = 483, 4760 + 483 = 5243, so the answer is \\boxed{5243}",
            "Let me recalculate. 85 * 56 = 4760, 69 * 7 = 483, 4760 + 483 = 5243. The answer is \\boxed{5243}",
        ],
        "adapter_verdicts": [
            "FAIL. 85 * 56 = 4760 is correct, but 69 * 7 = 483 is correct, and 4760 + 483 = 5243 is correct. Wait, let me recheck: 85*56=4760, 69*7=483, sum=5243. Actually this looks right.",
            "PASS. The recalculated answer matches.",
        ],
        "episode_feedback": "FAIL. Model generated 5243. Correct answer was 5243",
    },
    # --- 4. Greek symbol (α = subtraction), correct, adapter PASSes ---
    {
        "name": "greek_correct_pass",
        "raw_question": "I need you to calculate 13 α 49 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": ["If α represents subtraction, then 13 - 49 = -36. The answer is \\boxed{-36}"],
        "adapter_verdicts": ["PASS. The reasoning about α being subtraction is consistent and the calculation 13 - 49 = -36 is correct."],
        "episode_feedback": "PASS. Model generated -36. Correct answer is -36",
    },
    # --- 5. Greek symbol (γ = multiplication), API wrong, adapter wrongly PASSes ---
    {
        "name": "greek_incorrect_adapter_misses",
        "raw_question": "I need you to calculate 73 γ 45 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": ["If γ represents addition, then 73 + 45 = 118. The answer is \\boxed{118}"],
        "adapter_verdicts": ["PASS. The calculation 73 + 45 = 118 is arithmetically correct."],
        "episode_feedback": "FAIL. Model generated 118. Correct answer was 3285",
    },
    # --- 6. Greek symbol (θ), API wrong, adapter catches and API fixes ---
    {
        "name": "greek_incorrect_adapter_catches_api_fixes",
        "raw_question": "I need you to calculate 50 θ 12 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": [
            "If θ represents addition, then 50 + 12 = 62. The answer is \\boxed{62}",
            "Let me reconsider. If θ represents multiplication, then 50 * 12 = 600. The answer is \\boxed{600}",
        ],
        "adapter_verdicts": [
            "FAIL. The assumption that θ is addition may be wrong. Consider other operations.",
            "PASS. 50 * 12 = 600 is correct.",
        ],
        "episode_feedback": "PASS. Model generated 600. Correct answer is 600",
    },
    # --- 7. Multi-operator with greek symbols, API wrong sign ---
    {
        "name": "greek_multi_op_sign_error",
        "raw_question": "I need you to calculate 20 α 8 θ 3 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": ["Assuming α is subtraction and θ is multiplication: 20 - 8 = 12, 12 * 3 = 36. The answer is \\boxed{36}"],
        "adapter_verdicts": ["PASS. The step-by-step calculation looks correct."],
        "episode_feedback": "FAIL. Model generated 36. Correct answer was 36",
    },
    # --- 8. Parse failure: adapter response was garbled ---
    {
        "name": "parse_failure",
        "raw_question": "I need you to calculate 10 + 5 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": [],
        "adapter_verdicts": [],
        "episode_feedback": "Parse failed: could not parse adapter response",
    },
    # --- 9. No boxed answer from API ---
    {
        "name": "no_boxed_answer",
        "raw_question": "I need you to calculate 7 β 3 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": ["7 β 3 = 21. The answer is 21."],
        "adapter_verdicts": ["PASS. The calculation is correct."],
        "episode_feedback": "Parse failed: no \\boxed{} in API response: 7 β 3 = 21. The answer is 21.",
    },
    # --- 10. Multi-turn: adapter FAILs twice, API eventually gets it ---
    {
        "name": "multi_turn_eventual_success",
        "raw_question": "I need you to calculate 15 γ 4 and return the final answer in \\boxed{} format.\n\nExample: 3 + 4 = \\boxed{7}",
        "api_responses": [
            "If γ is addition: 15 + 4 = 19. The answer is \\boxed{19}",
            "If γ is subtraction: 15 - 4 = 11. The answer is \\boxed{11}",
            "If γ is multiplication: 15 * 4 = 60. The answer is \\boxed{60}",
        ],
        "adapter_verdicts": [
            "FAIL. The assumption about γ may be incorrect. Try a different operation.",
            "FAIL. Still not matching expected behavior. Consider multiplication or division.",
            "PASS. 15 * 4 = 60 is correct.",
        ],
        "episode_feedback": "PASS. Model generated 60. Correct answer is 60",
    },
]


def build_adapter_history(raw_question: str, api_responses: list[str], adapter_verdicts: list[str]) -> list[dict]:
    """Build adapter conversation history from scenario data.

    Mirrors ApiAdapterEnv.build_adapter_history logic:
    - First user message includes USER_REQUEST + first LLM_RESPONSE
    - Subsequent user messages are just LLM_RESPONSE (after adapter FAIL)
    """
    history: list[dict] = [{"role": "system", "content": ADAPTER_SYSTEM_PROMPT}]

    for i, api_resp in enumerate(api_responses):
        if i == 0:
            content = (
                f"<|USER_REQUEST_START|>\n{raw_question}\n<|USER_REQUEST_END|>\n\n"
                f"<|LLM_RESPONSE_START|>\n{api_resp}\n<|LLM_RESPONSE_END|>"
            )
        else:
            content = f"<|LLM_RESPONSE_START|>\n{api_resp}\n<|LLM_RESPONSE_END|>"

        history.append({"role": "user", "content": content})

        if i < len(adapter_verdicts):
            history.append({"role": "assistant", "content": adapter_verdicts[i]})

    return history


def generate_conditional_response(adapter_history: list[dict], reflector_feedback: str) -> str:
    """Generate adapter response with reflector feedback as privileged info.

    Injects feedback into last user message, then calls litellm.
    """
    cond_history = copy.deepcopy(adapter_history)

    # Strip the last assistant message (the original verdict we want to regenerate)
    if cond_history[-1]["role"] == "assistant":
        cond_history = cond_history[:-1]

    # Inject reflector feedback into last user message
    cond_history[-1]["content"] += "\n\n" + reflector_feedback

    response = litellm.completion(
        model=REFLECTOR_MODEL,
        max_tokens=512,
        messages=cond_history,
    )
    return response.choices[0].message.content.strip()


def main():
    for scenario in SCENARIOS:
        name = scenario.pop("name")
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"Question: {scenario['raw_question'][:80]}")
        print(f"Episode feedback: {scenario['episode_feedback']}")
        print(f"-" * 60)

        # Step 1: reflector feedback
        feedback = reflector.run_api_adapter(**scenario, adapter_system_prompt=ADAPTER_SYSTEM_PROMPT)
        print(f"Reflector: {feedback}")

        # Step 2: regenerate adapter response with privileged info
        if not scenario["api_responses"]:
            print(f"Conditional: [skipped — no API responses]")
            continue

        adapter_history = build_adapter_history(
            scenario["raw_question"],
            scenario["api_responses"],
            scenario["adapter_verdicts"],
        )
        original_verdict = scenario["adapter_verdicts"][-1] if scenario["adapter_verdicts"] else "[none]"
        conditional = generate_conditional_response(adapter_history, feedback)

        print(f"Original:    {original_verdict}")
        print(f"Conditional: {conditional}")


if __name__ == "__main__":
    main()
