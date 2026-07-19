"""Reflector: generates dynamic privileged feedback for the teacher.

Given (question, golden_answer, model_response), an external LLM grades the
response and produces a one-line feedback. Returns structured {verdict, feedback}.
"""

import json

import litellm
litellm.suppress_debug_info = True
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import REFLECTOR_MODEL


REFLECTOR_SYSTEM_PROMPT = """\
You are a grader comparing a model's response against the correct answer.
Output EXACTLY this JSON format and nothing else:

```json
{"verdict": "PASS", "feedback": "one sentence why, max 30 words"}
```

verdict must be PASS or FAIL. No other text outside the json block."""

REFLECTOR_USER_TEMPLATE = """\
Question:
{question}

Correct Answer:
{golden_answer}

Model's Response:
{model_response}"""



@retry(
  stop=stop_after_attempt(3),
  wait=wait_exponential(multiplier=1, min=0.2, max=10),
  retry=retry_if_exception_type((litellm.exceptions.APIError, litellm.exceptions.APIConnectionError, json.JSONDecodeError)),
)
def run(question: str, golden_answer: str, model_response: str) -> dict[str, str]:
    """Reflect on model_response vs golden_answer.

    Returns: {"verdict": "PASS"|"FAIL", "feedback": "one line reason"}
    """
    user_content: str = REFLECTOR_USER_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        model_response=model_response,
    )
    response = litellm.completion(
        model=REFLECTOR_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw: str = response.choices[0].message.content.strip()
    raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    parsed: dict[str, str] = json.loads(raw)
    logger.debug(f"Reflector: {parsed['verdict']} — {parsed['feedback']}")
    return parsed


# ---------------------------------------------------------------------------
# API-Adapter reflector
# ---------------------------------------------------------------------------

ADAPTER_REFLECTOR_SYSTEM_PROMPT = """\
You are reviewing an episode of an API adapter that vets math responses.

The adapter sits between a user and an API LLM. It evaluates API responses \
and issues PASS/FAIL verdicts with feedback. On FAIL, feedback is sent back \
to the API to regenerate.

The math uses encrypted operators (α, β, θ, γ) that each represent one of \
addition, multiplication, subtraction, or division.

=== EXAMPLE 1 ===
Episode:
- User question: What is 5 * 9?
- API response(s): ["5 * 9 = 45, so the answer is \\\\boxed{45}"]
- Adapter verdict(s)/feedback: ["PASS. The calculation is correct."]
- Episode feedback: PASS. Model generated 45. Correct answer is 45

```json
{"feedback_for_adapter": "Adapter correctly verified 5 * 9 = 45 and issued PASS. Good job."}
```

=== EXAMPLE 2 ===
Episode:
- User question: What is 3 - 9?
- API response(s): ["3 - 9 = 6, so the answer is \\\\boxed{6}"]
- Adapter verdict(s)/feedback: ["PASS. The calculation looks correct."]
- Episode feedback: FAIL. Model generated 6. Correct answer was -6

```json
{"feedback_for_adapter": "Adapter missed that 3 - 9 = -6 not 6. Should have issued FAIL and pointed out the sign error."}
```

=== YOUR TURN ===
Output EXACTLY this JSON and nothing else:
```json
{"feedback_for_adapter": "one sentence, max 30 words"}
```"""

ADAPTER_REFLECTOR_USER_TEMPLATE = """\
Episode:
- Adapter system prompt: {adapter_system_prompt}
- User question: {raw_question}
- API response(s): {api_responses}
- Adapter verdict(s)/feedback: {adapter_verdicts}
- Episode feedback: {episode_feedback}"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.2, max=10),
    retry=retry_if_exception_type((litellm.exceptions.APIError, litellm.exceptions.APIConnectionError, json.JSONDecodeError)),
)
def run_api_adapter(
    raw_question: str,
    api_responses: list[str],
    adapter_verdicts: list[str],
    episode_feedback: str,
    adapter_system_prompt: str = "",
) -> str:
    """Reflect on an API-adapter episode.

    Returns: feedback_for_adapter string.
    """
    user_content = ADAPTER_REFLECTOR_USER_TEMPLATE.format(
        adapter_system_prompt=adapter_system_prompt,
        raw_question=raw_question,
        api_responses=json.dumps(api_responses),
        adapter_verdicts=json.dumps(adapter_verdicts),
        episode_feedback=episode_feedback,
    )
    response = litellm.completion(
        model=REFLECTOR_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": ADAPTER_REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw: str = response.choices[0].message.content.strip()
    raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    parsed: dict[str, str] = json.loads(raw)
    feedback = parsed["feedback_for_adapter"]
    logger.debug(f"Adapter reflector: {feedback}")
    return feedback
