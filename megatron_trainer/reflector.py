"""Reflector: generates dynamic privileged feedback for the teacher.

Given (question, golden_answer, model_response), an external LLM grades the
response and produces a one-line feedback. Returns structured {verdict, feedback}.
"""

import json

import anthropic
from anthropic import AnthropicVertex
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from megatron_trainer.config import REFLECTOR_MODEL, REFLECTOR_REGION, REFLECTOR_PROJECT_ID


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


_client: AnthropicVertex | None = None


def _get_client() -> AnthropicVertex:
    global _client
    if _client is None:
        _client = AnthropicVertex(
            region=REFLECTOR_REGION,
            project_id=REFLECTOR_PROJECT_ID,
        )
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.2, max=10),
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, json.JSONDecodeError)),
)
def run(question: str, golden_answer: str, model_response: str) -> dict[str, str]:
    """Reflect on model_response vs golden_answer.

    Returns: {"verdict": "PASS"|"FAIL", "feedback": "one line reason"}
    """
    client = _get_client()
    user_content: str = REFLECTOR_USER_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        model_response=model_response,
    )
    response = client.messages.create(
        model=REFLECTOR_MODEL,
        max_tokens=1024,
        system=REFLECTOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw: str = response.content[0].text.strip()
    raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    parsed: dict[str, str] = json.loads(raw)
    logger.debug(f"Reflector: {parsed['verdict']} — {parsed['feedback']}")
    return parsed
