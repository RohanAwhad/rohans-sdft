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
You are a reflection llm.
We are RL training an LLM to learn the user preferences from training data and relay that information to the API LLM when needed.
So the LLM which we are training is called an Adapter LLM.

Because this in reinforcement learning format, I will do a rollout in an env with adapter and api llms, and provide you with their conversation.
I will also provide you with adapter's system prompt, user request, golden answer, episode feedback from the env, adapter conversation history,
and final episode answer.

Now, because it is expected that the adapter will learn user preferences over time and store that knowledge in its weights, it may or may not provide \
justification for why it did what it did. Your job is not to ask for justification, but check whether the learned preference is correct or not.


=== EXAMPLE 1 ===
# Episode Data:
- User question: What is 5 * 9?
- Golden answer: 45
- Episode feedback: PASS. Model generated 45. Correct answer is 45

### Adapter Conversation History:

User: <|USER_REQUEST_START|>
What is 5 * 9?
<|USER_REQUEST_END|>

<|LLM_RESPONSE_START|>
5 * 9 = 45, so the answer is \boxed{45}
<|LLM_RESPONSE_END|>

=== EXAMPLE 2 ===
# Episode Data:
- User question: What is 3 - 9?
- Golden answer: -6
- Episode feedback: FAIL. Model generated 6. Correct answer was -6

### Adapter Conversation History:

User: <|USER_REQUEST_START|>
What is 3 - 9?
<|USER_REQUEST_END|>

<|LLM_RESPONSE_START|>
3 - 9 = 6, so the answer is \boxed{6}
<|LLM_RESPONSE_END|>

=== YOUR TURN ===
Output EXACTLY this JSON and nothing else:
```json
{"feedback_for_adapter": "one sentence, max 30 words"}
```"""

ADAPTER_REFLECTOR_USER_TEMPLATE = """\
# Episode Data:
- Adapter system prompt: {adapter_system_prompt}
- User question: {raw_question}
- Golden answer: {golden_answer}
- Episode feedback: {episode_feedback}

### Adapter Conversation History:

{adapter_conversation_history}

### Final Episode Response

{episode_answer}

---

Based on this conversation, give feedback for adapter's last turn, such that adapter can respond to api for the correct answer in one short.
You can dump as much information as you can fit in 30 words about the adapter's response.
""".strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.2, max=10),
    retry=retry_if_exception_type((litellm.exceptions.APIError, litellm.exceptions.APIConnectionError, json.JSONDecodeError)),
)
def run_api_adapter(
    raw_question: str,
    golden_answer: str,
    adapter_conversation_history: str,
    episode_answer: str,
    episode_feedback: str,
    adapter_system_prompt: str = "",
) -> str:
    """Reflect on an API-adapter episode.

    Returns: feedback_for_adapter string.
    """
    user_content = ADAPTER_REFLECTOR_USER_TEMPLATE.format(
        adapter_system_prompt=adapter_system_prompt,
        raw_question=raw_question,
        golden_answer=golden_answer,
        adapter_conversation_history=adapter_conversation_history,
        episode_answer=episode_answer,
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
