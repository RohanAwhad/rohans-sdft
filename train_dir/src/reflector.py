"""Reflector: generates dynamic privileged feedback for the teacher.

Given (question, golden_answer, model_response), an external LLM grades the
response and produces a one-line feedback. Returns structured {verdict, feedback}.
"""

import json

import litellm
litellm.suppress_debug_info = True
from loguru import logger
from src.config import REFLECTOR_MODEL, llm_retry


def _extract_json(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    return json.loads(raw)


def _get_content(response) -> str:
    content = response.choices[0].message.content
    if content is not None:
        return content
    rc = getattr(response.choices[0].message, "reasoning_content", None) or ""
    return rc or ""


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



@llm_retry
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
        max_tokens=65536,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw: str = _get_content(response)
    parsed: dict[str, str] = _extract_json(raw)
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


@llm_retry
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
        max_tokens=65536,
        messages=[
            {"role": "system", "content": ADAPTER_REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw: str = _get_content(response)
    parsed: dict[str, str] = _extract_json(raw)
    feedback = parsed["feedback_for_adapter"]
    logger.debug(f"Adapter reflector: {feedback}")
    return feedback


# ---------------------------------------------------------------------------
# IFBench API-Adapter reflector
# ---------------------------------------------------------------------------

IFBENCH_ADAPTER_REFLECTOR_SYSTEM_PROMPT = """\
You are a reflection llm.
We are RL training an LLM to learn to verify output-constraint satisfaction and relay that information to the API LLM when needed.
So the LLM which we are training is called an Adapter LLM.

Because this in reinforcement learning format, I will do a rollout in an env with adapter and api llms, and provide you with their conversation.
I will also provide you with adapter's system prompt, user request with constraints, programmatic verification results, episode feedback from the env, \
adapter conversation history, and final episode answer.

The adapter's job is to check whether the API LLM's response satisfies all the output constraints in the user's request. \
These are format/structure constraints like word count ranges, sentence structures, keyword placement, etc.

Now, because it is expected that the adapter will learn constraint verification over time and store that knowledge in its weights, \
it may or may not provide justification for why it did what it did. Your job is not to ask for justification, \
but check whether the constraint verification was correct or not.


=== EXAMPLE 1 ===
# Episode Data:
- User question: Write a short story about a cat. The response must contain between 100 and 120 words.
- Constraints: The response must contain between 100 and 120 words.
- Verification results: PASS. All 1 constraints satisfied.
- Episode feedback: PASS. All 1 constraints satisfied.

### Adapter Conversation History:

User: <|USER_REQUEST_START|>
Write a short story about a cat. The response must contain between 100 and 120 words.
<|USER_REQUEST_END|>

<|LLM_RESPONSE_START|>
[A 110-word story about a cat]
<|LLM_RESPONSE_END|>

=== EXAMPLE 2 ===
# Episode Data:
- User question: Explain photosynthesis. Use at least 5 different coordinating conjunctions in the response. Please use an emoji at the end of every sentence.
- Constraints: Use at least 5 different coordinating conjunctions. Please use an emoji at the end of every sentence.
- Verification results: FAIL. Constraints: 1/2. Failed: format:emoji
- Episode feedback: FAIL. Constraints: 1/2. Failed: format:emoji

### Adapter Conversation History:

User: <|USER_REQUEST_START|>
Explain photosynthesis. Use at least 5 different coordinating conjunctions in the response. Please use an emoji at the end of every sentence.
<|USER_REQUEST_END|>

<|LLM_RESPONSE_START|>
[Response without emojis at end of sentences]
<|LLM_RESPONSE_END|>

=== YOUR TURN ===
Output EXACTLY this JSON and nothing else:
```json
{"feedback_for_adapter": "one sentence, max 30 words"}
```"""

IFBENCH_ADAPTER_REFLECTOR_USER_TEMPLATE = """\
# Episode Data:
- Adapter system prompt: {adapter_system_prompt}
- User question: {raw_question}
- Constraints: {constraint_description}
- Verification results: {verification_results}
- Episode feedback: {episode_feedback}

### Adapter Conversation History:

{adapter_conversation_history}

### Final Episode Response

{episode_answer}

---

Based on this conversation, give feedback for adapter's last turn, such that adapter can correctly identify constraint violations in one shot.
You can dump as much information as you can fit in 30 words about the adapter's response.
""".strip()


@llm_retry
def run_ifbench_adapter(
    raw_question: str,
    constraint_description: str,
    verification_results: str,
    adapter_conversation_history: str,
    episode_answer: str,
    episode_feedback: str,
    adapter_system_prompt: str = "",
) -> str:
    """Reflect on an IFBench API-adapter episode.

    Returns: feedback_for_adapter string.
    """
    user_content = IFBENCH_ADAPTER_REFLECTOR_USER_TEMPLATE.format(
        adapter_system_prompt=adapter_system_prompt,
        raw_question=raw_question,
        constraint_description=constraint_description,
        verification_results=verification_results,
        adapter_conversation_history=adapter_conversation_history,
        episode_answer=episode_answer,
        episode_feedback=episode_feedback,
    )
    response = litellm.completion(
        model=REFLECTOR_MODEL,
        max_tokens=65536,
        messages=[
            {"role": "system", "content": IFBENCH_ADAPTER_REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw: str = _get_content(response)
    parsed: dict[str, str] = _extract_json(raw)
    feedback = parsed["feedback_for_adapter"]
    logger.debug(f"IFBench adapter reflector: {feedback}")
    return feedback
