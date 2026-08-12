"""Compare different privileged information variants for SDFT teacher prompts.

For each sample, generates one student completion (base model), then computes
teacher-student advantage heatmaps under 3 privileged info strategies:
  1. golden_answer — just the correct answer
  2. golden_answer_plus_feedback — correct answer + reflector feedback
  3. feedback_only — only reflector feedback (no golden answer)

Run with the existing train_dir/.venv:

    CUDA_VISIBLE_DEVICES=6 ../train_dir/.venv/bin/python compare_privileged_info.py
"""

import copy
import json
import os

import anthropic
import numpy as np
import torch
import torch.nn.functional as F
from anthropic import AnthropicVertex
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import opd_diagnostic as od
from plot_heatmap import plot_heatmap

# ---------------------------------------------------------------------------
# Reflector config
# ---------------------------------------------------------------------------
REFLECTOR_MODEL = os.environ.get("REFLECTOR_MODEL", "claude-sonnet-4-6@default")
REFLECTOR_REGION = os.environ.get("REFLECTOR_REGION", "us-east5")
REFLECTOR_PROJECT_ID = os.environ.get("REFLECTOR_PROJECT_ID", "itpc-gcp-ai-eng-claude")

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

# ---------------------------------------------------------------------------
# Privileged info templates (appended to last user message)
# ---------------------------------------------------------------------------
VARIANTS = {
    "golden_answer": (
        "The following is the correct answer. "
        "Use this to guide your response: {answer}"
    ),
    "golden_answer_plus_feedback": (
        "Correct solution:\n{answer}\n\n"
        "The following is feedback from your earlier attempt:\n{feedback}"
    ),
    "feedback_only": (
        "The following is feedback from your earlier attempt:\n{feedback}"
    ),
    "golden_answer_plus_self_feedback": (
        "Correct solution:\n{answer}\n\n"
        "The following is feedback from your earlier attempt:\n{self_feedback}"
    ),
    "self_feedback_only": (
        "The following is feedback from your earlier attempt:\n{self_feedback}"
    ),
}

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
_client: AnthropicVertex | None = None


def _get_client() -> AnthropicVertex:
    global _client
    if _client is None:
        _client = AnthropicVertex(
            region=REFLECTOR_REGION,
            project_id=REFLECTOR_PROJECT_ID,
        )
    return _client


def call_reflector(question: str, golden_answer: str, model_response: str) -> dict[str, str]:
    """Grade model_response vs golden_answer. Returns {verdict, feedback}."""
    client = _get_client()
    user_content = REFLECTOR_USER_TEMPLATE.format(
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
    logger.info(f"Reflector: {parsed['verdict']} — {parsed['feedback']}")
    return parsed


# ---------------------------------------------------------------------------
# Self-reflector (uses the loaded base model instead of Anthropic)
# ---------------------------------------------------------------------------
def call_self_reflector(
    model, tokenizer, question: str, golden_answer: str, model_response: str,
) -> dict[str, str]:
    """Grade model_response vs golden_answer using the base model itself."""
    user_content = REFLECTOR_USER_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        model_response=model_response,
    )
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    enc = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt",
        truncation=True, max_length=2048,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # parse JSON — be lenient with the base model's output
    parsed: dict[str, str] | None = None
    for extractor in [
        lambda s: s.split("```json", 1)[1].split("```", 1)[0].strip() if "```json" in s else None,
        lambda s: s[s.index("{"):s.rindex("}") + 1] if "{" in s and "}" in s else None,
    ]:
        candidate = extractor(raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None:
        # fallback: use the raw output as feedback
        logger.warning(f"Self-reflector: could not parse JSON, using raw output as feedback: {raw[:200]}")
        parsed = {"verdict": "UNKNOWN", "feedback": raw[:200]}
    else:
        logger.info(f"Self-reflector: {parsed.get('verdict', '?')} — {parsed.get('feedback', '?')}")
    return parsed


# ---------------------------------------------------------------------------
# Teacher prompt builders
# ---------------------------------------------------------------------------
def build_teacher_prompts(
    tokenizer,
    normalized_messages: list[dict],
    golden_answer: str,
    feedback: str,
    self_feedback: str,
) -> dict[str, str]:
    """Build one teacher prompt per variant. Returns {variant_name: prompt_text}."""
    format_args = {
        "golden_answer": {"answer": golden_answer},
        "golden_answer_plus_feedback": {"answer": golden_answer, "feedback": feedback},
        "feedback_only": {"feedback": feedback},
        "golden_answer_plus_self_feedback": {"answer": golden_answer, "self_feedback": self_feedback},
        "self_feedback_only": {"self_feedback": self_feedback},
    }
    prompts = {}
    for name, template in VARIANTS.items():
        history = copy.deepcopy(normalized_messages)
        text = template.format(**format_args[name])
        history[-1]["content"] += "\n\n" + text
        prompts[name] = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    return prompts


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_advantages_for_variant(
    model,
    tokenizer,
    student_prompt: str,
    teacher_prompt: str,
    completion_ids: list[int],
) -> np.ndarray:
    """A_i = teacher_logp_i - student_logp_i."""
    student_logits = od.get_completion_logits(
        model, tokenizer, student_prompt, completion_ids, od.STUDENT_MAX_PROMPT_LEN,
    )
    teacher_logits = od.get_completion_logits(
        model, tokenizer, teacher_prompt, completion_ids, od.TEACHER_MAX_PROMPT_LEN,
    )
    s_log = F.log_softmax(student_logits.float(), dim=-1)
    t_log = F.log_softmax(teacher_logits.float(), dim=-1)

    idx = torch.arange(len(completion_ids), device=s_log.device)
    tok_ids = torch.tensor(completion_ids, device=s_log.device, dtype=torch.long)
    return (t_log[idx, tok_ids] - s_log[idx, tok_ids]).cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    output_dir = os.environ.get("OUTPUT_DIR", "heatmaps")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {od.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(od.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        od.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=DEVICE,
    )
    model.eval()

    examples = od.load_examples(od.DATA_PATH, od.NUM_SAMPLES)
    print(f"Loaded {len(examples)} examples from {od.DATA_PATH}")

    # {variant_name: [advantages_per_sample]}
    all_advantages: dict[str, list[np.ndarray]] = {name: [] for name in VARIANTS}

    for i, example in enumerate(examples):
        # -- shared across variants --
        clean_prompt = od.normalize_messages(example["prompt"])
        raw_question = clean_prompt[-1]["content"]
        answer_data = example["user_response"]
        golden_answer = (answer_data.get("value") or answer_data.get("content")).strip()

        student_prompt = tokenizer.apply_chat_template(
            clean_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        print(f"[sample {i}] generating completion...")
        completion_ids = od.generate_completion(model, tokenizer, student_prompt)
        if len(completion_ids) == 0:
            print(f"[sample {i}] empty completion, skipping")
            continue

        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # -- reflector calls --
        print(f"[sample {i}] calling reflector (anthropic)...")
        reflector_result = call_reflector(raw_question, golden_answer, completion_text)
        feedback = reflector_result["feedback"]

        print(f"[sample {i}] calling reflector (self)...")
        self_reflector_result = call_self_reflector(
            model, tokenizer, raw_question, golden_answer, completion_text,
        )
        self_feedback = self_reflector_result.get("feedback", "No feedback available.")

        # -- build teacher prompts for all variants --
        teacher_prompts = build_teacher_prompts(
            tokenizer, clean_prompt, golden_answer, feedback, self_feedback,
        )

        # -- compute advantages per variant --
        for name, teacher_prompt in teacher_prompts.items():
            adv = compute_advantages_for_variant(
                model, tokenizer, student_prompt, teacher_prompt, completion_ids,
            )
            print(
                f"  [{name}] {len(adv)} tokens, "
                f"adv range: [{adv.min():.2f}, {adv.max():.2f}], "
                f"mean: {adv.mean():.3f}"
            )
            all_advantages[name].append(adv)

    # -- plot one heatmap per variant --
    vmin = -15.0
    vmax = 3.0

    for name, advs in all_advantages.items():
        title = f"Privileged Info: {name.replace('_', ' ')}"
        out_path = os.path.join(output_dir, f"privileged_{name}.png")
        plot_heatmap(advs, title, out_path, vmin=vmin, vmax=vmax)

    print("Done.")


if __name__ == "__main__":
    main()
