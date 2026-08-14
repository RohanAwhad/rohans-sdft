# /// script
# dependencies = [
#   "transformers",
#   "litellm",
#   "tenacity",
#   "loguru",
#   "requests",
#   "google-cloud-aiplatform",
#   "torch",
# ]
# ///

import os
os.environ["MODEL_NAME"] = "/mnt/nvme7n1/rawhad/sdft_api_adapter/synthetic_algebra/sdft_test_run_2/step_100"
os.environ["VLLM_PORT"] = "8024"
os.environ["VERTEXAI_LOCATION"] = "us-east5"

import sys
sys.path.insert(0, "train_dir")

from transformers import AutoTokenizer
from src.env.api_adapter_env import ApiAdapterEnv

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

import json
with open("train_dir/data/synthetic_algebra/train_sdft.jsonl") as f:
    lines = f.readlines()
row = json.loads(lines[1])  # 2nd row (0-indexed)
raw_question = row["prompt"][0]["value"]
golden_answer = row["user_response"]["value"]
print(f"Question: {raw_question}")
print(f"Golden answer: {golden_answer}\n")

env = ApiAdapterEnv(
    prompt_text="",
    vllm_base_url="http://localhost:8024",
    raw_question=raw_question,
    golden_answer=golden_answer,
    tokenizer=tokenizer,
)

print("=== Starting rollout ===\n")
result = env.rollout(raw_question)

print(f"\n{'='*60}")
print(f"Rollout returned: {result[:300] if result else None}")
print(f"Verdict: {env.verdict}")
print(f"Feedback: {env.feedback}")

print(f"\n{'='*60}")
print(f"=== Adapter History ({len(env.adapter_history)} messages) ===")
for i, msg in enumerate(env.adapter_history):
    print(f"\n--- [{msg['role']}] ---")
    print(msg["content"])

print(f"\n{'='*60}")
print(f"=== API History ({len(env.api_history)} messages) ===")
for i, msg in enumerate(env.api_history):
    print(f"\n--- [{msg['role']}] ---")
    print(msg["content"][:800])

# --- Privileged information call ---
import copy
from src.env.api_adapter_env import HINDSIGHT_TEMPLATE

# Evaluate to get real verdict/feedback
model_answer = env.parse_model_answer(result) if result else None
if model_answer is not None:
    env.evaluate(model_answer, golden_answer)

verdict_str = "PASS" if env.verdict else "FAIL"
hindsight = HINDSIGHT_TEMPLATE.format(llm_response=result, feedback=env.feedback)

cond_history = copy.deepcopy(env.adapter_history[:-1])
cond_history[-1]["content"] += "\n\n" + hindsight

print(f"\n{'='*60}")
print(f"=== Privileged call (hindsight on last user msg) ===")
print(f"Hindsight: {hindsight}")

prompt_text = tokenizer.apply_chat_template(
    cond_history, tokenize=False, add_generation_prompt=True, enable_thinking=True,
)

from src.vllm_utils import vllm_generate
from src.config import THINKING_BUDGET, GEN_MAX_NEW_TOKENS

text, finish_reason = vllm_generate(prompt_text, base_url="http://localhost:8024", max_tokens=THINKING_BUDGET)
if finish_reason == "length" and "</think>" not in text:
    text = text.rstrip() + ".\n</think>\n\n"
    answer_text, _ = vllm_generate(
        prompt_text + text, base_url="http://localhost:8024", max_tokens=GEN_MAX_NEW_TOKENS - THINKING_BUDGET,
    )
    text += answer_text

print(f"\n--- Privileged adapter response ---")
print(text)
