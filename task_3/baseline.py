#!/usr/bin/env python
"""Baseline evaluation for Qwen3-8B on the algebra adapter task.

1. API model accuracy: can Qwen3-8B solve the expressions directly?
2. Adapter accuracy: can Qwen3-8B (as adapter) detect wrong API answers?

Usage:
    .ds_gen_venv/bin/python baseline.py --data data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda:0")

API_PROMPT = (
    "The symbols θ, α, γ, β each represent one of the four basic arithmetic "
    "operations (+, -, ×, ÷). Each symbol maps to exactly one operation. "
    "Standard operator precedence (BODMAS) applies.\n\n"
    "Compute: {expression}\n"
    "Answer with just the number."
)

ADAPTER_SYSTEM = (
    "You are an adapter that checks an API model's arithmetic answer. "
    "If it is correct, respond \\boxed{CORRECT} and it will be sent directly to the user. "
    "If wrong or missing, provide guidance back to the API on how to fix it "
    "(do not compute the answer yourself)."
)


def _extract_content(msg: dict) -> str:
    return msg.get("value") or msg.get("content", "")


def _build_adapter_prompt(sample: dict) -> str:
    messages = sample["prompt"]
    return "\n\n".join(_extract_content(m) for m in messages)


def _parse_int(text: str) -> int | None:
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


def _has_correct(text: str) -> bool:
    return "\\boxed{CORRECT}" in text or "boxed{CORRECT}" in text


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)


def baseline_api(model, tokenizer, samples: list[dict]) -> dict:
    """Qwen3-8B as the API model — computes expressions directly."""
    correct_custom = 0
    correct_standard = 0
    total_custom = 0
    total_standard = 0

    for i, s in enumerate(samples):
        meta = s["_meta"]
        expr = meta["expression"]
        is_custom = meta["type"] == "custom"

        prompt = API_PROMPT.format(expression=expr)
        completion = generate(model, tokenizer, prompt)
        parsed = _parse_int(completion)

        if parsed == meta["answer"]:
            if is_custom:
                correct_custom += 1
            else:
                correct_standard += 1

        if is_custom:
            total_custom += 1
        else:
            total_standard += 1

        if (i + 1) % 50 == 0:
            print(f"  API baseline ... {i + 1}/{len(samples)}")

    return {
        "custom": {"correct": correct_custom, "total": total_custom, "acc": correct_custom / total_custom if total_custom else 0},
        "standard": {"correct": correct_standard, "total": total_standard, "acc": correct_standard / total_standard if total_standard else 0},
    }


def baseline_adapter(model, tokenizer, samples: list[dict]) -> dict:
    """Qwen3-8B as the adapter — binary classification of API answers."""
    tp = fp = tn = fn = 0
    custom_tp = custom_fp = custom_tn = custom_fn = 0
    std_tp = std_fp = std_tn = std_fn = 0

    for i, s in enumerate(samples):
        meta = s["_meta"]
        actually_correct = meta["is_correct"]
        is_custom = meta["type"] == "custom"

        prompt = _build_adapter_prompt(s)
        completion = generate(model, tokenizer, prompt)
        predicted_correct = _has_correct(completion)

        if actually_correct and predicted_correct:
            tn += 1
            if is_custom: custom_tn += 1
            else: std_tn += 1
        elif actually_correct and not predicted_correct:
            fp += 1
            if is_custom: custom_fp += 1
            else: std_fp += 1
        elif not actually_correct and predicted_correct:
            fn += 1
            if is_custom: custom_fn += 1
            else: std_fn += 1
        else:
            tp += 1
            if is_custom: custom_tp += 1
            else: std_tp += 1

        if (i + 1) % 50 == 0:
            print(f"  Adapter baseline ... {i + 1}/{len(samples)}")

    total = tp + fp + tn + fn
    return {
        "overall": {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "total": total,
            "accuracy": (tp + tn) / total if total else 0,
            "precision": tp / (tp + fp) if (tp + fp) else 0,
            "recall": tp / (tp + fn) if (tp + fn) else 0,
        },
        "custom": {
            "TP": custom_tp, "TN": custom_tn, "FP": custom_fp, "FN": custom_fn,
            "total": custom_tp + custom_fp + custom_tn + custom_fn,
            "accuracy": (custom_tp + custom_tn) / (custom_tp + custom_fp + custom_tn + custom_fn) if (custom_tp + custom_fp + custom_tn + custom_fn) else 0,
        },
        "standard": {
            "TP": std_tp, "TN": std_tn, "FP": std_fp, "FN": std_fn,
            "total": std_tp + std_fp + std_tn + std_fn,
            "accuracy": (std_tp + std_tn) / (std_tp + std_fp + std_tn + std_fn) if (std_tp + std_fp + std_tn + std_fn) else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")

    # Load data
    samples: list[dict] = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Test samples: {len(samples)}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=DEVICE,
    )
    model.eval()
    print("Model loaded.")

    # 1. API model accuracy
    print("\n=== API Model Accuracy (Qwen3-8B computing expressions) ===")
    api = baseline_api(model, tokenizer, samples)
    print(f"  Custom:   {api['custom']['correct']}/{api['custom']['total']} = {api['custom']['acc']:.1%}")
    print(f"  Standard: {api['standard']['correct']}/{api['standard']['total']} = {api['standard']['acc']:.1%}")

    # 2. Adapter classification accuracy (binary)
    print("\n=== Adapter Accuracy (Qwen3-8B detecting wrong API answers) ===")
    adapter = baseline_adapter(model, tokenizer, samples)
    print(f"  Overall:  acc={adapter['overall']['accuracy']:.1%} prec={adapter['overall']['precision']:.1%} rec={adapter['overall']['recall']:.1%}")
    print(f"    TP={adapter['overall']['TP']} TN={adapter['overall']['TN']} FP={adapter['overall']['FP']} FN={adapter['overall']['FN']}")
    print(f"  Custom:   acc={adapter['custom']['accuracy']:.1%}")
    print(f"  Standard: acc={adapter['standard']['accuracy']:.1%}")

    # Save
    output = {"api": api, "adapter": adapter}
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to baseline_results.json")


if __name__ == "__main__":
    main()
