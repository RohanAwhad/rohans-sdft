"""Adapter evaluation script.

Two modes:
  Mode 1 (binary): classify adapter output (CORRECT vs guidance), compute metrics.
  Mode 2 (two-pass): feed adapter guidance back to API, check if error was fixed.

Usage:
    python task_3/eval.py --ckpt train_dir/output/step_400 --data data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthetic_algebra_dataset import (
    CUSTOM_SYMBOLS,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    USER_RESPONSE_TEMPLATE,
    evaluate,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CORRECT_MARKER = r"\boxed{CORRECT}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_message_content(msg: dict) -> str:
    return msg.get("value") or msg.get("content", "")


def _build_student_prompt(sample: dict) -> str:
    messages = sample["prompt"]
    parts = [_extract_message_content(m) for m in messages]
    return "\n\n".join(parts)


def _has_correct(completion: str) -> bool:
    return CORRECT_MARKER.replace("\\", "") in completion or CORRECT_MARKER in completion


def _parse_int(text: str) -> Optional[int]:
    """Extract the last integer from text (API retry answer)."""
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


def _format_retry_prompt(expression: str, api_answer: str, guidance: str) -> str:
    """Build prompt for API retry: previous answer + guidance → recompute."""
    mapping = "θ=+, α=-, γ=×, β=÷"
    return (
        f"{mapping}\n\n"
        f"Previous answer was {api_answer}. "
        f"The checker said: {guidance}\n\n"
        f"Solve: {expression}\n"
        f"Answer with just the number."
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Mode 1: Binary classification
# ---------------------------------------------------------------------------


def run_binary_eval(model, tokenizer, samples: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    results: list[dict] = []

    for i, sample in enumerate(samples):
        meta = sample["_meta"]
        actually_correct = meta["is_correct"]
        prompt_text = _build_student_prompt(sample)
        completion = generate(model, tokenizer, prompt_text)
        predicted_correct = _has_correct(completion)

        if actually_correct and predicted_correct:
            tn += 1
        elif actually_correct and not predicted_correct:
            fp += 1
        elif not actually_correct and predicted_correct:
            fn += 1
        else:
            tp += 1

        results.append({
            "expr": meta["expression"],
            "type": meta["type"],
            "api_answer": meta["api_answer"],
            "correct_answer": meta["answer"],
            "actually_correct": actually_correct,
            "predicted_correct": predicted_correct,
            "completion": completion[:200],
        })

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(samples)}")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    return {
        "total": total,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Mode 2: Two-pass guidance quality
# ---------------------------------------------------------------------------


def run_guidance_eval(model, tokenizer, samples: list[dict]) -> dict:
    """For samples where the adapter gave guidance (not CORRECT), feed
    guidance back to the API model and check if the retry is correct.

    NOTE: This metric is confounded — it mixes guidance quality with the
    API model's execution capability. The API model may fail to follow good
    guidance, or self-correct despite bad guidance.
    """
    corrected = 0
    total_wrong = 0
    results: list[dict] = []

    for i, sample in enumerate(samples):
        meta = sample["_meta"]
        actually_correct = meta["is_correct"]
        if actually_correct:
            continue

        total_wrong += 1
        prompt_text = _build_student_prompt(sample)
        completion = generate(model, tokenizer, prompt_text)

        if _has_correct(completion):
            # Adapter missed the error — no guidance to evaluate
            results.append({
                "expr": meta["expression"],
                "type": meta["type"],
                "correct_answer": meta["answer"],
                "api_answer": meta["api_answer"],
                "guidance": completion[:200],
                "retry_correct": False,
                "missed": True,
            })
            continue

        # Adapter gave guidance — feed to API retry
        retry_prompt = _format_retry_prompt(
            meta["expression"], meta["api_answer"], completion
        )
        retry_completion = generate(model, tokenizer, retry_prompt, max_new_tokens=64)
        retry_int = _parse_int(retry_completion)
        retry_correct = retry_int == meta["answer"]

        if retry_correct:
            corrected += 1

        results.append({
            "expr": meta["expression"],
            "type": meta["type"],
            "correct_answer": meta["answer"],
            "api_answer": meta["api_answer"],
            "guidance": completion[:200],
            "retry_output": retry_completion[:200],
            "retry_parsed": retry_int,
            "retry_correct": retry_correct,
            "missed": False,
        })

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(samples)}")

    guidance_rate = corrected / total_wrong if total_wrong else 0

    return {
        "total_wrong": total_wrong,
        "corrected": corrected,
        "guidance_corrected_accuracy": guidance_rate,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Per-type breakdown
# ---------------------------------------------------------------------------


def per_type_breakdown(binary_results: dict) -> dict:
    by_type: Dict[str, dict] = {}
    for r in binary_results["results"]:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}
        bucket = by_type[t]
        bucket["total"] += 1
        if r["actually_correct"] and r["predicted_correct"]:
            bucket["tn"] += 1
        elif r["actually_correct"] and not r["predicted_correct"]:
            bucket["fp"] += 1
        elif not r["actually_correct"] and r["predicted_correct"]:
            bucket["fn"] += 1
        else:
            bucket["tp"] += 1

    breakdown = {}
    for t, b in by_type.items():
        breakdown[t] = {
            "accuracy": (b["tp"] + b["tn"]) / b["total"] if b["total"] else 0,
            "precision": b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 0,
            "recall": b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0,
        }
    return breakdown


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDFT adapter")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--mode", type=str, choices=["all", "binary", "guidance"], default="all")
    parser.add_argument("--output", type=str, help="Path to save detailed results (JSON)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {DEVICE}")

    # Load test data
    test_samples: list[dict] = []
    with open(args.data) as f:
        for line in f:
            test_samples.append(json.loads(line))
    print(f"Loaded {len(test_samples)} test samples")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=DEVICE,
    )
    model.eval()
    print("Model loaded.")

    # --- Mode 1: Binary classification ---
    if args.mode in ("all", "binary"):
        print("\n=== Mode 1: Binary Classification ===")
        binary = run_binary_eval(model, tokenizer, test_samples)
        print(f"\nTotal: {binary['total']}")
        print(f"  TP: {binary['TP']}  TN: {binary['TN']}  FP: {binary['FP']}  FN: {binary['FN']}")
        print(f"  Accuracy:  {binary['accuracy']:.4f}")
        print(f"  Precision: {binary['precision']:.4f}")
        print(f"  Recall:    {binary['recall']:.4f}")

        print("\nPer-type breakdown:")
        for t, m in per_type_breakdown(binary).items():
            print(f"  {t:10s}: acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}")

    # --- Mode 2: Two-pass guidance ---
    if args.mode in ("all", "guidance"):
        print("\n=== Mode 2: Two-Pass Guidance Quality ===")
        print("(NOTE: confounded — mixes guidance quality with API execution capability)")
        guidance = run_guidance_eval(model, tokenizer, test_samples)
        print(f"\nWrong answers: {guidance['total_wrong']}")
        print(f"Corrected after guidance: {guidance['corrected']}")
        print(f"Guidance-corrected rate: {guidance['guidance_corrected_accuracy']:.4f}")

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = {}
        if args.mode in ("all", "binary"):
            to_save["binary"] = {"summary": {k: v for k, v in binary.items() if k != "results"}, "results": binary["results"]}
        if args.mode in ("all", "guidance"):
            to_save["guidance"] = {"summary": {k: v for k, v in guidance.items() if k != "results"}, "results": guidance["results"]}
        with open(output_path, "w") as f:
            json.dump(to_save, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
