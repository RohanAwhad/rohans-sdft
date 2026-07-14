"""Synthetic algebra dataset generator.

Generates arithmetic expressions with custom Greek symbols mapped to basic ops.
Symbol mapping (fixed):
    θ (theta) → +     α (alpha) → -
    γ (gamma) → ×     β (beta)  → ÷

Two output formats:
    Raw:   {"expression": "...", "answer": 42, "type": "custom"}
    MaaS:  {"prompt": [...], "user_response": {...}}  (--maas flag)

Usage:
    python synthetic_algebra_dataset.py [--n-custom 1000] [--n-standard 1000] [--seed 42] [--output data] [--maas]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CUSTOM_SYMBOLS = {"θ": "+", "α": "-", "γ": "*", "β": "/"}
STANDARD_OPS = ["+", "-", "*", "/"]
HIGH_PRECEDENCE = {"*", "/"}
LOW_PRECEDENCE = {"+", "-"}

SYMBOL_DEFINITIONS = (
    "Custom arithmetic symbols:\n"
    "  θ means addition (+)\n"
    "  α means subtraction (-)\n"
    "  γ means multiplication (×)\n"
    "  β means division (÷)\n"
    "Precedence: β and γ bind tighter than θ and α (same as standard BODMAS)."
)

SYMBOL_DEFINITIONS_VAGUE = (
    "The symbols θ, α, γ, β each represent one of the four basic arithmetic "
    "operations (+, -, ×, ÷). Each symbol maps to exactly one operation. "
    "Standard operator precedence (BODMAS) applies."
)

SYSTEM_PROMPT = (
    "You are an adapter that checks an API model's arithmetic answer. "
    "If it is correct, respond \\boxed{CORRECT} and it will be sent directly to the user. "
    "If wrong or missing, provide guidance back to the API on how to fix it "
    "(do not compute the answer yourself)."
)

FEW_SHOT_EXAMPLES = (
    "The symbols θ, α, γ, β each represent one of the four basic arithmetic "
    "operations (+, -, ×, ÷). Each symbol maps to exactly one operation. "
    "Standard operator precedence (BODMAS) applies.\n\n"
    "Examples:\n"
    "Expression: 3 θ 4 | API answer: 7 → \\boxed{CORRECT}\n"
    "Expression: 10 α 3 | API answer: 5 → That's incorrect. Recount step by step.\n"
    "Expression: 2 γ 6 | API answer: none → Compute the expression using the symbol definitions."
)

USER_RESPONSE_TEMPLATE = "θ=+, α=-, γ=×, β=÷. Correct answer: {answer}"


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in expr.replace(" ", ""):
        if ch.isdigit():
            current += ch
        else:
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
    if current:
        tokens.append(current)
    return tokens


def _to_standard(tokens: list[str]) -> list[str]:
    return [CUSTOM_SYMBOLS.get(t, t) for t in tokens]


def evaluate(expr: str) -> int:
    """Evaluate an arithmetic expression (custom or standard symbols)."""
    tokens = _to_standard(_tokenize(expr))

    nums: list[float] = []
    ops: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i].lstrip("-").isdigit():
            nums.append(float(tokens[i]))
        else:
            ops.append(tokens[i])
        i += 1

    new_nums: list[float] = [nums[0]]
    new_ops: list[str] = []
    for j, op in enumerate(ops):
        if op in HIGH_PRECEDENCE:
            left = new_nums.pop()
            right = nums[j + 1]
            if op == "*":
                new_nums.append(left * right)
            else:
                new_nums.append(left / right)
        else:
            new_nums.append(nums[j + 1])
            new_ops.append(op)

    result = new_nums[0]
    for j, op in enumerate(new_ops):
        if op == "+":
            result += new_nums[j + 1]
        else:
            result -= new_nums[j + 1]

    return int(result)


# ---------------------------------------------------------------------------
# API answer generation (for MaaS format)
# ---------------------------------------------------------------------------

CUSTOM_OP_SYMBOLS = list(CUSTOM_SYMBOLS.keys())


def _generate_wrong_answer(
    answer: int, expr: str, use_custom: bool, rng: random.Random
) -> int:
    """Generate a plausible-but-wrong API answer.

    For custom expressions: swap one operator to a different symbol, re-evaluate.
    For standard: add/subtract a random offset [1, 20].
    Falls back to offset if operator swap produces the correct answer.
    """
    if use_custom:
        tokens = _tokenize(expr)
        op_indices = [i for i, t in enumerate(tokens) if t in CUSTOM_OP_SYMBOLS]
        if op_indices:
            idx = rng.choice(op_indices)
            original = tokens[idx]
            replacements = [s for s in CUSTOM_OP_SYMBOLS if s != original]
            tokens[idx] = rng.choice(replacements)
            try:
                wrong = evaluate(" ".join(tokens))
                if wrong != answer:
                    return wrong
            except (ZeroDivisionError, ValueError):
                pass

    # Fallback: random offset
    for _ in range(50):
        offset = rng.randint(1, 20) * rng.choice([-1, 1])
        wrong = answer + offset
        if wrong >= 0 and wrong != answer:
            return wrong
    return answer + 1


def generate_api_answer(
    answer: int, expr: str, use_custom: bool, rng: random.Random
) -> tuple[str, bool]:
    """Return (api_answer_text, is_correct).

    CORRECT: 30% — api_answer equals ground truth
    WRONG:   50% — plausible but wrong answer
    NONE:    20% — api answer not provided
    """
    roll = rng.random()
    if roll < 0.30:
        return str(answer), True
    elif roll < 0.80:
        wrong = _generate_wrong_answer(answer, expr, use_custom, rng)
        return str(wrong), False
    else:
        return "none", False


def build_maas_sample(
    expr: str,
    answer: int,
    api_answer: str,
    is_correct: bool,
) -> dict:
    """Build a MaaS-format sample with prompt and user_response."""
    user_text = (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Expression: {expr} | API answer: {api_answer} →"
    )
    user_response_text = USER_RESPONSE_TEMPLATE.format(answer=answer)

    return {
        "prompt": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": user_text},
        ],
        "user_response": {"value": user_response_text},
        "_meta": {
            "expression": expr,
            "answer": answer,
            "api_answer": api_answer,
            "is_correct": is_correct,
            "type": "custom" if any(s in expr for s in CUSTOM_OP_SYMBOLS) else "standard",
        },
    }


def generate_expression(
    num_operands: int,
    use_custom: bool,
    rng: random.Random | None = None,
) -> tuple[str, int]:
    """Generate a random arithmetic expression with integer result.

    Args:
        num_operands: Number of operands (2-4).
        use_custom: If True, use custom symbols (θ, α, γ, β). Otherwise standard (+, -, *, /).
        rng: Random instance for reproducibility.

    Returns:
        (expression_string, correct_answer)
    """
    if rng is None:
        rng = random.Random()

    custom_ops = list(CUSTOM_SYMBOLS.keys())
    max_attempts = 1000

    for _ in range(max_attempts):
        operands = [rng.randint(1, 99) for _ in range(num_operands)]
        if use_custom:
            operators = [rng.choice(custom_ops) for _ in range(num_operands - 1)]
        else:
            operators = [rng.choice(STANDARD_OPS) for _ in range(num_operands - 1)]

        parts = [str(operands[0])]
        for k in range(len(operators)):
            parts.append(operators[k])
            parts.append(str(operands[k + 1]))
        expr = " ".join(parts)

        try:
            result = evaluate(expr)
        except (ZeroDivisionError, ValueError):
            continue

        tokens = _to_standard(_tokenize(expr))
        nums_f: list[float] = []
        ops_f: list[str] = []
        for t in tokens:
            if t.lstrip("-").isdigit():
                nums_f.append(float(t))
            else:
                ops_f.append(t)

        new_nums_f: list[float] = [nums_f[0]]
        new_ops_f: list[str] = []
        valid = True
        for j, op in enumerate(ops_f):
            if op in HIGH_PRECEDENCE:
                left = new_nums_f.pop()
                right = nums_f[j + 1]
                if op == "/" and right == 0:
                    valid = False
                    break
                val = left * right if op == "*" else left / right
                if op == "/" and val != int(val):
                    valid = False
                    break
                new_nums_f.append(val)
            else:
                new_nums_f.append(nums_f[j + 1])
                new_ops_f.append(op)

        if not valid:
            continue

        float_result = new_nums_f[0]
        for j, op in enumerate(new_ops_f):
            if op == "+":
                float_result += new_nums_f[j + 1]
            else:
                float_result -= new_nums_f[j + 1]

        if float_result != int(float_result):
            continue

        return expr, int(float_result)

    raise RuntimeError(f"Failed to generate valid expression after {max_attempts} attempts")


def generate_dataset(
    n_custom: int = 1000,
    n_standard: int = 1000,
    seed: int = 42,
    train_ratio: float = 0.8,
    maas: bool = False,
) -> dict[str, list[dict]]:
    """Generate arithmetic dataset with stratified train/test split.

    If maas=True, samples include prompt + user_response for SDFT training.
    If maas=False, samples are raw {expression, answer, type} dicts.
    """
    rng = random.Random(seed)

    custom_samples: list[dict] = []
    for _ in range(n_custom):
        num_ops = rng.randint(2, 4)
        expr, answer = generate_expression(num_ops, use_custom=True, rng=rng)
        custom_samples.append({"expression": expr, "answer": answer, "type": "custom"})

    standard_samples: list[dict] = []
    for _ in range(n_standard):
        num_ops = rng.randint(2, 4)
        expr, answer = generate_expression(num_ops, use_custom=False, rng=rng)
        standard_samples.append({"expression": expr, "answer": answer, "type": "standard"})

    rng.shuffle(custom_samples)
    rng.shuffle(standard_samples)

    n_custom_train = int(len(custom_samples) * train_ratio)
    n_standard_train = int(len(standard_samples) * train_ratio)

    train = custom_samples[:n_custom_train] + standard_samples[:n_standard_train]
    test = custom_samples[n_custom_train:] + standard_samples[n_standard_train:]

    rng.shuffle(train)
    rng.shuffle(test)

    if maas:
        train = [_to_maas_sample(s, rng) for s in train]
        test = [_to_maas_sample(s, rng) for s in test]

    return {"train": train, "test": test}


def _to_maas_sample(raw: dict, rng: random.Random) -> dict:
    """Convert a raw sample to MaaS format with prompt + user_response."""
    expr: str = raw["expression"]
    answer: int = raw["answer"]
    use_custom = raw["type"] == "custom"
    api_answer, is_correct = generate_api_answer(answer, expr, use_custom, rng)
    return build_maas_sample(expr, answer, api_answer, is_correct)


def save_dataset(dataset: dict[str, list[dict]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in dataset.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic algebra dataset")
    parser.add_argument("--n-custom", type=int, default=1000)
    parser.add_argument("--n-standard", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument(
        "--maas",
        action="store_true",
        help="Generate MaaS-format dataset (prompt + user_response) for SDFT training",
    )
    args = parser.parse_args()

    fmt = "MaaS" if args.maas else "raw"
    print(f"Generating dataset (custom={args.n_custom}, standard={args.n_standard}, seed={args.seed}, format={fmt})...")
    dataset = generate_dataset(
        n_custom=args.n_custom,
        n_standard=args.n_standard,
        seed=args.seed,
        maas=args.maas,
    )
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Test:  {len(dataset['test'])} samples")

    save_dataset(dataset, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
