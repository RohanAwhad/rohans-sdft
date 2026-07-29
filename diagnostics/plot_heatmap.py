"""Plot SDPO-style token-level advantage heatmaps for SDFT diagnostics.

For each on-policy rollout (x, y, o):
    A_i = log pi(y_i | x, o, y_<i) - log pi(y_i | x, y_<i)
        = teacher_logprob_i - student_logprob_i

A_i > 0: teacher (sees hindsight o) agrees with the sampled token more than
         the student itself does -- reinforce.
A_i < 0: teacher thinks the sampled token was a mistake given hindsight --
         correct.

Reuses generation/forward-pass helpers from opd_diagnostic.py (same dir) --
does its own on-policy rollout + forward passes (standalone recompute, not
reading opd_diagnostic/*.json).

Run with the existing train_dir/.venv:

    CUDA_VISIBLE_DEVICES=6 ../train_dir/.venv/bin/python plot_heatmap.py

Student/teacher can be different checkpoints (e.g. a trained student vs the
original base model as teacher):

    CUDA_VISIBLE_DEVICES=6 ../train_dir/.venv/bin/python plot_heatmap.py \\
        --student-model /path/to/step_113 \\
        --teacher-model Qwen/Qwen3-8B
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from transformers import AutoModelForCausalLM, AutoTokenizer

import opd_diagnostic as od


def compute_advantages(student_model, teacher_model, tokenizer, example: dict) -> np.ndarray:
    """A_i = teacher_logprob_i - student_logprob_i for one on-policy rollout.

    student_model and teacher_model may be different checkpoints (e.g. a
    fine-tuned student vs the original base model as teacher).
    """
    student_prompt, teacher_prompt, _, _ = od.build_prompts(tokenizer, example)

    completion_ids = od.generate_completion(student_model, tokenizer, student_prompt)
    if len(completion_ids) == 0:
        return np.array([])

    student_logits = od.get_completion_logits(
        student_model, tokenizer, student_prompt, completion_ids, od.STUDENT_MAX_PROMPT_LEN,
    )
    teacher_logits = od.get_completion_logits(
        teacher_model, tokenizer, teacher_prompt, completion_ids, od.TEACHER_MAX_PROMPT_LEN,
    )

    s_log = F.log_softmax(student_logits.float(), dim=-1)
    t_log = F.log_softmax(teacher_logits.float(), dim=-1)

    idx = torch.arange(len(completion_ids), device=s_log.device)
    tok_ids = torch.tensor(completion_ids, device=s_log.device, dtype=torch.long)
    student_logp = s_log[idx, tok_ids]
    teacher_logp = t_log[idx, tok_ids]

    return (teacher_logp - student_logp).cpu().numpy()


def plot_heatmap(
    all_advantages: list[np.ndarray],
    title: str,
    output_path: str,
    vmin: float,
    vmax: float,
) -> None:
    max_len = max(len(a) for a in all_advantages)
    n_examples = len(all_advantages)

    matrix = np.full((n_examples, max_len), np.nan)
    for i, adv in enumerate(all_advantages):
        matrix[i, : len(adv)] = adv

    fig, ax = plt.subplots(figsize=(14, max(3, n_examples * 0.4 + 1)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    masked = np.ma.masked_invalid(matrix)

    im = ax.pcolormesh(masked, cmap="RdBu", norm=norm, edgecolors="white", linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Example", fontsize=12)
    ax.set_yticks(np.arange(n_examples) + 0.5)
    ax.set_yticklabels([f"{i + 1}" for i in range(n_examples)], fontsize=9)
    ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Advantage: log pi(y|x,o) - log pi(y|x)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=od.DATA_PATH)
    parser.add_argument(
        "--student-model", type=str, default=od.MODEL_NAME,
        help="Student checkpoint to generate rollouts + student logp from (e.g. a trained step_N dir)",
    )
    parser.add_argument(
        "--teacher-model", type=str, default=od.MODEL_NAME,
        help="Teacher checkpoint to score (x, o, y) with (defaults to base model)",
    )
    parser.add_argument("--num-samples", type=int, default=od.NUM_SAMPLES)
    parser.add_argument("--gen-max-new-tokens", type=int, default=od.GEN_MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", type=str, default="heatmaps")
    parser.add_argument("--vmin", type=float, default=-15.0)
    parser.add_argument("--vmax", type=float, default=3.0)
    parser.add_argument(
        "--exclude-first-token", action="store_true",
        help="Drop position 0 (structural discourse-opener artifact, not real signal)",
    )
    args = parser.parse_args()

    od.GEN_MAX_NEW_TOKENS = args.gen_max_new_tokens
    os.makedirs(args.output_dir, exist_ok=True)

    same_model = args.student_model == args.teacher_model

    print(f"Loading student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=od.DEVICE,
    )
    student_model.eval()

    if same_model:
        teacher_model = student_model
    else:
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=od.DEVICE,
        )
        teacher_model.eval()

    examples = od.load_examples(args.data, args.num_samples)
    print(f"Loaded {len(examples)} examples from {args.data}")

    all_advantages = []
    for i, example in enumerate(examples):
        advantages = compute_advantages(student_model, teacher_model, tokenizer, example)
        if len(advantages) == 0:
            print(f"[sample {i}] empty completion, skipping")
            continue
        if args.exclude_first_token and len(advantages) > 1:
            advantages = advantages[1:]
        print(
            f"[sample {i}] {len(advantages)} tokens, "
            f"adv range: [{advantages.min():.2f}, {advantages.max():.2f}]"
        )
        all_advantages.append(advantages)

    student_tag = os.path.basename(os.path.normpath(args.student_model))
    teacher_tag = os.path.basename(os.path.normpath(args.teacher_model))
    title = (
        "SDFT Token-Level Advantage (teacher logp - student logp)"
        if same_model
        else f"SDFT Token-Level Advantage (student={student_tag}, teacher={teacher_tag})"
    )
    out_name = "advantage_heatmap.png" if same_model else f"advantage_heatmap_{student_tag}_vs_{teacher_tag}.png"

    plot_heatmap(
        all_advantages,
        title,
        os.path.join(args.output_dir, out_name),
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
