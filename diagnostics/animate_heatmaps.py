"""Animate SDFT token-level advantage heatmaps across a training run.

Loops over checkpoints base -> step_10 -> step_20 -> ... -> step_N, computing
A_i = teacher_logp_i - student_logp_i (teacher = fixed base model, student =
checkpoint at that step) for the same 10 examples at each step, and stitches
the per-checkpoint heatmap PNGs into a looping GIF.

Reuses compute_advantages()/plot_heatmap() from plot_heatmap.py (same dir).
Teacher model is loaded once and kept resident; student checkpoints are
loaded/freed one at a time so only two 8B models are ever resident on GPU.

Run with the existing train_dir/.venv:

    CUDA_VISIBLE_DEVICES=6 ../train_dir/.venv/bin/python animate_heatmaps.py \\
        --checkpoint-dir /mnt/nvme7n1/rawhad/amortize_maas_rag/sdft_sdg_hub_v3_run_3
"""

import argparse
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import opd_diagnostic as od
from plot_heatmap import compute_advantages, plot_heatmap

STEP_RE = re.compile(r"^step_(\d+)$")


def discover_checkpoints(checkpoint_dir: str) -> list[tuple[str, str]]:
    """Returns [(tag, path), ...] for step_N subdirs, sorted ascending by N."""
    entries = []
    for name in os.listdir(checkpoint_dir):
        m = STEP_RE.match(name)
        if m and os.path.isdir(os.path.join(checkpoint_dir, name)):
            entries.append((int(m.group(1)), name))
    entries.sort()
    return [(name, os.path.join(checkpoint_dir, name)) for _, name in entries]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--teacher-model", type=str, default=od.MODEL_NAME)
    parser.add_argument("--data", type=str, default=od.DATA_PATH)
    parser.add_argument("--num-samples", type=int, default=od.NUM_SAMPLES)
    parser.add_argument("--gen-max-new-tokens", type=int, default=od.GEN_MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", type=str, default="heatmaps/progression")
    parser.add_argument("--gif-path", type=str, default="heatmaps/progression.gif")
    parser.add_argument("--vmin", type=float, default=-15.0)
    parser.add_argument("--vmax", type=float, default=3.0)
    parser.add_argument("--duration-ms", type=int, default=200)
    parser.add_argument("--loop", type=int, default=1)
    args = parser.parse_args()

    od.GEN_MAX_NEW_TOKENS = args.gen_max_new_tokens
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = discover_checkpoints(args.checkpoint_dir)
    frames = [("base", None)] + checkpoints
    print(f"Found {len(checkpoints)} checkpoints: {[tag for tag, _ in checkpoints]}")
    print(f"Animation will have {len(frames)} frames (incl. base)")

    print(f"Loading teacher model: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=od.DEVICE,
    )
    teacher_model.eval()

    examples = od.load_examples(args.data, args.num_samples)
    print(f"Loaded {len(examples)} examples from {args.data}")

    frame_paths = []
    for i, (tag, path) in enumerate(frames):
        t0 = time.monotonic()
        if path is None:
            student_model = teacher_model
        else:
            print(f"[{i + 1}/{len(frames)}] Loading student checkpoint: {path}")
            student_model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=od.DEVICE,
            )
            student_model.eval()

        all_advantages = []
        for j, example in enumerate(examples):
            advantages = compute_advantages(student_model, teacher_model, tokenizer, example)
            if len(advantages) == 0:
                print(f"  [example {j}] empty completion, skipping")
                continue
            all_advantages.append(advantages)

        frame_path = os.path.join(args.output_dir, f"frame_{i:02d}_{tag}.png")
        plot_heatmap(
            all_advantages,
            f"SDFT Token-Level Advantage \u2014 {tag}",
            frame_path,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        frame_paths.append(frame_path)

        if path is not None:
            del student_model
            torch.cuda.empty_cache()

        print(f"[{i + 1}/{len(frames)}] {tag} done in {time.monotonic() - t0:.1f}s")

    print(f"Stitching {len(frame_paths)} frames into GIF (duration={args.duration_ms}ms, loop={args.loop})")
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    images[0].save(
        args.gif_path,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=args.loop,
    )
    print(f"Saved animation: {args.gif_path}")


if __name__ == "__main__":
    main()
