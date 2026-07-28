"""Download and sample IFBench dataset for SDFT training.

Downloads allenai/IF_multi_constraints_upto5, seeded shuffle, samples 320 rows.
Output: train_sdft.jsonl in the same directory as this script.
"""

import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
NUM_SAMPLES = 320
DATASET_ID = "allenai/IF_multi_constraints_upto5"
OUTPUT_DIR = Path(__file__).parent


def main():
    print(f"Loading {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Loaded {len(dataset)} examples.")

    # Seeded shuffle + sample
    shuffled = dataset.shuffle(seed=SEED)
    subset = shuffled.select(range(NUM_SAMPLES))
    print(f"Sampled {len(subset)} examples (seed={SEED}).")

    output_path = OUTPUT_DIR / "train_sdft.jsonl"
    subset.to_json(str(output_path), orient="records", lines=True)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
