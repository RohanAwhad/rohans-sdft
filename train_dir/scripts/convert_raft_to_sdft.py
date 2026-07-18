"""Convert RAFT dataset (instruction + answer) to SDFT format (prompt + user_response)."""

import json
import sys
from pathlib import Path


def convert(input_path: str, output_path: str) -> None:
    rows = []
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append({
                "prompt": [{"role": "user", "content": ex["instruction"]}],
                "user_response": {"content": ex["answer"]},
            })

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"{input_path} -> {output_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data" / "raft"
    out_dir = data_dir

    convert(str(data_dir / "train.jsonl"), str(out_dir / "train_sdft.jsonl"))
    convert(str(data_dir / "eval.jsonl"), str(out_dir / "eval_sdft.jsonl"))
