"""Convert OpenAI messages format to SDFT format (prompt + user_response)."""

import json
import sys


def convert(input_path: str, output_path: str) -> None:
    rows = []
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]

            answer = next(
                m["content"] for m in reversed(msgs) if m["role"] == "assistant"
            )

            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]

            rows.append({
                "prompt": prompt_msgs,
                "user_response": {"content": answer},
            })

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"{input_path} -> {output_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".jsonl", "_sdft.jsonl")
    convert(input_path, output_path)
