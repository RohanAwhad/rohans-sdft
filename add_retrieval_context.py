"""Preprocess a dataset JSONL to inject retrieved context into user messages.

Usage:
    python add_retrieval_context.py <input.jsonl> <output.jsonl> [--retriever-url URL] [--k K]

For each row, extracts the last human message, calls the retriever, and prepends
the retrieved context into that message's value.
"""

import argparse
import json
import sys

import requests


def retrieve(question: str, url: str, k: int) -> str:
    resp = requests.post(f"{url}/retrieve", json={"question": question, "k": k})
    resp.raise_for_status()
    return resp.json()["context"]


def inject_context(row: dict, retriever_url: str, k: int) -> dict:
    last_human_idx = None
    for i, msg in enumerate(row["prompt"]):
        if msg["from"] == "human":
            last_human_idx = i

    if last_human_idx is None:
        return row

    question = row["prompt"][last_human_idx]["value"]
    context = retrieve(question, retriever_url, k)

    row["prompt"][last_human_idx]["value"] = (
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description="Add retrieval context to dataset")
    parser.add_argument("input", help="Input JSONL path")
    parser.add_argument("output", help="Output JSONL path")
    parser.add_argument("--retriever-url", default="http://localhost:9090")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    total = len(rows)
    for i, row in enumerate(rows):
        inject_context(row, args.retriever_url, args.k)
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"  {i + 1}/{total}", file=sys.stderr)

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done: {total} rows written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
