#!/usr/bin/env python3
"""Campaign analysis: sync vs async curves from parse_log CSVs + eval results."""
import csv
import json
import os
import sys


def load_steps(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_eval(path: str) -> tuple[int, int, int]:
    recs = [json.loads(l) for l in open(path)]
    n = len(recs)
    noctx = sum(1 for r in recs if r["no_context_pass"])
    wctx = sum(1 for r in recs if r["with_context_pass"])
    return noctx, wctx, n


def main() -> None:
    sync_csv, async_csv = sys.argv[1], sys.argv[2]
    eval_dir = sys.argv[3] if len(sys.argv) > 3 else None

    for tag, path in (("sync", sync_csv), ("async", async_csv)):
        rows = load_steps(path)
        n = len(rows)
        steps = [int(r["step"]) for r in rows]
        loss = [float(r["loss"]) for r in rows]
        gn = [float(r["grad_norm"]) for r in rows]
        wall = sum(float(r.get("total_s") or 0) for r in rows)
        clip = [float(r.get("is/clip_rate") or 0) for r in rows]
        lags = [float(r["lag_mean"]) for r in rows if r.get("lag_mean")]
        print(f"{tag}: steps={n} wall={wall/60:.1f} min ({wall/n:.2f}s/step)")
        print(f"  loss {loss[0]:.3f} -> {sum(loss[-10:])/10:.3f}  "
              f"grad_norm {gn[0]:.1f} -> {sum(gn[-10:])/10:.1f}  "
              f"clip_mean={sum(clip)/len(clip):.5f}")
        if lags:
            print(f"  lag mean={sum(lags)/len(lags):.1f} max={max(lags)}")

    if eval_dir:
        print("\nEval pass rates (no_context / with_context):")
        for name in sorted(os.listdir(eval_dir)):
            f = os.path.join(eval_dir, name, "eval_results.jsonl")
            if not os.path.exists(f):
                continue
            try:
                no, wc, n = load_eval(f)
            except Exception:
                print(f"  {name}: <unreadable>")
                continue
            print(f"  {name}: {no}/{n} ({100*no/n:.0f}%)  |  {wc}/{n} ({100*wc/n:.0f}%)")


if __name__ == "__main__":
    main()
