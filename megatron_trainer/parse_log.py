#!/usr/bin/env python3
"""Parse trainer.log -> per-step metrics CSV for campaign curves."""
import csv
import re
import sys


def parse(path: str) -> list[dict]:
    rows = []
    step_re = re.compile(
        r"opt_step=(\d+) loss=([\d.]+) comp_len=([\d.]+) grad_norm=([\d.]+)"
    )
    timing_re = re.compile(
        r"TIMING step=(\d+) \| total=([\d.]+)s gen=([\d.]+)s "
        r"teacher=([\d.]+)s student=([\d.]+)s loss_bwd=([\d.]+)s "
        r"optim=([\d.]+)s wsync=([\d.]+)s"
        r"(?: producer_wait=([\d.]+)s gen_overlap=([\d.]+)s"
        r"(?: lag_mean=([\d.]+) lag_max=(\d+))?)?"
    )
    metric_re = re.compile(r"([\w/.]+)=(-?[\d.e+-]+)")
    with open(path) as f:
        for line in f:
            m = step_re.search(line)
            if m:
                row = {
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "comp_len": float(m.group(3)),
                    "grad_norm": float(m.group(4)),
                }
                for k, v in metric_re.findall(line[m.end():]):
                    if k not in ("loss", "comp_len", "grad_norm"):
                        row[k] = float(v)
                rows.append(row)
            m = timing_re.search(line)
            if m and rows:
                last = rows[-1]
                last["total_s"] = float(m.group(2))
                last["gen_s"] = float(m.group(3))
                last["teacher_s"] = float(m.group(4))
                last["student_s"] = float(m.group(5))
                last["loss_bwd_s"] = float(m.group(6))
                last["optim_s"] = float(m.group(7))
                last["wsync_s"] = float(m.group(8))
                if m.group(9) is not None:
                    last["producer_wait_s"] = float(m.group(9))
                    last["gen_overlap_s"] = float(m.group(10))
                if m.group(11) is not None:
                    last["lag_mean"] = float(m.group(11))
                    last["lag_max"] = int(m.group(12))
    return rows


def main() -> None:
    for path in sys.argv[1:]:
        rows = parse(path)
        if not rows:
            print(f"{path}: no steps found")
            continue
        keys = list(rows[0].keys())
        out = path.replace(".log", "_steps.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        n = len(rows)
        print(f"{path}: {n} steps -> {out}")
        if "gen_overlap_s" in keys:
            overlap = [r["gen_overlap_s"] for r in rows if r.get("gen_overlap_s", 0) > 0]
            waits = [r["producer_wait_s"] for r in rows]
            lags = [r["lag_mean"] for r in rows if r.get("lag_mean")]
            print(f"  gen_overlap mean={sum(overlap)/max(len(overlap),1):.1f}s "
                  f"producer_wait mean={sum(waits)/max(len(waits),1):.2f}s "
                  f"lag mean={sum(lags)/max(len(lags),1):.1f} max={max(lags) if lags else '-'}")
            print(f"  total mean={sum(r['total_s'] for r in rows)/n:.1f}s")


if __name__ == "__main__":
    main()
