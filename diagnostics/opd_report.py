"""Aggregate OPD diagnostic report across sample JSONs produced by opd_diagnostic.py.

Usage:
    python opd_report.py [opd_diagnostic]
"""

import glob
import json
import math
import sys

DIAG_DIR = sys.argv[1] if len(sys.argv) > 1 else "opd_diagnostic"
TOP_DIVERGENT = 10


def log_ratio(t: dict) -> float:
    sp = max(t["student_prob"], 1e-10)
    tp = max(t["teacher_prob"], 1e-10)
    return math.log(tp) - math.log(sp)


def main():
    paths = sorted(glob.glob(f"{DIAG_DIR}/*.json"))
    if not paths:
        print(f"No JSON files found in {DIAG_DIR}/")
        return

    print(f"{'sample':<20}{'len':>5}{'overlap':>9}{'H_stu':>8}{'H_tea':>8}{'H_gap':>8}{'P_stu':>8}{'P_tea':>8}")
    print("-" * 74)

    all_tokens = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        s = d["summary"]
        name = path.split("/")[-1].replace(".json", "")
        print(
            f"{name:<20}{d['completion_length']:>5}{s['mean_overlap_ratio']:>9.3f}"
            f"{s['mean_student_entropy']:>8.3f}{s['mean_teacher_entropy']:>8.3f}"
            f"{s['mean_entropy_gap']:>8.3f}{s['mean_student_prob']:>8.3f}{s['mean_teacher_prob']:>8.3f}"
        )

        for t in d["tokens"]:
            t["_log_ratio"] = log_ratio(t)
            t["_sample"] = name
        all_tokens.extend(d["tokens"])

    print()
    print(f"Top {TOP_DIVERGENT} highest-divergence token positions (|log(teacher_p/student_p)|):")
    top = sorted(all_tokens, key=lambda t: -abs(t["_log_ratio"]))[:TOP_DIVERGENT]
    for t in top:
        tok = t["token_str"].replace("\n", "\\n")
        print(
            f"  [{t['_sample']:<16} pos={t['position']:>3}] {tok!r:<20} "
            f"student_p={t['student_prob']:.4f} teacher_p={t['teacher_prob']:.4f} "
            f"log_ratio={t['_log_ratio']:+.3f} overlap={t['overlap_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
