"""Layer 1 plumbing-equivalence verification (see docs/megatron_trainer/async_rollouts.md).

Run on the cluster after the sync and async-in-order verification runs (both
with DEBUG_ROLLOUT_HASH=1 + ROLLOUT_REPLAY_PATH):

    python3 megatron_trainer/verify_layer1.py logs/smoke_replay_sync.log logs/smoke_replay_async.log

Checks:
  1. Within-run assignment: CONSUME_HASH (step, rank, micro) equals the
     produced ROLLOUT_HASH (batch, idx) under each mode's mapping
     (sync: rank r micro m <- sample r*L+m; async in-order: column-major
     queue — same map). Every microbatch of the run is checked.
  2. Cross-mode: produced hash streams identical (replay guarantees the
     rollout data is byte-identical; the check confirms the producer pulls
     the same batches in the same order).
"""
import re
import sys


def parse(fn: str):
    produced: dict[int, dict[int, str]] = {}
    consumed: list[tuple[int, int, int, str]] = []
    with open(fn) as f:
        for line in f:
            m = re.search(r"ROLLOUT_HASH batch=(\d+) idx=(\d+) hash=(\w+)", line)
            if m:
                produced.setdefault(int(m.group(1)), {})[int(m.group(2))] = m.group(3)
            m = re.search(r"CONSUME_HASH step=(\d+) rank=(\d+) micro=(\d+) hash=(\w+)", line)
            if m:
                consumed.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)))
    return produced, consumed


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: verify_layer1.py <sync_log> <async_in_order_log>")
        sys.exit(2)
    sync_log, async_log = sys.argv[1], sys.argv[2]

    ok = True
    for tag, fn in (("sync", sync_log), ("async-in-order", async_log)):
        produced, consumed = parse(fn)
        ranks = {r for (_, r, _, _) in consumed}
        world_size = max(ranks) + 1
        per_step: dict[int, dict[tuple[int, int], str]] = {}
        for (s, r, m, h) in consumed:
            per_step.setdefault(s, {})[(r, m)] = h
        for s in sorted(per_step):
            local_accum = len(produced[s]) // world_size
            for (r, m), got in per_step[s].items():
                expect = produced[s][r * local_accum + m]
                if got != expect:
                    ok = False
                    print(f"{tag}: MISMATCH step={s} rank={r} micro={m}")
        print(f"{tag}: within-run assignment {'PASS' if ok else 'FAIL'}")

    p_s, _ = parse(sync_log)
    p_a, _ = parse(async_log)
    same = p_s == p_a
    ok = ok and same
    print(f"cross-mode produced streams: {'identical' if same else 'DIFFER'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
