"""E2E test: pure NCCL logprob server + trainer.

Launches server (rank 1) and trainer (rank 0), then drives the trainer
via stdin commands to get logprobs, perturb weights, sync, and compare.

Usage:
    GPU_TRAINER=2 GPU_SERVER=3 python test_e2e.py
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))
GPU_SERVER = os.environ.get("GPU_SERVER", "3")
GPU_TRAINER = os.environ.get("GPU_TRAINER", "2")


def _read_until(proc: subprocess.Popen, marker: str) -> tuple[bool, str]:
    """Read trainer stdout until a line contains marker."""
    while True:
        line = proc.stdout.readline()
        if not line:
            return False, ""
        print(f"  [trainer] {line.rstrip()}", flush=True)
        if marker in line:
            return True, line
        if proc.poll() is not None:
            return False, ""


def _send(proc: subprocess.Popen, cmd: str) -> None:
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def main() -> int:
    common_env = {
        **os.environ,
        "MODEL_NAME": MODEL_NAME,
        "MASTER_PORT": str(MASTER_PORT),
    }
    env_server = {**common_env, "CUDA_VISIBLE_DEVICES": GPU_SERVER}
    env_trainer = {**common_env, "CUDA_VISIBLE_DEVICES": GPU_TRAINER}

    print(f"=== E2E Test (pure NCCL) ===", flush=True)
    print(f"Model:   {MODEL_NAME}", flush=True)
    print(f"Server:  GPU {GPU_SERVER}", flush=True)
    print(f"Trainer: GPU {GPU_TRAINER}", flush=True)
    print(f"NCCL:    localhost:{MASTER_PORT}\n", flush=True)

    server_proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "src", "server.py")],
        env=env_server,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    trainer_proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "src", "trainer.py")],
        env=env_trainer,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )

    exit_code = 1
    try:
        print("Waiting for trainer READY ...", flush=True)
        if not _read_until(trainer_proc, "READY")[0]:
            print("ERROR: Trainer never reached READY", flush=True)
            return 1

        # Step 1: get initial logprobs via NCCL
        print("\nStep 1: Get initial logprobs via NCCL", flush=True)
        _send(trainer_proc, "logprobs v1")
        if not _read_until(trainer_proc, "LOGPROBS v1")[0]:
            print("ERROR: Failed to get v1 logprobs", flush=True)
            return 1

        # Step 2: perturb weights on trainer
        print("\nStep 2: Perturb trainer weights", flush=True)
        _send(trainer_proc, "perturb")
        if not _read_until(trainer_proc, "PERTURBED")[0]:
            print("ERROR: Failed to perturb", flush=True)
            return 1

        # Step 3: sync weights to server via NCCL
        print("\nStep 3: Sync weights to server via NCCL", flush=True)
        _send(trainer_proc, "sync_weights")
        if not _read_until(trainer_proc, "SYNCED")[0]:
            print("ERROR: Failed to sync weights", flush=True)
            return 1

        # Step 4: get updated logprobs via NCCL
        print("\nStep 4: Get updated logprobs via NCCL", flush=True)
        _send(trainer_proc, "logprobs v2")
        if not _read_until(trainer_proc, "LOGPROBS v2")[0]:
            print("ERROR: Failed to get v2 logprobs", flush=True)
            return 1

        # Step 5: compare
        print("\nStep 5: Compare", flush=True)
        _send(trainer_proc, "compare v1 v2")
        found, line = _read_until(trainer_proc, "DIFF")
        if not found:
            print("ERROR: Failed to compare", flush=True)
            return 1

        # Parse: DIFF mean=X max=Y
        parts = line.strip().split()
        mean_diff = float([p for p in parts if p.startswith("mean=")][0].split("=")[1])
        max_diff = float([p for p in parts if p.startswith("max=")][0].split("=")[1])

        print(f"\n=== Results ===", flush=True)
        print(f"Mean |diff|: {mean_diff:.6f}", flush=True)
        print(f"Max  |diff|: {max_diff:.6f}", flush=True)

        if mean_diff > 0.01:
            print("\nSUCCESS: Logprobs changed significantly after NCCL weight push!", flush=True)
            exit_code = 0
        else:
            print("\nFAILURE: Logprobs did not change enough", flush=True)
            exit_code = 1

    finally:
        print("\nShutting down ...", flush=True)
        if trainer_proc.poll() is None:
            _send(trainer_proc, "shutdown")
            time.sleep(2)
        for proc in (trainer_proc, server_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (trainer_proc, server_proc):
            if proc.poll() is None:
                proc.wait(timeout=10)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
