"""E2E test: launch server + trainer, get logprobs, push random weights, compare.

Weight sync coordination:
  1. Tell trainer to "perturb" weights
  2. POST /sync_weights to server in background thread (blocks on NCCL broadcast)
  3. Tell trainer to "broadcast" (both sides participate in the collective)
  4. Both complete → get logprobs again

Usage:
    GPU_TRAINER=2 GPU_SERVER=3 python test_e2e.py
"""

import io
import os
import subprocess
import sys
import threading
import time

import numpy as np
import requests
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))
SERVER_URL = f"http://localhost:{SERVER_PORT}"
GPU_SERVER = os.environ.get("GPU_SERVER", "3")
GPU_TRAINER = os.environ.get("GPU_TRAINER", "2")


def wait_for_server(timeout: int = 180) -> bool:
    """Poll health endpoint until server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def get_logprobs(token_ids: list[int]) -> np.ndarray:
    """Fetch logprobs from server. Returns (seq_len, vocab_size) float32 array."""
    r = requests.post(
        f"{SERVER_URL}/logprobs",
        json={"token_ids": token_ids},
        timeout=60,
    )
    r.raise_for_status()
    return np.load(io.BytesIO(r.content))


def _read_trainer_until(proc: subprocess.Popen, marker: str) -> bool:
    """Read trainer stdout lines until one contains `marker`. Returns False on EOF."""
    while True:
        line = proc.stdout.readline()
        if not line:
            return False
        print(f"  [trainer] {line.rstrip()}", flush=True)
        if marker in line:
            return True
        if proc.poll() is not None:
            return False


def main() -> int:
    common_env = {
        **os.environ,
        "MODEL_NAME": MODEL_NAME,
        "MASTER_PORT": str(MASTER_PORT),
        "SERVER_PORT": str(SERVER_PORT),
    }
    env_server = {**common_env, "CUDA_VISIBLE_DEVICES": GPU_SERVER}
    env_trainer = {**common_env, "CUDA_VISIBLE_DEVICES": GPU_TRAINER}

    print(f"=== E2E Test ===", flush=True)
    print(f"Model:   {MODEL_NAME}", flush=True)
    print(f"Server:  GPU {GPU_SERVER}  (port {SERVER_PORT})", flush=True)
    print(f"Trainer: GPU {GPU_TRAINER}", flush=True)
    print(f"NCCL:    localhost:{MASTER_PORT}\n", flush=True)

    # Launch both processes
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
        # Wait for trainer READY
        print("Waiting for trainer to be ready ...", flush=True)
        if not _read_trainer_until(trainer_proc, "READY"):
            print("ERROR: Trainer failed to reach READY", flush=True)
            return 1

        # Wait for server health
        print("Waiting for server health check ...", flush=True)
        if not wait_for_server():
            print("ERROR: Server failed to start within 180s", flush=True)
            return 1
        print("Server is ready!\n", flush=True)

        # Tokenize a test sentence
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        test_text = "The quick brown fox jumps over the lazy dog"
        test_tokens = tokenizer.encode(test_text)
        print(f"Test text:   '{test_text}'", flush=True)
        print(f"Token IDs:   {test_tokens}  (len={len(test_tokens)})\n", flush=True)

        # --- Step 1: get initial logprobs ---
        print("Step 1: Getting initial logprobs ...", flush=True)
        lp1 = get_logprobs(test_tokens)
        print(f"  shape: {lp1.shape}  mean: {lp1.mean():.4f}\n", flush=True)

        # --- Step 2: perturb weights (trainer side only) ---
        print("Step 2: Perturbing weights on trainer ...", flush=True)
        trainer_proc.stdin.write("perturb\n")
        trainer_proc.stdin.flush()
        if not _read_trainer_until(trainer_proc, "PERTURBED"):
            print("ERROR: Trainer failed during perturbation", flush=True)
            return 1
        print("  Weights perturbed.\n", flush=True)

        # --- Step 3: NCCL weight sync ---
        # Server must enter broadcast_weights before the trainer does.
        # We POST /sync_weights in a background thread (it blocks on NCCL),
        # then tell the trainer to broadcast.
        print("Step 3: NCCL weight sync ...", flush=True)
        sync_result: dict = {}

        def _trigger_server_sync():
            try:
                r = requests.post(f"{SERVER_URL}/sync_weights", timeout=120)
                sync_result["status"] = r.status_code
            except Exception as e:
                sync_result["error"] = str(e)

        sync_thread = threading.Thread(target=_trigger_server_sync)
        sync_thread.start()
        time.sleep(1)  # let server enter the NCCL broadcast

        # Now tell trainer to broadcast (both sides participate)
        trainer_proc.stdin.write("broadcast\n")
        trainer_proc.stdin.flush()
        if not _read_trainer_until(trainer_proc, "BROADCAST_DONE"):
            print("ERROR: Trainer failed during broadcast", flush=True)
            return 1

        sync_thread.join(timeout=30)
        print(f"  Sync result: {sync_result}", flush=True)
        if sync_result.get("status") != 200:
            print(f"ERROR: Server sync failed: {sync_result}", flush=True)
            return 1
        print("  NCCL weight sync complete!\n", flush=True)

        # --- Step 4: get updated logprobs ---
        print("Step 4: Getting updated logprobs ...", flush=True)
        lp2 = get_logprobs(test_tokens)
        print(f"  shape: {lp2.shape}  mean: {lp2.mean():.4f}\n", flush=True)

        # --- Step 5: compare ---
        diff = np.abs(lp1 - lp2)
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        print(f"=== Results ===", flush=True)
        print(f"Mean |diff|: {mean_diff:.6f}", flush=True)
        print(f"Max  |diff|: {max_diff:.6f}", flush=True)

        if mean_diff > 0.01:
            print("\nSUCCESS: Logprobs changed significantly after NCCL weight push!", flush=True)
            exit_code = 0
        else:
            print("\nFAILURE: Logprobs did not change enough (mean diff <= 0.01)", flush=True)
            exit_code = 1

    finally:
        # Cleanup
        print("\nShutting down ...", flush=True)
        if trainer_proc.poll() is None and trainer_proc.stdin:
            trainer_proc.stdin.write("shutdown\n")
            trainer_proc.stdin.flush()
            trainer_proc.wait(timeout=15)
        if server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=15)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
