"""E2E test: launch server + trainer, get logprobs, push random weights, compare.

Usage (from task_2/):
    CUDA_VISIBLE_DEVICES=2,3 python test_e2e.py

Or with explicit GPU assignment:
    GPU_TRAINER=2 GPU_SERVER=3 python test_e2e.py
"""

import io
import os
import subprocess
import sys
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


def main() -> int:
    common_env = {
        **os.environ,
        "MODEL_NAME": MODEL_NAME,
        "MASTER_PORT": str(MASTER_PORT),
        "SERVER_PORT": str(SERVER_PORT),
    }
    env_server = {**common_env, "CUDA_VISIBLE_DEVICES": GPU_SERVER}
    env_trainer = {**common_env, "CUDA_VISIBLE_DEVICES": GPU_TRAINER}

    print(f"=== E2E Test ===")
    print(f"Model:   {MODEL_NAME}")
    print(f"Server:  GPU {GPU_SERVER}  (port {SERVER_PORT})")
    print(f"Trainer: GPU {GPU_TRAINER}")
    print(f"NCCL:    localhost:{MASTER_PORT}")
    print()

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
        print("Waiting for trainer to be ready ...")
        while True:
            line = trainer_proc.stdout.readline()
            if not line:
                print("ERROR: Trainer stdout closed unexpectedly")
                return 1
            print(f"  [trainer] {line.rstrip()}")
            if "READY" in line:
                break
            if trainer_proc.poll() is not None:
                print("ERROR: Trainer exited prematurely")
                return 1

        # Wait for server health
        print("Waiting for server health check ...")
        if not wait_for_server():
            print("ERROR: Server failed to start within 180s")
            return 1
        print("Server is ready!\n")

        # Tokenize a test sentence
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        test_text = "The quick brown fox jumps over the lazy dog"
        test_tokens = tokenizer.encode(test_text)
        print(f"Test text:   '{test_text}'")
        print(f"Token IDs:   {test_tokens}  (len={len(test_tokens)})\n")

        # --- Step 1: get initial logprobs ---
        print("Step 1: Getting initial logprobs ...")
        lp1 = get_logprobs(test_tokens)
        print(f"  shape: {lp1.shape}  mean: {lp1.mean():.4f}\n")

        # --- Step 2: perturb + push weights via NCCL ---
        print("Step 2: Perturbing weights and pushing via NCCL ...")
        trainer_proc.stdin.write("perturb_and_push\n")
        trainer_proc.stdin.flush()

        while True:
            line = trainer_proc.stdout.readline()
            if not line:
                print("ERROR: Trainer stdout closed during push")
                return 1
            print(f"  [trainer] {line.rstrip()}")
            if "WEIGHTS_PUSHED" in line:
                break
            if trainer_proc.poll() is not None:
                print("ERROR: Trainer exited during weight push")
                return 1
        print("  Weights pushed successfully!")
        time.sleep(1)  # let server settle
        print()

        # --- Step 3: get updated logprobs ---
        print("Step 3: Getting updated logprobs ...")
        lp2 = get_logprobs(test_tokens)
        print(f"  shape: {lp2.shape}  mean: {lp2.mean():.4f}\n")

        # --- Step 4: compare ---
        diff = np.abs(lp1 - lp2)
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        print(f"=== Results ===")
        print(f"Mean |diff|: {mean_diff:.6f}")
        print(f"Max  |diff|: {max_diff:.6f}")

        if mean_diff > 0.01:
            print("\nSUCCESS: Logprobs changed significantly after NCCL weight push!")
            exit_code = 0
        else:
            print("\nFAILURE: Logprobs did not change enough (mean diff <= 0.01)")
            exit_code = 1

    finally:
        # Cleanup
        print("\nShutting down ...")
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
