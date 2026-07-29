#!/usr/bin/env python
"""Check A: NCCL fidelity for teacher logprob transfer.

Sends N random sequences (4096 tokens each) through the real NCCL path
and compares against direct single-process computation on the trainer GPU.

Architecture:
  - Server subprocess (GPU_SERVER): loads model, handles NCCL logprob requests
  - Trainer subprocess (GPU_TRAINER): receives NCCL logprobs, then loads model
    locally and computes direct logprobs, compares in-memory. No files needed.
  - Orchestrator: spawns both, waits for exit codes.

Usage (from repo root):
    MODEL_NAME=Qwen/Qwen3-8B python diagnostics/verify_teacher_logprobs.py
"""

import argparse
import os
import random
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEFAULT_MASTER_PORT = 29502

N_SEQUENCES = 10
SEQ_LEN = 4096
PROMPT_LEN = 128


# ---------------------------------------------------------------------------
# Server (rank 1)
# ---------------------------------------------------------------------------


def _run_server(master_port: int, n: int) -> None:
    sys.path.insert(0, os.path.join(REPO_ROOT, "train_dir"))

    import torch
    from transformers import AutoModelForCausalLM

    from src.nccl_comm import (
        CMD_SHUTDOWN,
        CMD_TEACHER_LOGPROBS,
        cleanup,
        handle_teacher_log_probs,
        init_nccl,
        recv_command,
    )

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device,
    )
    model.eval()
    print(f"[server] Model loaded: {MODEL_NAME}, vocab={model.config.vocab_size}", flush=True)

    init_nccl(rank=1, world_size=2, master_port=master_port)
    print("[server] NCCL initialized", flush=True)

    for i in range(n):
        cmd = recv_command(device)
        assert cmd == CMD_TEACHER_LOGPROBS, f"Expected logprob cmd, got {cmd}"
        handle_teacher_log_probs(model, device)
        print(f"[server] Handled seq {i}", flush=True)

    cmd = recv_command(device)
    assert cmd == CMD_SHUTDOWN, f"Expected shutdown, got {cmd}"
    cleanup()
    print("[server] Done", flush=True)


# ---------------------------------------------------------------------------
# Trainer (rank 0) — does NCCL receive + direct compute + comparison
# ---------------------------------------------------------------------------


def _run_trainer(master_port: int, n: int) -> int:
    sys.path.insert(0, os.path.join(REPO_ROOT, "train_dir"))

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.nccl_comm import (
        CMD_SHUTDOWN,
        cleanup,
        init_nccl,
        request_teacher_log_probs,
        send_command,
    )

    device = torch.device("cuda:0")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    vocab_size = config.vocab_size
    print(f"[trainer] vocab_size={vocab_size}", flush=True)

    # === Phase 1: NCCL — receive logprobs, store on CPU ===
    init_nccl(rank=0, world_size=2, master_port=master_port)
    print("[trainer] NCCL initialized", flush=True)

    random.seed(42)
    all_token_ids: list[list[int]] = []
    nccl_results: list[torch.Tensor] = []

    for i in range(n):
        token_ids = [random.randint(0, vocab_size - 1) for _ in range(SEQ_LEN)]
        all_token_ids.append(token_ids)

        t0 = time.monotonic()
        log_probs = request_teacher_log_probs(
            token_ids=token_ids,
            prompt_len=PROMPT_LEN,
            vocab_size=vocab_size,
            device=device,
        )
        elapsed = time.monotonic() - t0

        nccl_results.append(log_probs.cpu())
        print(
            f"[trainer] NCCL seq {i}: shape={log_probs.shape}, nccl_time={elapsed:.4f}s",
            flush=True,
        )

    send_command(CMD_SHUTDOWN, device)
    cleanup()
    print("[trainer] NCCL phase done, loading model for direct computation...", flush=True)

    # === Phase 2: Direct computation — same GPU, no NCCL ===
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device,
    )
    model.eval()
    print(f"[trainer] Model loaded for direct computation", flush=True)

    direct_results: list[torch.Tensor] = []
    for i in range(n):
        token_ids = all_token_ids[i]
        completion_len = SEQ_LEN - PROMPT_LEN

        ids = torch.tensor(token_ids, device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(ids.unsqueeze(0)).logits[0]

        completion_logits = logits[PROMPT_LEN - 1 : PROMPT_LEN + completion_len - 1, :]
        log_probs = F.log_softmax(completion_logits.float(), dim=-1).bfloat16()

        direct_results.append(log_probs.cpu())
        print(f"[trainer] Direct seq {i}: shape={log_probs.shape}", flush=True)

    del model
    torch.cuda.empty_cache()

    # === Phase 3: Compare ===
    print(f"\n{'='*80}")
    print("Comparison: NCCL vs Direct (in-memory, same trainer GPU)")
    print(f"{'='*80}")
    print(f"{'#':<3} {'Shape':<22} {'MaxDiff':<12} {'MeanDiff':<14} {'Match'}")
    print("-" * 65)

    all_pass = True
    for i in range(n):
        nccl_lp = nccl_results[i].float()
        direct_lp = direct_results[i].float()

        assert nccl_lp.shape == direct_lp.shape, (
            f"Shape mismatch seq {i}: nccl={nccl_lp.shape} direct={direct_lp.shape}"
        )

        diff = (nccl_lp - direct_lp).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match = max_diff < 1e-3

        if not match:
            all_pass = False

        shape_str = f"({nccl_lp.size(0)}, {nccl_lp.size(1)})"
        status = "PASS" if match else "FAIL"
        print(f"{i:<3} {shape_str:<22} {max_diff:<12.6f} {mean_diff:<14.10f} {status}")

    print()
    if all_pass:
        print("RESULT: ALL PASS -- NCCL transfer is faithful")
    else:
        print("RESULT: FAIL -- NCCL transfer has discrepancies")

    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def orchestrate(gpu_server: int, gpu_trainer: int, master_port: int) -> int:
    n = N_SEQUENCES
    print(f"Model:       {MODEL_NAME}")
    print(f"GPU server:  {gpu_server}")
    print(f"GPU trainer: {gpu_trainer}")
    print(f"NCCL port:   {master_port}")
    print(f"Sequences:   {n} x {SEQ_LEN} tokens (prompt={PROMPT_LEN}, completion={SEQ_LEN - PROMPT_LEN})")
    print()

    common_args = [
        sys.executable, os.path.abspath(__file__),
        "--master-port", str(master_port),
        "--n", str(n),
    ]

    server_proc = subprocess.Popen(
        common_args + ["--role", "server"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_server)},
    )
    trainer_proc = subprocess.Popen(
        common_args + ["--role", "trainer"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_trainer)},
    )

    trainer_rc = trainer_proc.wait()
    server_rc = server_proc.wait()

    if trainer_rc != 0 or server_rc != 0:
        print(f"ERROR: subprocess failed (trainer={trainer_rc}, server={server_rc})")
        return 1
    return trainer_rc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify teacher logprob NCCL fidelity")
    parser.add_argument("--role", choices=["server", "trainer"], default=None)
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    parser.add_argument("--n", type=int, default=N_SEQUENCES)
    parser.add_argument("--gpu-server", type=int, default=6)
    parser.add_argument("--gpu-trainer", type=int, default=7)
    args = parser.parse_args()

    if args.role is None:
        sys.exit(orchestrate(args.gpu_server, args.gpu_trainer, args.master_port))
    elif args.role == "server":
        _run_server(args.master_port, args.n)
    elif args.role == "trainer":
        sys.exit(_run_trainer(args.master_port, args.n))
