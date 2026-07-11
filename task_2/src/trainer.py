"""Training / weight-pusher process (NCCL rank 0).

Stdin commands:
  logprobs <name>    — request logprobs from server via NCCL, store as <name>
  perturb            — randomly perturb all weights
  sync_weights       — push weights to server via NCCL broadcast
  compare <n1> <n2>  — compare two stored logprob snapshots
  shutdown           — signal server to exit, then exit
"""

import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from nccl_comm import (
    CMD_LOGPROBS,
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    broadcast_weights,
    cleanup,
    init_nccl,
    send_command,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda:0")
NCCL_RANK = 0
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))
TEST_TEXT = "The quick brown fox jumps over the lazy dog"


def request_logprobs(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Send token IDs to server via NCCL, receive logprobs back.

    Returns (seq_len, vocab_size) float32 tensor on DEVICE.
    """
    seq_len = token_ids.shape[0]

    # 1. Send command
    send_command(CMD_LOGPROBS, DEVICE)

    # 2. Send seq_len
    dist.broadcast(torch.tensor([seq_len], dtype=torch.long, device=DEVICE), src=0)

    # 3. Send token_ids
    dist.broadcast(token_ids, src=0)

    # 4. Receive logprobs (broadcast from rank 1)
    logprobs = torch.zeros(seq_len, vocab_size, dtype=torch.float32, device=DEVICE)
    dist.broadcast(logprobs, src=1)

    return logprobs


def main() -> None:
    print(f"[trainer] Loading model {MODEL_NAME} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    vocab_size = model.config.vocab_size
    print(f"[trainer] Model loaded. vocab_size={vocab_size}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_tokens = tokenizer.encode(TEST_TEXT)
    token_ids = torch.tensor(test_tokens, dtype=torch.long, device=DEVICE)
    print(f"[trainer] Test: '{TEST_TEXT}' → {test_tokens} (len={len(test_tokens)})", flush=True)

    print(f"[trainer] Initializing NCCL (rank={NCCL_RANK}) ...", flush=True)
    init_nccl(rank=NCCL_RANK, master_port=MASTER_PORT)
    print("[trainer] NCCL initialized.", flush=True)

    print("READY", flush=True)

    stored: dict[str, torch.Tensor] = {}

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]

        if cmd == "logprobs":
            name = parts[1] if len(parts) > 1 else "latest"
            lp = request_logprobs(token_ids, vocab_size)
            stored[name] = lp
            print(f"LOGPROBS {name} shape={tuple(lp.shape)} mean={lp.mean():.4f}", flush=True)

        elif cmd == "perturb":
            with torch.no_grad():
                for param in model.parameters():
                    param.data += torch.randn_like(param.data) * 0.5
            print("PERTURBED", flush=True)

        elif cmd == "sync_weights":
            send_command(CMD_SYNC_WEIGHTS, DEVICE)
            broadcast_weights(model, src=0)
            print("SYNCED", flush=True)

        elif cmd == "compare":
            n1, n2 = parts[1], parts[2]
            diff = (stored[n1] - stored[n2]).abs()
            print(f"DIFF mean={diff.mean():.6f} max={diff.max():.6f}", flush=True)

        elif cmd == "shutdown":
            send_command(CMD_SHUTDOWN, DEVICE)
            break

    cleanup()


if __name__ == "__main__":
    main()
