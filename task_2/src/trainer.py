"""Training / weight-pusher process (NCCL rank 0).

Loads the same HF model, accepts commands on stdin:
  - "perturb"   → randomly perturb all weights
  - "broadcast" → broadcast weights to server via NCCL
  - "shutdown"  → exit
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM

from nccl_comm import init_nccl, broadcast_weights, cleanup

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda:0")
NCCL_RANK = 0
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))


def main() -> None:
    print(f"[trainer] Loading model {MODEL_NAME} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    print("[trainer] Model loaded.", flush=True)

    print(f"[trainer] Initializing NCCL (rank={NCCL_RANK}) ...", flush=True)
    init_nccl(rank=NCCL_RANK, master_port=MASTER_PORT)
    print("[trainer] NCCL initialized.", flush=True)

    print("READY", flush=True)

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        if cmd == "perturb":
            with torch.no_grad():
                for param in model.parameters():
                    param.data += torch.randn_like(param.data) * 0.5
            print("PERTURBED", flush=True)

        elif cmd == "broadcast":
            broadcast_weights(model, src=0)
            print("BROADCAST_DONE", flush=True)

        elif cmd == "shutdown":
            break

    cleanup()


if __name__ == "__main__":
    main()
