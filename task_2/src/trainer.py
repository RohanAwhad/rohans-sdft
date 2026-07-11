"""Training / weight-pusher process (NCCL rank 0).

Loads the same HF model, accepts commands on stdin to perturb weights
and broadcast them to the inference server (rank 1) via NCCL.
"""

import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from nccl_comm import init_nccl, broadcast_weights, cleanup

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda:0")
NCCL_RANK = 0
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))


def push_weights(model: torch.nn.Module) -> None:
    """Signal rank 1 that weights are coming, then broadcast all params."""
    signal = torch.ones(1, device=DEVICE)
    dist.broadcast(signal, src=0)
    broadcast_weights(model, src=0)


def shutdown_receiver() -> None:
    """Send shutdown signal (negative) to rank 1's receiver loop."""
    signal = torch.tensor([-1.0], device=DEVICE)
    dist.broadcast(signal, src=0)


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

        if cmd == "perturb_and_push":
            # Randomly perturb all weights (large noise so logprobs change obviously)
            with torch.no_grad():
                for param in model.parameters():
                    param.data += torch.randn_like(param.data) * 0.5
            push_weights(model)
            print("WEIGHTS_PUSHED", flush=True)

        elif cmd == "shutdown":
            shutdown_receiver()
            break

    cleanup()


if __name__ == "__main__":
    main()
