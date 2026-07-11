"""Logprob server (NCCL rank 1) — pure NCCL, no HTTP.

Blocking command loop:
  CMD_LOGPROBS     → receive token_ids, compute logprobs, send back
  CMD_SYNC_WEIGHTS → receive updated weights from rank 0
  CMD_SHUTDOWN     → exit
"""

import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from nccl_comm import (
    CMD_LOGPROBS,
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    broadcast_weights,
    cleanup,
    init_nccl,
    recv_command,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda:0")
NCCL_RANK = 1
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))


def main() -> None:
    print(f"[server] Loading model {MODEL_NAME} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"[server] Model loaded. vocab_size={vocab_size}", flush=True)

    print(f"[server] Initializing NCCL (rank={NCCL_RANK}) ...", flush=True)
    init_nccl(rank=NCCL_RANK, master_port=MASTER_PORT)
    print("[server] NCCL initialized. Entering command loop.", flush=True)

    while True:
        cmd = recv_command(DEVICE)

        if cmd == CMD_LOGPROBS:
            # 1. Receive seq_len
            seq_len_t = torch.zeros(1, dtype=torch.long, device=DEVICE)
            dist.broadcast(seq_len_t, src=0)
            seq_len = int(seq_len_t.item())

            # 2. Receive token_ids
            token_ids = torch.zeros(seq_len, dtype=torch.long, device=DEVICE)
            dist.broadcast(token_ids, src=0)

            # 3. Compute logprobs
            with torch.no_grad():
                logits = model(token_ids.unsqueeze(0)).logits[0]  # (seq_len, vocab_size)
                logprobs = torch.log_softmax(logits, dim=-1).float()

            # 4. Send logprobs back (broadcast from rank 1)
            dist.broadcast(logprobs, src=1)
            print(f"[server] logprobs sent: shape={tuple(logprobs.shape)}", flush=True)

        elif cmd == CMD_SYNC_WEIGHTS:
            broadcast_weights(model, src=0)
            print("[server] weights synced.", flush=True)

        elif cmd == CMD_SHUTDOWN:
            print("[server] shutdown.", flush=True)
            break

    cleanup()


if __name__ == "__main__":
    main()
