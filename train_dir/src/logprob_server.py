"""Teacher logprob server (NCCL rank 1, GPU 2).

Pure NCCL command loop — no HTTP. Receives token sequences from the trainer,
computes full log-softmax at completion positions, and broadcasts them back.
Also handles weight sync at epoch boundaries.
"""

import os

import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from src.config import MODEL_NAME, NCCL_MASTER_PORT
from src.nccl_comm import (
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    CMD_TEACHER_LOGPROBS,
    broadcast_weights,
    cleanup,
    handle_teacher_log_probs,
    init_nccl,
    recv_command,
)

DEVICE = torch.device("cuda:0")


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add("logs/logprob_server.log", level=log_level)

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)
    model.eval()
    logger.info(f"Model loaded. vocab_size={model.config.vocab_size}")

    logger.info("Initializing NCCL (rank=1)...")
    init_nccl(rank=1, world_size=2, master_port=NCCL_MASTER_PORT)
    logger.info("NCCL initialized. Entering command loop.")

    request_count = 0
    while True:
        cmd = recv_command(DEVICE)

        if cmd == CMD_TEACHER_LOGPROBS:
            handle_teacher_log_probs(model, DEVICE)
            request_count += 1
            if request_count % 50 == 0:
                logger.info(f"Served {request_count} logprob requests")

        elif cmd == CMD_SYNC_WEIGHTS:
            logger.info("Receiving weight sync from trainer...")
            broadcast_weights(model, src=0)
            logger.info("Weights synced.")

        elif cmd == CMD_SHUTDOWN:
            logger.info("Shutdown command received.")
            break

        else:
            logger.warning(f"Unknown command: {cmd}")

    cleanup()
    logger.info(f"Server exited. Total requests served: {request_count}")


if __name__ == "__main__":
    main()
