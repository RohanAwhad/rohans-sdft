"""Teacher logprob server (NCCL rank 1) — Megatron Bridge version.

Pure NCCL command loop. Receives token sequences from the trainer,
computes full log-softmax at completion positions, broadcasts back.
Also handles EMA weight sync at step boundaries.
"""

import os

import torch
from loguru import logger

from megatron_trainer.config import EMA_ALPHA, HF_MODEL_PATH, NCCL_MASTER_PORT
from megatron_trainer.model_utils import cleanup, init_distributed, load_model
from megatron_trainer.nccl_comm import (
    CMD_SHUTDOWN,
    CMD_SYNC_WEIGHTS,
    CMD_TEACHER_LOGPROBS,
    broadcast_weights_ema,
    handle_teacher_log_probs,
    recv_command,
)

DEVICE = torch.device("cuda:0")


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add("logs/logprob_server.log", level=log_level)

    # ---- Initialize Megatron + torch.distributed ----
    init_distributed(rank=1, world_size=2, master_port=NCCL_MASTER_PORT)
    logger.info("Distributed init complete (logprob_server=rank1).")

    # ---- Load model ----
    logger.info(f"Loading model: {HF_MODEL_PATH}")
    model = load_model(HF_MODEL_PATH)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # ---- Command loop ----
    logger.info("Entering command loop.")
    request_count = 0
    while True:
        cmd = recv_command(DEVICE)

        if cmd == CMD_TEACHER_LOGPROBS:
            handle_teacher_log_probs(model, DEVICE)
            request_count += 1
            if request_count % 50 == 0:
                logger.info(f"Served {request_count} logprob requests")

        elif cmd == CMD_SYNC_WEIGHTS:
            logger.info(f"Receiving weight sync (EMA alpha={EMA_ALPHA})...")
            broadcast_weights_ema(model, alpha=EMA_ALPHA, src=0)
            logger.info("Weights EMA-blended.")

        elif cmd == CMD_SHUTDOWN:
            logger.info("Shutdown command received.")
            break

        else:
            logger.warning(f"Unknown command: {cmd}")

    cleanup()
    logger.info(f"Server exited. Total requests served: {request_count}")


if __name__ == "__main__":
    main()
