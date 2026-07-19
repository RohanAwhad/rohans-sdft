"""Teacher logprob server — HTTP (FastAPI) version.

Standalone process with its own torch.distributed world_size=1 (for Megatron
model loading only). Serves teacher log-probs via HTTP and accepts weight
sync via PyNcclCommunicator (initialized by trainer via HTTP handshake).

Endpoints:
    GET  /health          — readiness probe
    POST /logprobs        — compute log-probs, return binary (float16)
    POST /init_weight_sync — NCCL communicator init handshake
    POST /sync_weights     — receive weights via NCCL + EMA blend
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel

from megatron_trainer.config import EMA_ALPHA, HF_MODEL_PATH, LOGPROB_PORT
from megatron_trainer.model_utils import init_distributed_standalone, load_model

DEVICE = torch.device("cuda:0")


class LogprobRequest(BaseModel):
    token_ids: list[int]
    prompt_len: int


class NCCLInitRequest(BaseModel):
    master_address: str
    master_port: int
    world_size: int


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add("logs/logprob_server.log", level=log_level)

    logger.info("=== Logprob Server (HTTP) Starting ===")

    # ---- Standalone torch.distributed for Megatron model loading ----
    init_distributed_standalone()
    logger.info("Distributed init complete (standalone, world_size=1).")

    # ---- Load model ----
    logger.info(f"Loading model: {HF_MODEL_PATH}")
    model = load_model(HF_MODEL_PATH)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # ---- PyNcclCommunicator — initialized later via HTTP handshake ----
    logprob_nccl_comm = None
    request_count = 0

    # ---- FastAPI app ----
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/logprobs")
    def compute_logprobs(request: LogprobRequest):
        nonlocal request_count

        seq_len = len(request.token_ids)
        completion_len = seq_len - request.prompt_len
        token_ids = torch.tensor(request.token_ids, device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            input_ids = token_ids.unsqueeze(0)
            position_ids = torch.arange(seq_len, device=DEVICE, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits = logits[0]  # (S, V)

            comp_logits = logits[request.prompt_len - 1 : request.prompt_len + completion_len - 1]
            log_probs = F.log_softmax(comp_logits.float(), dim=-1)

        # Binary response: float16 numpy bytes — ~145 MB for C=500, V=151936
        response_bytes = log_probs.cpu().to(torch.float16).numpy().tobytes()

        request_count += 1
        if request_count % 50 == 0:
            logger.info(f"Served {request_count} logprob requests")

        return Response(content=response_bytes, media_type="application/octet-stream")

    @app.post("/init_weight_sync")
    def init_weight_sync(request: NCCLInitRequest):
        """HTTP handshake: trainer rank 0 sends NCCL init info, we create our communicator."""
        nonlocal logprob_nccl_comm

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import _stateless_init_process_group

        logger.info(
            f"Initializing NCCL weight sync: {request.master_address}:{request.master_port} "
            f"(rank=1, world_size={request.world_size})"
        )

        pg = _stateless_init_process_group(
            master_address=request.master_address,
            master_port=request.master_port,
            rank=1,  # logprob server is rank 1
            world_size=request.world_size,
            device=DEVICE,
        )
        logprob_nccl_comm = PyNcclCommunicator(group=pg, device=DEVICE)
        logger.info("NCCL weight sync communicator ready.")
        return {"status": "ok"}

    @app.post("/sync_weights")
    def sync_weights():
        """Receive weights from trainer via NCCL, EMA blend into model."""
        assert logprob_nccl_comm is not None, "Call /init_weight_sync first"

        for param in model.parameters():
            incoming = torch.empty_like(param.data)
            logprob_nccl_comm.broadcast(incoming, src=0)
            param.data.lerp_(incoming, EMA_ALPHA)

        logger.debug(f"Weights EMA-blended (alpha={EMA_ALPHA}).")
        return {"status": "ok"}

    # ---- Run server ----
    logger.info(f"Starting HTTP server on port {LOGPROB_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=LOGPROB_PORT, log_level="warning")


if __name__ == "__main__":
    main()
