"""FastAPI logprob server (NCCL rank 1).

Loads an HF model on GPU, serves log-probabilities over HTTP,
and receives weight updates from the trainer process via NCCL broadcast.

Weight sync is triggered via POST /sync_weights — the handler blocks on
broadcast_weights() until the trainer (rank 0) also calls broadcast.
No background NCCL thread needed.
"""

import io
import os
import threading

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nccl_comm import init_nccl, broadcast_weights, cleanup

# --- Config (all overridable via env) ---
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda:0")
NCCL_RANK = 1
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
MASTER_PORT = int(os.environ.get("MASTER_PORT", "29500"))

# --- Globals ---
app = FastAPI()
model: torch.nn.Module | None = None
tokenizer: AutoTokenizer | None = None
weight_lock = threading.Lock()


class LogprobRequest(BaseModel):
    token_ids: list[int]


@app.post("/logprobs")
def get_logprobs(req: LogprobRequest) -> Response:
    """Return log-probabilities for the given token sequence.

    Response: binary numpy array of shape (seq_len, vocab_size), float32.
    """
    with weight_lock:
        input_ids = torch.tensor([req.token_ids], device=DEVICE)
        with torch.no_grad():
            logits = model(input_ids).logits[0]  # (seq_len, vocab_size)
        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs_np = logprobs.cpu().float().numpy()

    buf = io.BytesIO()
    np.save(buf, logprobs_np)
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


@app.post("/sync_weights")
def sync_weights():
    """Receive weight update from trainer via NCCL broadcast.

    Blocks until the trainer (rank 0) also calls broadcast_weights.
    The caller must ensure the trainer broadcasts concurrently.
    """
    print("[server] /sync_weights called — entering NCCL broadcast ...", flush=True)
    with weight_lock:
        broadcast_weights(model, src=0)
    print("[server] /sync_weights done — weights updated.", flush=True)
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


def main() -> None:
    global model, tokenizer

    print(f"[server] Loading model {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()
    print("[server] Model loaded.", flush=True)

    print(f"[server] Initializing NCCL (rank={NCCL_RANK}) ...", flush=True)
    init_nccl(rank=NCCL_RANK, master_port=MASTER_PORT)
    print("[server] NCCL initialized.", flush=True)

    print(f"[server] Starting HTTP on port {SERVER_PORT} ...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    cleanup()


if __name__ == "__main__":
    main()
