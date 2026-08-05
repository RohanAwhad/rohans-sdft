"""Teacher logprob server — TCP data plane + HTTP control plane.

Standalone process with its own torch.distributed world_size=1 (for Megatron
model loading only). Serves teacher log-probs via a persistent-TCP binary
protocol (fast path, replaces the HTTP /logprobs data plane) and accepts
weight sync via PyNcclCommunicator (initialized by trainer via HTTP handshake).

Two modes:
    TEACHER_MODEL_PATH unset  — teacher = student (HF_MODEL_PATH), Megatron bridge
                                load, per-step NCCL weight sync + EMA blend.
    TEACHER_MODEL_PATH set    — frozen external teacher loaded via plain HF
                                transformers (mxfp4 auto-detected for native
                                checkpoints like openai/gpt-oss-120b); weight
                                sync endpoints are disabled.

Endpoints:
    TCP  LOGPROB_TCP_PORT  — teacher log-probs (length-prefixed binary, keepalive)
                             Request:  [int32 prompt_len][int32 seq_len][int32 x seq_len ids]
                             Response: [int32 completion_len][fp16 x C*V logprobs]
    GET  /health          — readiness probe
    POST /logprobs        — compute log-probs, return binary (float16) [debug fallback]
    POST /logprobs_batch  — batched log-probs, return binary (length-prefixed) [unused]
    POST /init_weight_sync — NCCL communicator init handshake (disabled for frozen teacher)
    POST /sync_weights     — receive weights via NCCL + EMA blend (disabled for frozen teacher)
"""

import os
import socket
import struct
import threading

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel

from megatron_trainer.config import (
    EMA_ALPHA,
    HF_MODEL_PATH,
    LOGPROB_PORT,
    LOGPROB_TCP_PORT,
    TEACHER_MODEL_PATH,
)
from megatron_trainer.model_utils import init_distributed_standalone, load_model

DEVICE = torch.device("cuda:0")
LOGPROB_BATCH_SIZE = int(os.environ.get("LOGPROB_BATCH_SIZE", "16"))

# Frozen external teacher (e.g. openai/gpt-oss-120b mxfp4): loaded via plain HF
# transformers, no weight sync. Empty = teacher is the student model (HF_MODEL_PATH).
USE_EXTERNAL_TEACHER = bool(TEACHER_MODEL_PATH)


class LogprobRequest(BaseModel):
    token_ids: list[int]
    prompt_len: int


class BatchLogprobRequest(BaseModel):
    items: list[LogprobRequest]


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
    # (only needed for the bridge path; the frozen-teacher HF path skips it)
    if not USE_EXTERNAL_TEACHER:
        init_distributed_standalone()
        logger.info("Distributed init complete (standalone, world_size=1).")

    # ---- Load model ----
    model_path = TEACHER_MODEL_PATH or HF_MODEL_PATH
    logger.info(f"Loading teacher model: {model_path} (external_frozen={USE_EXTERNAL_TEACHER})")
    if USE_EXTERNAL_TEACHER:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_PATH, torch_dtype=torch.bfloat16
        )
        model.to(DEVICE)

        # gpt-oss eager attention materializes full (H, S, S) scores — at
        # H=64, S=8192 that is 8.6 GiB per transient (bf16) and pushes the 120b
        # past 80 GB. Replace it with a blockwise-exact version (query rows in
        # chunks of 1024, two-pass row-max trick): same softmax values as the
        # full-row version up to fp32 accumulation. Patch is safe: "eager" is
        # not in the AttentionInterface registry, so get_interface() resolves
        # this module-global at call time.
        from transformers.models.gpt_oss import modeling_gpt_oss as _gpt_oss_mod

        _ATTN_CHUNK = 1024

        def _chunked_eager_attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            key = _gpt_oss_mod.repeat_kv(key, module.num_key_value_groups)
            value = _gpt_oss_mod.repeat_kv(value, module.num_key_value_groups)
            b, h, s, d = query.shape
            k_t = key.transpose(2, 3)
            sink = module.sinks.to(query.dtype).reshape(1, -1, 1, 1)

            # Pass 1: per-row max over all blocks (and the sink column).
            bmaxes = []
            for st in range(0, s, _ATTN_CHUNK):
                en = min(st + _ATTN_CHUNK, s)
                sc = torch.matmul(query[:, :, st:en], k_t).mul_(scaling)
                if attention_mask is not None:
                    sc = sc + attention_mask[:, :, st:en]
                bmaxes.append(sc.max(dim=-1, keepdim=True).values)
            row_max = torch.cat(bmaxes, dim=2)  # (b, h, s, 1)
            row_max = torch.maximum(row_max, sink)

            # Pass 2: fp32 exp-sums + weighted value accumulation.
            sum_exp = torch.zeros(b, h, s, 1, dtype=torch.float32, device=query.device)
            attn_out = torch.zeros(b, h, s, d, dtype=torch.float32, device=query.device)
            for st in range(0, s, _ATTN_CHUNK):
                en = min(st + _ATTN_CHUNK, s)
                sc = torch.matmul(query[:, :, st:en], k_t).mul_(scaling)
                if attention_mask is not None:
                    sc = sc + attention_mask[:, :, st:en]
                w = torch.exp((sc - row_max[:, :, st:en]).float())  # (b, h, chunk, s) fp32
                sum_exp[:, :, st:en] = w.sum(dim=-1, keepdim=True)
                attn_out[:, :, st:en] = torch.matmul(w, value.float())  # fp32 accum
            sum_exp = sum_exp + torch.exp((sink - row_max).float())
            attn_out = attn_out / sum_exp
            attn_output = attn_out.transpose(1, 2).contiguous().to(key.dtype)
            return attn_output, None

        _gpt_oss_mod.eager_attention_forward = _chunked_eager_attn
        logger.info("External teacher loaded via HF transformers (mxfp4 auto-detected if native; chunked attention active).")
    else:
        model = load_model(HF_MODEL_PATH)
        logger.info("Teacher = student model loaded via Megatron bridge.")
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # ---- PyNcclCommunicator — initialized later via HTTP handshake ----
    logprob_nccl_comm = None
    request_count = 0
    # Serialize model access: FastAPI runs sync endpoints in a threadpool,
    # and Megatron's global RNG state tracker is not thread-safe.
    model_lock = threading.Lock()

    # ---- Shared compute: forward → completion log-probs as CPU fp16 (C, V) ----
    # Caller must hold model_lock. fp16 cast happens on GPU (halves D2H vs
    # casting on CPU). Returns a contiguous CPU tensor suitable for zero-copy
    # sends via memoryview.
    def _model_logits(model_output) -> torch.Tensor:
        # MCore returns a tuple, HF returns a ModelOutput dataclass — both
        # expose logits as the first element; normalize to a plain tensor.
        # NOTE: HF keeps a leading batch dim (1, S, V); the single-request
        # caller strips it (the bridge model already returns (S, V)).
        return model_output.logits if hasattr(model_output, "logits") else model_output[0]

    def compute_logprobs_fp16(token_ids: list[int], prompt_len: int) -> torch.Tensor:
        seq_len = len(token_ids)
        completion_len = seq_len - prompt_len
        ids = torch.tensor(token_ids, device=DEVICE, dtype=torch.long)

        input_ids = ids.unsqueeze(0)
        position_ids = torch.arange(seq_len, device=DEVICE, dtype=torch.long).unsqueeze(0)
        logits = _model_logits(
            model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        )  # (S, V) or (1, S, V)
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # HF keeps the batch dim

        comp_logits = logits[prompt_len - 1 : prompt_len + completion_len - 1]  # (C, V)
        # Chunked fp32 log_softmax: a full (C, V) fp32 softmax is ~4.9 GiB at
        # C=6144 — chunking rows caps the transient at ~0.8 GiB.
        log_probs = torch.empty(
            completion_len, logits.size(-1), dtype=torch.float16, device=DEVICE
        )
        for s in range(0, completion_len, 1024):
            e = min(s + 1024, completion_len)
            log_probs[s:e] = F.log_softmax(comp_logits[s:e].float(), dim=-1).to(torch.float16)
        # .contiguous(): mxfp4 kernel outputs can be strided — memoryview.cast("B")
        # requires C-contiguous buffers for the zero-copy send.
        return log_probs.cpu().contiguous()  # (C, V) CPU fp16

    # ---- TCP data plane ----
    # One handler thread per connection (one per trainer rank). model_lock is
    # held only around compute — the ~0.7s socket send happens outside the
    # lock so other ranks' forwards overlap with sends.
    def recv_exact(conn: socket.socket, n: int) -> bytes:
        chunks = []
        rem = n
        while rem > 0:
            d = conn.recv(min(1 << 20, rem))
            if not d:
                raise ConnectionError("client closed connection")
            chunks.append(d)
            rem -= len(d)
        return b"".join(chunks)

    def handle_tcp_conn(conn: socket.socket) -> None:
        nonlocal request_count
        while True:
            try:
                hdr = recv_exact(conn, 8)
                prompt_len, seq_len = struct.unpack("<ii", hdr)
                ids_bytes = recv_exact(conn, seq_len * 4)
            except ConnectionError:
                break
            token_ids = list(struct.unpack(f"<{seq_len}i", ids_bytes))

            with model_lock, torch.no_grad():
                lp = compute_logprobs_fp16(token_ids, prompt_len)  # (C, V) cpu fp16

            completion_len = seq_len - prompt_len
            conn.sendall(struct.pack("<i", completion_len))
            conn.sendall(memoryview(lp.numpy()).cast("B"))

            request_count += 1
            if request_count % 50 == 0:
                logger.info(f"Served {request_count} logprob requests (tcp)")
        conn.close()

    def tcp_listener() -> None:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", LOGPROB_TCP_PORT))
        srv.listen(8)
        logger.info(f"TCP logprob listener on port {LOGPROB_TCP_PORT}")
        while True:
            conn, addr = srv.accept()
            logger.info(f"TCP logprob client connected: {addr}")
            threading.Thread(target=handle_tcp_conn, args=(conn,), daemon=True).start()

    threading.Thread(target=tcp_listener, daemon=True).start()

    # ---- FastAPI app ----
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/logprobs")
    def compute_logprobs(request: LogprobRequest):
        nonlocal request_count

        with model_lock, torch.no_grad():
            lp = compute_logprobs_fp16(request.token_ids, request.prompt_len)

        # Binary response: float16 numpy bytes — ~145 MB for C=500, V=151936
        response_bytes = lp.numpy().tobytes()

        request_count += 1
        if request_count % 50 == 0:
            logger.info(f"Served {request_count} logprob requests")

        return Response(content=response_bytes, media_type="application/octet-stream")

    @app.post("/logprobs_batch")
    def compute_logprobs_batch(request: BatchLogprobRequest):
        """Batched teacher log-probs. Processes items in sub-batches of LOGPROB_BATCH_SIZE.

        Response format (binary):
            For each item: 4-byte int32 (completion_len), then completion_len * vocab_size float16 values.
        """
        nonlocal request_count

        all_items = request.items
        vocab_size: int | None = None
        response_parts: list[bytes] = []

        with model_lock, torch.no_grad():
            for chunk_start in range(0, len(all_items), LOGPROB_BATCH_SIZE):
                chunk = all_items[chunk_start : chunk_start + LOGPROB_BATCH_SIZE]
                B = len(chunk)

                # Pad sequences to max length in this chunk
                seq_lens = [len(item.token_ids) for item in chunk]
                max_seq_len = max(seq_lens)

                input_ids = torch.zeros(B, max_seq_len, device=DEVICE, dtype=torch.long)
                position_ids = torch.zeros(B, max_seq_len, device=DEVICE, dtype=torch.long)
                attention_mask = torch.zeros(B, max_seq_len, device=DEVICE, dtype=torch.long)

                for i, item in enumerate(chunk):
                    s = seq_lens[i]
                    input_ids[i, :s] = torch.tensor(item.token_ids, device=DEVICE, dtype=torch.long)
                    position_ids[i, :s] = torch.arange(s, device=DEVICE, dtype=torch.long)
                    attention_mask[i, :s] = 1

                logits = _model_logits(
                    model(
                        input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
                    )
                )  # (B, S_max, V)

                if vocab_size is None:
                    vocab_size = logits.size(-1)

                # Extract per-item completion logprobs
                for i, item in enumerate(chunk):
                    completion_len = seq_lens[i] - item.prompt_len
                    comp_logits = logits[i, item.prompt_len - 1 : item.prompt_len + completion_len - 1]
                    log_probs = F.log_softmax(comp_logits.float(), dim=-1)
                    blob = log_probs.cpu().to(torch.float16).numpy().tobytes()
                    response_parts.append(struct.pack("<i", completion_len))
                    response_parts.append(blob)

        request_count += len(all_items)
        if request_count % 50 < len(all_items):
            logger.info(f"Served {request_count} logprob requests (batch)")

        return Response(content=b"".join(response_parts), media_type="application/octet-stream")

    @app.post("/init_weight_sync")
    def init_weight_sync(request: NCCLInitRequest):
        """HTTP handshake: trainer rank 0 sends NCCL init info, we create our communicator."""
        nonlocal logprob_nccl_comm

        if USE_EXTERNAL_TEACHER:
            logger.info("init_weight_sync ignored: external frozen teacher (no weight sync).")
            return {"status": "disabled"}

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        logger.info(
            f"Initializing NCCL weight sync: {request.master_address}:{request.master_port} "
            f"(rank=1, world_size={request.world_size})"
        )

        pg = StatelessProcessGroup.create(
            host=request.master_address, port=request.master_port,
            rank=1, world_size=request.world_size,
        )
        logprob_nccl_comm = PyNcclCommunicator(pg, device=DEVICE)
        logger.info("NCCL weight sync communicator ready.")
        return {"status": "ok"}

    @app.post("/sync_weights")
    def sync_weights():
        """Receive weights from trainer via NCCL, EMA blend into model."""
        if USE_EXTERNAL_TEACHER:
            logger.info("sync_weights ignored: external frozen teacher (no weight sync).")
            return {"status": "disabled"}

        assert logprob_nccl_comm is not None, "Call /init_weight_sync first"

        with model_lock:
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
