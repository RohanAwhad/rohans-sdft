"""TCP + HTTP client + PyNcclCommunicator for logprob server communication.

Replaces the NCCL command protocol from nccl_comm.py. Three responsibilities:
    1. request_teacher_log_probs_tcp() — get teacher log-probs via persistent TCP
    2. init_logprob_weight_engine() — set up standalone NCCL for weight sync
    3. sync_weights_to_logprob_server() — push weights via NCCL (HTTP-triggered)

HTTP logprob functions are kept as debug fallbacks.
"""

import socket
import struct
import threading
import time

import numpy as np
import requests
import torch
from loguru import logger

from megatron_trainer.config import GEN_MAX_NEW_TOKENS, LOGPROB_BASE_URL, LOGPROB_TCP_PORT
from megatron_trainer.model_utils import gather_raw_params_iter


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

def wait_for_logprob_server(timeout: int = 300) -> None:
    """Wait for logprob server HTTP to be healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{LOGPROB_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Logprob server is healthy.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Logprob server not healthy after {timeout}s")


# ---------------------------------------------------------------------------
# Teacher log-probs via TCP (fast path)
# ---------------------------------------------------------------------------
# Persistent per-rank connection (each trainer rank is its own process, so
# module state is naturally per-rank) + ONE preallocated recv buffer reused
# across all calls. Per-request multi-GB allocations zero-fill under the GIL
# and stall the peer's send loop (devlogs Known Issues #11) — do NOT allocate
# per request here.

_tcp_sock: socket.socket | None = None
_recv_buf: bytearray | None = None


def _get_tcp_connection(vocab_size: int) -> tuple[socket.socket, bytearray]:
    global _tcp_sock, _recv_buf
    if _tcp_sock is None:
        sock = socket.create_connection(("127.0.0.1", LOGPROB_TCP_PORT), timeout=60)
        sock.settimeout(None)  # blocking; server death surfaces as ConnectionError
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _recv_buf = bytearray(GEN_MAX_NEW_TOKENS * vocab_size * 2)  # fp16, ~2.5 GB
        logger.info(
            f"TCP logprob connection established (port {LOGPROB_TCP_PORT}, "
            f"recv buf {len(_recv_buf)/1e9:.2f} GB)"
        )
        _tcp_sock = sock
    return _tcp_sock, _recv_buf


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    rem = n
    while rem > 0:
        d = sock.recv(rem)
        if not d:
            raise ConnectionError("logprob server closed connection")
        chunks.append(d)
        rem -= len(d)
    return b"".join(chunks)


def request_teacher_log_probs_tcp(
    token_ids: list[int],
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Request teacher log-probs via TCP from logprob server.

    Drop-in replacement for request_teacher_log_probs_http — same contract.
    Returns: (completion_len, vocab_size) tensor in bfloat16 on device.
    """
    sock, buf = _get_tcp_connection(vocab_size)

    seq_len = len(token_ids)
    ids_bytes = struct.pack(f"<{seq_len}i", *token_ids)
    sock.sendall(struct.pack("<ii", prompt_len, seq_len) + ids_bytes)

    (completion_len,) = struct.unpack("<i", _recv_exact(sock, 4))
    nbytes = completion_len * vocab_size * 2

    mv = memoryview(buf)
    got = 0
    while got < nbytes:
        n = sock.recv_into(mv[got:nbytes], nbytes - got)
        if n == 0:
            raise ConnectionError("logprob server closed connection")
        got += n

    # Zero-copy view into buf, single H2D copy (buf reuse next call is safe —
    # .to(device) has already copied out by then).
    lp = torch.frombuffer(mv[:nbytes], dtype=torch.float16).reshape(completion_len, vocab_size)
    return lp.to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Teacher log-probs via HTTP
# ---------------------------------------------------------------------------

def request_teacher_log_probs_http(
    token_ids: list[int],
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Request teacher log-probs via HTTP from logprob server.

    Any trainer rank can call this independently — no coordination needed.
    Returns: (completion_len, vocab_size) tensor in bfloat16.
    """
    completion_len = len(token_ids) - prompt_len
    resp = requests.post(
        f"{LOGPROB_BASE_URL}/logprobs",
        json={"token_ids": token_ids, "prompt_len": prompt_len},
        timeout=120,
    )
    if not resp.ok:
        logger.error(f"Logprob server error ({resp.status_code}): {resp.text}")
        resp.raise_for_status()

    # Decode binary response (float16 numpy → bfloat16 torch)
    log_probs_np = np.frombuffer(resp.content, dtype=np.float16).copy()
    log_probs_np = log_probs_np.reshape(completion_len, vocab_size)
    return torch.from_numpy(log_probs_np).to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Weight sync engine (PyNcclCommunicator, mirrors vllm_utils pattern)
# ---------------------------------------------------------------------------

def init_logprob_weight_engine(device: torch.device):
    """Create standalone NCCL communicator for logprob weight sync.

    Mirrors init_vllm_weight_engine() in vllm_utils.py. Uses the same
    internal mechanism as NCCLWeightTransferEngine.trainer_init():
    _stateless_init_process_group() creates a StatelessProcessGroup,
    then PyNcclCommunicator wraps it.

    Trainer is rank 0, logprob server is rank 1 in this 2-process group.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.utils.network_utils import get_ip, get_open_port

    master_address = get_ip()
    master_port = get_open_port()

    logger.info(
        f"Initializing logprob NCCL group: {master_address}:{master_port} "
        f"(world_size=2)"
    )

    def _init_server_side():
        requests.post(
            f"{LOGPROB_BASE_URL}/init_weight_sync",
            json={
                "master_address": master_address,
                "master_port": master_port,
                "world_size": 2,
            },
            timeout=60,
        ).raise_for_status()

    t = threading.Thread(target=_init_server_side)
    t.start()

    # Trainer is rank 0 in this 2-process NCCL group
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=0, world_size=2,
    )
    comm = PyNcclCommunicator(pg, device=device)
    t.join()
    logger.info("Logprob NCCL weight engine initialized.")
    return comm


def sync_weights_to_logprob_server(
    model: torch.nn.Module,
    logprob_comm,
    rank: int = 0,
) -> None:
    """Push trainer weights to logprob server via standalone NCCL.

    Mirrors sync_weights_to_vllm() in vllm_utils.py — background thread
    triggers the server to enter NCCL receive loop, main thread broadcasts.
    Both sides iterate model.parameters() in the same order (both are
    Megatron models loaded via the same AutoBridge path).

    Under FSDP all trainer ranks must enter gather_raw_params_iter() in
    lockstep (collective all-gather); only rank 0 broadcasts on the logprob
    NCCL group. Under DDP this is equivalent to the old rank-0-only behavior.

    EMA blending happens on the server side (in the /sync_weights handler).
    """

    def _trigger_recv():
        requests.post(f"{LOGPROB_BASE_URL}/sync_weights", timeout=300).raise_for_status()

    if rank == 0:
        t = threading.Thread(target=_trigger_recv)
        t.start()

    for full in gather_raw_params_iter(model):
        if rank == 0:
            logprob_comm.broadcast(full, src=0)

    if rank == 0:
        t.join()
        logger.debug("Weights synced to logprob server.")
