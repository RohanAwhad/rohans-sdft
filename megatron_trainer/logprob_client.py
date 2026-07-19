"""HTTP client + PyNcclCommunicator for logprob server communication.

Replaces the NCCL command protocol from nccl_comm.py. Three responsibilities:
    1. request_teacher_log_probs_http() — get teacher log-probs via HTTP
    2. init_logprob_weight_engine() — set up standalone NCCL for weight sync
    3. sync_weights_to_logprob_server() — push weights via NCCL (HTTP-triggered)
"""

import threading
import time

import numpy as np
import requests
import torch
from loguru import logger

from megatron_trainer.config import LOGPROB_BASE_URL


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
) -> None:
    """Push trainer weights to logprob server via standalone NCCL.

    Mirrors sync_weights_to_vllm() in vllm_utils.py — background thread
    triggers the server to enter NCCL receive loop, main thread broadcasts.
    Both sides iterate model.parameters() in the same order (both are
    Megatron models loaded via the same AutoBridge path).

    EMA blending happens on the server side (in the /sync_weights handler).
    """

    def _trigger_recv():
        requests.post(f"{LOGPROB_BASE_URL}/sync_weights", timeout=300).raise_for_status()

    t = threading.Thread(target=_trigger_recv)
    t.start()

    for param in model.parameters():
        logprob_comm.broadcast(param.data, src=0)

    t.join()
    logger.debug("Weights synced to logprob server.")
