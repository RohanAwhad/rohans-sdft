"""vLLM HTTP client for generation + NCCL weight sync (task 1 pattern).

Two communication channels:
    HTTP   — generation requests (/v1/completions), control plane
    NCCL   — weight transfer via vLLM's NCCLWeightTransferEngine
"""

import threading
import time

import requests
import torch
from loguru import logger

from src.config import GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P, MODEL_NAME, VLLM_BASE_URL


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

def wait_for_vllm(timeout: int = 300) -> None:
    """Block until vLLM server is healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info("vLLM server is healthy.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server not healthy after {timeout}s")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def vllm_generate(
    prompt_text: str,
    max_tokens: int = GEN_MAX_NEW_TOKENS,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
) -> str:
    """Generate a completion via vLLM's OpenAI-compatible API."""
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Weight sync (HTTP control plane + NCCL data plane)
# ---------------------------------------------------------------------------

def _get_weight_metadata(model: torch.nn.Module):
    names, dtype_names, shapes = [], [], []
    for name, p in model.named_parameters():
        names.append(name)
        dtype_names.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))
    return names, dtype_names, shapes


def init_vllm_weight_engine(device: torch.device):
    """Set up the NCCL group between trainer and vLLM server.

    Returns the model_update_group handle used for subsequent weight transfers.
    """
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    # Get vLLM inference world size
    r = requests.get(f"{VLLM_BASE_URL}/get_world_size", timeout=10)
    r.raise_for_status()
    inference_world_size = r.json()["world_size"]
    world_size = inference_world_size + 1  # +1 for trainer

    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1

    logger.info(
        f"Initializing vLLM NCCL group: {master_address}:{master_port} "
        f"(world_size={world_size})"
    )

    # Both sides must init simultaneously — use a thread for the HTTP call
    def _init_server_side():
        requests.post(
            f"{VLLM_BASE_URL}/init_weight_transfer_engine",
            json={
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                }
            },
            timeout=60,
        ).raise_for_status()

    t = threading.Thread(target=_init_server_side)
    t.start()

    model_update_group = NCCLWeightTransferEngine.trainer_init(
        {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
        }
    )
    t.join()
    logger.info("vLLM NCCL group initialized.")
    return model_update_group


def sync_weights_to_vllm(
    model: torch.nn.Module,
    device: torch.device,
    model_update_group,
) -> None:
    """Push current model weights to vLLM server via NCCL."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    names, dtype_names, shapes = _get_weight_metadata(model)

    # Pause inference during update
    requests.post(f"{VLLM_BASE_URL}/pause", timeout=60).raise_for_status()
    requests.post(f"{VLLM_BASE_URL}/start_weight_update", json={}, timeout=60).raise_for_status()

    # update_weights HTTP call blocks until NCCL recv completes — run in thread
    def _trigger_recv():
        requests.post(
            f"{VLLM_BASE_URL}/update_weights",
            json={
                "update_info": {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    "packed": True,
                }
            },
            timeout=300,
        ).raise_for_status()

    t = threading.Thread(target=_trigger_recv)
    t.start()

    # NCCL send (data plane)
    trainer_args = NCCLTrainerSendWeightsArgs(
        group=model_update_group, packed=True
    )
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=model.named_parameters(),
        trainer_args=trainer_args,
    )
    t.join()

    requests.post(f"{VLLM_BASE_URL}/finish_weight_update", json={}, timeout=60).raise_for_status()
    requests.post(f"{VLLM_BASE_URL}/resume", timeout=60).raise_for_status()
    logger.info("Weights synced to vLLM.")
