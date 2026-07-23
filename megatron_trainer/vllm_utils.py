"""vLLM HTTP client for generation + NCCL weight sync.

Adapted from the HF reference implementation. Key change: weight sync converts
Megatron parameter format to HuggingFace format before sending to vLLM, since
vLLM expects HF-format parameter names.
"""

import threading
import time

import requests
import torch
from loguru import logger

from megatron_trainer.config import GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P, MODEL_NAME, VLLM_BASE_URL, VLLM_BASE_URLS
from megatron_trainer.model_utils import export_hf_weights_iter, get_hf_weight_metadata


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

def wait_for_vllm(timeout: int = 300) -> None:
    """Wait for all vLLM instances to become healthy."""
    for url in VLLM_BASE_URLS:
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{url}/v1/models", timeout=5)
                if r.status_code == 200:
                    logger.info(f"vLLM server healthy: {url}")
                    break
            except requests.ConnectionError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"vLLM server {url} not healthy after {timeout}s")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def vllm_generate(
    prompt_text: str,
    base_url: str = VLLM_BASE_URL,
    max_tokens: int = GEN_MAX_NEW_TOKENS,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
) -> tuple[str, str]:
    """Generate a completion via vLLM's OpenAI-compatible API.

    Returns (generated_text, finish_reason).
    finish_reason is "length" if max_tokens was hit, "stop" if natural stop.
    """
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=180,
    )
    if not resp.ok:
        logger.error(f"vLLM completions error ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    choice = resp.json()["choices"][0]
    return choice["text"], choice["finish_reason"]


# ---------------------------------------------------------------------------
# Weight sync (HTTP control plane + NCCL data plane)
# ---------------------------------------------------------------------------

def _init_single_vllm_weight_engine(base_url: str, device: torch.device):
    """Initialize NCCL weight transfer for a single vLLM instance."""
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    r = requests.get(f"{base_url}/get_world_size", timeout=10)
    r.raise_for_status()
    inference_world_size = r.json()["world_size"]
    world_size = inference_world_size + 1

    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1

    logger.info(
        f"Initializing vLLM NCCL group for {base_url}: {master_address}:{master_port} "
        f"(world_size={world_size})"
    )

    def _init_server_side():
        requests.post(
            f"{base_url}/init_weight_transfer_engine",
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
    logger.info(f"vLLM NCCL group initialized for {base_url}.")
    return model_update_group


def init_vllm_weight_engine(device: torch.device):
    """Initialize NCCL weight transfer for all vLLM instances.

    Returns a list of NCCL groups (one per instance).
    """
    groups = []
    for url in VLLM_BASE_URLS:
        group = _init_single_vllm_weight_engine(url, device)
        groups.append(group)
    logger.info(f"All {len(groups)} vLLM weight engines initialized.")
    return groups


def _sync_weights_to_single_vllm(
    model: torch.nn.Module,
    device: torch.device,
    base_url: str,
    model_update_group,
) -> None:
    """Push model weights to a single vLLM instance."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    names, dtype_names, shapes = get_hf_weight_metadata(model)

    requests.post(f"{base_url}/pause", timeout=60).raise_for_status()
    requests.post(f"{base_url}/start_weight_update", json={}, timeout=60).raise_for_status()

    def _trigger_recv():
        requests.post(
            f"{base_url}/update_weights",
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

    trainer_args = NCCLTrainerSendWeightsArgs(
        group=model_update_group, packed=True
    )
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=export_hf_weights_iter(model),
        trainer_args=trainer_args,
    )
    t.join()

    requests.post(f"{base_url}/finish_weight_update", json={}, timeout=60).raise_for_status()
    requests.post(f"{base_url}/resume", timeout=60).raise_for_status()
    logger.info(f"Weights synced to vLLM {base_url}.")


def sync_weights_to_vllm(
    model: torch.nn.Module,
    device: torch.device,
    model_update_groups: list,
) -> None:
    """Push model weights to all vLLM instances sequentially."""
    for url, group in zip(VLLM_BASE_URLS, model_update_groups):
        _sync_weights_to_single_vllm(model, device, url, group)
    logger.info(f"Weights synced to all {len(model_update_groups)} vLLM instances.")
