"""vLLM HTTP client for generation + NCCL weight sync.

Adapted from the HF reference implementation. Key change: weight sync converts
Megatron parameter format to HuggingFace format before sending to vLLM, since
vLLM expects HF-format parameter names.
"""

import json
import os
import threading
import time

import requests
import torch
from loguru import logger

from megatron_trainer.config import (
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    LORA_ADAPTER_NAME,
    MODEL_NAME,
    TRAIN_MODE,
    VLLM_BASE_URL,
    VLLM_BASE_URLS,
    VLLM_SEED,
)
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
) -> tuple[str, str, list[float] | None]:
    """Generate a completion via vLLM's OpenAI-compatible API.

    Returns (generated_text, finish_reason, token_logprobs).
    finish_reason is "length" if max_tokens was hit, "stop" if natural stop.
    token_logprobs is the per-token log-probability of each generated token
    under the actual sampling distribution (1:1 aligned with the output
    tokens), or None if the server did not return logprobs. Used as the
    rollout proposal logp for importance sampling.

    In TRAIN_MODE=lora requests select the hot-swapped policy adapter via
    "model": LORA_ADAPTER_NAME (unknown adapter name → 404, never a silent
    base fallback).
    """
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": LORA_ADAPTER_NAME if TRAIN_MODE == "lora" else MODEL_NAME,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": 1,
            "skip_special_tokens": False,
            **({"seed": VLLM_SEED} if VLLM_SEED is not None else {}),
        },
        # 2048-token completions at ~20 tok/s (slow processed_logprobs path)
        # run ~100s; 180s read timeout killed runs on tail-heavy prompts.
        timeout=600,
    )
    if not resp.ok:
        logger.error(f"vLLM completions error ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    choice = resp.json()["choices"][0]
    logprobs = None
    if choice.get("logprobs") is not None:
        logprobs = choice["logprobs"].get("token_logprobs")
    return choice["text"], choice["finish_reason"], logprobs


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

    requests.post(f"{base_url}/pause?mode=keep", timeout=60).raise_for_status()
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
    rank: int = 0,
) -> None:
    """Push model weights to all vLLM instances sequentially.

    Under FSDP the HF-format export passes are collectives — non-zero ranks
    must consume them in lockstep with rank 0 (which does the actual NCCL
    send + HTTP orchestration). Under DDP only rank 0 runs.
    """
    if rank == 0:
        for url, group in zip(VLLM_BASE_URLS, model_update_groups):
            _sync_weights_to_single_vllm(model, device, url, group)
        logger.info(f"Weights synced to all {len(model_update_groups)} vLLM instances.")
        return

    # Consume the same export passes rank 0 performs: one metadata pass per
    # instance (cached after the first call) + one send-pass export each.
    for _ in VLLM_BASE_URLS:
        get_hf_weight_metadata(model)
        for _ in export_hf_weights_iter(model):
            pass


# ---------------------------------------------------------------------------
# LoRA adapter sync (TRAIN_MODE=lora) — hot-swap instead of weight transfer
# ---------------------------------------------------------------------------

def push_lora_adapter(
    model: torch.nn.Module,
    adapter_dir: str,
    rank: int = 0,
) -> None:
    """Export the LoRA adapter and hot-swap it into all vLLM instances.

    Collective export (all ranks participate in the adapter gather — the
    bridge save_hf_adapter is collective); rank 0 does the HTTP push.

    Push protocol per instance:
        1. Drain barrier: /pause (mode=keep) — in-flight rollouts finish
           before the swap. vLLM has no per-request adapter versioning; a
           request spanning the swap would silently continue with new weights.
        2. POST /v1/load_lora_adapter {lora_name, lora_path, load_inplace}
           — same name keeps the preallocated GPU slot; zero new allocation.
           load-before-remove: a failed load keeps the OLD adapter serving.
        3. /resume

    NOTE: never use /unload_lora_adapter + load instead of load_inplace —
    unload is frontend-only and in-flight requests resurrect the old path.
    """
    from megatron_trainer.model_utils import save_hf_adapter_checkpoint

    save_hf_adapter_checkpoint(model, adapter_dir, rank=rank)
    if rank != 0:
        return

    for url in VLLM_BASE_URLS:
        t0 = time.monotonic()
        requests.post(f"{url}/pause?mode=keep", timeout=60).raise_for_status()
        r = requests.post(
            f"{url}/v1/load_lora_adapter",
            json={
                "lora_name": LORA_ADAPTER_NAME,
                "lora_path": os.path.abspath(adapter_dir),
                "load_inplace": True,
            },
            timeout=120,
        )
        if not r.ok:
            # Failed load keeps the old adapter serving (load-before-remove);
            # log the raw body and retry next step. Never crash the loop.
            logger.error(
                f"LoRA push failed to {url} ({r.status_code}): {r.text} — "
                f"old adapter keeps serving"
            )
        else:
            logger.info(f"LoRA adapter pushed to {url}: {r.text.strip()}")
        requests.post(f"{url}/resume", timeout=60).raise_for_status()
        swap_ms = (time.monotonic() - t0) * 1000
        logger.info(f"adapter/swap_latency_ms={swap_ms:.0f} url={url}")
