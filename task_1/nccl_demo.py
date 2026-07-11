"""
NCCL weight transfer demo: HF model (GPU 1) -> vLLM server (GPU 0).

Adapted from vllm/examples/rl/rlhf_http_nccl.py

Prerequisites:
    1. Run setup.sh to create venvs
    2. Start vLLM server: bash task_1/start_server.sh
    3. Run this script:   .hf_venv/bin/python task_1/nccl_demo.py

Flow:
    1. Query vLLM (dummy weights) -> gibberish
    2. Load real HF model on GPU 1
    3. Transfer real weights via NCCL -> sensible output
    4. Randomly perturb weights, transfer again -> different output
"""

import threading

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"
TRAINER_DEVICE = "cuda:1"


# ---------------------------------------------------------------------------
# HTTP helpers (control plane)
# ---------------------------------------------------------------------------

def get_world_size() -> int:
    r = requests.get(f"{BASE_URL}/get_world_size", timeout=10)
    r.raise_for_status()
    return r.json()["world_size"]


def init_weight_transfer_engine(
    master_address: str, master_port: int,
    rank_offset: int, world_size: int,
) -> None:
    r = requests.post(
        f"{BASE_URL}/init_weight_transfer_engine",
        json={"init_info": dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
        )},
        timeout=60,
    )
    r.raise_for_status()


def start_weight_update() -> None:
    r = requests.post(f"{BASE_URL}/start_weight_update", json={}, timeout=60)
    r.raise_for_status()


def update_weights(
    names: list[str], dtype_names: list[str],
    shapes: list[list[int]], packed: bool = True,
) -> None:
    r = requests.post(
        f"{BASE_URL}/update_weights",
        json={"update_info": dict(
            names=names, dtype_names=dtype_names,
            shapes=shapes, packed=packed,
        )},
        timeout=300,
    )
    r.raise_for_status()


def finish_weight_update() -> None:
    r = requests.post(f"{BASE_URL}/finish_weight_update", json={}, timeout=60)
    r.raise_for_status()


def pause_generation() -> None:
    r = requests.post(f"{BASE_URL}/pause", timeout=60)
    r.raise_for_status()


def resume_generation() -> None:
    r = requests.post(f"{BASE_URL}/resume", timeout=60)
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Inference helper (via OpenAI-compatible API)
# ---------------------------------------------------------------------------

def generate(client: OpenAI, prompts: list[str]) -> list[str]:
    results = []
    for prompt in prompts:
        resp = client.completions.create(
            model=MODEL_NAME, prompt=prompt,
            max_tokens=32, temperature=0,
        )
        results.append(resp.choices[0].text)
    return results


def print_outputs(label: str, prompts: list[str], outputs: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for prompt, text in zip(prompts, outputs):
        print(f"  Prompt:    {prompt!r}")
        print(f"  Generated: {text!r}")
        print(f"  {'-' * 56}")


# ---------------------------------------------------------------------------
# Weight metadata helpers
# ---------------------------------------------------------------------------

def get_weight_metadata(model: torch.nn.Module):
    names, dtype_names, shapes = [], [], []
    for name, p in model.named_parameters():
        names.append(name)
        dtype_names.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))
    return names, dtype_names, shapes


# ---------------------------------------------------------------------------
# NCCL weight transfer
# ---------------------------------------------------------------------------

def transfer_weights(
    model: torch.nn.Module,
    model_update_group,
    packed: bool = True,
) -> None:
    """Broadcast all model weights to vLLM via NCCL."""
    names, dtype_names, shapes = get_weight_metadata(model)

    # Pause inference during weight update
    pause_generation()
    start_weight_update()

    # update_weights HTTP call blocks until NCCL receives complete,
    # so run it in a thread.
    t = threading.Thread(
        target=update_weights,
        args=(names, dtype_names, shapes, packed),
    )
    t.start()

    # Broadcast weights via NCCL (data plane)
    trainer_args = NCCLTrainerSendWeightsArgs(
        group=model_update_group, packed=packed,
    )
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=model.named_parameters(),
        trainer_args=trainer_args,
    )
    t.join()

    finish_weight_update()
    resume_generation()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

    # --- Phase 1: generate with dummy weights (expect gibberish) ---
    outputs_dummy = generate(client, prompts)
    print_outputs("PHASE 1: Dummy weights (gibberish expected)", prompts, outputs_dummy)

    # --- Load HF model on GPU 1 ---
    print(f"\nLoading HF model on {TRAINER_DEVICE}...")
    torch.accelerator.set_device_index(TRAINER_DEVICE)
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(TRAINER_DEVICE)
    print("Model loaded.")

    # --- Set up NCCL group ---
    inference_world_size = get_world_size()
    world_size = inference_world_size + 1  # +1 for trainer
    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1

    print(f"Initializing NCCL group: {master_address}:{master_port} "
          f"(world_size={world_size})")

    # Init on vLLM side (async HTTP call) + trainer side (blocking)
    init_thread = threading.Thread(
        target=init_weight_transfer_engine,
        args=(master_address, master_port, rank_offset, world_size),
    )
    init_thread.start()

    model_update_group = NCCLWeightTransferEngine.trainer_init(
        dict(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
        ),
    )
    init_thread.join()
    print("NCCL group initialized.")

    # --- Phase 2: transfer real weights -> expect sensible output ---
    print("\nTransferring real weights via NCCL...")
    transfer_weights(train_model, model_update_group)

    outputs_real = generate(client, prompts)
    print_outputs("PHASE 2: Real weights (sensible output expected)", prompts, outputs_real)

    # --- Phase 3: perturb weights, transfer again ---
    print("\nRandomly perturbing model weights...")
    with torch.no_grad():
        for param in train_model.parameters():
            param.data += torch.randn_like(param.data) * 0.1
    print("Weights perturbed.")

    print("Transferring perturbed weights via NCCL...")
    transfer_weights(train_model, model_update_group)

    outputs_perturbed = generate(client, prompts)
    print_outputs("PHASE 3: Perturbed weights (different output expected)", prompts, outputs_perturbed)

    print("\nDone. All 3 phases complete.")


if __name__ == "__main__":
    main()
