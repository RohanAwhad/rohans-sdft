"""Megatron model initialization and weight conversion utilities.

Handles:
    - torch.distributed + Megatron parallel state initialization
    - Model loading via AutoBridge
    - Megatron→HF weight format conversion for vLLM sync
    - HF checkpoint export
"""

import os
import socket
from typing import Iterator

import torch
import torch.distributed as dist
from loguru import logger


_bridge_instance = None
_hf_weight_meta_cache = None


def _get_free_port() -> int:
    """Get an available port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed_standalone() -> None:
    """Initialize torch.distributed world_size=1 for standalone Megatron model loading.

    Used by both the trainer (Phase 1, single-rank) and the logprob server.
    Each process gets its own independent torch.distributed world — no shared
    process group with other processes. CUDA_VISIBLE_DEVICES should be set
    externally to scope to a single GPU.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_get_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    logger.info("torch.distributed initialized (standalone, world_size=1)")


def init_distributed(rank: int, world_size: int = 2, master_port: int = 29500) -> None:
    """Initialize torch.distributed for trainer <-> logprob server communication.

    Sets up a 2-rank NCCL group (trainer=rank0, logprob_server=rank1).
    Megatron parallel state (TP=1, PP=1, DP=world_size) is initialized
    automatically by to_megatron_model() when load_model() is called.
    The DP group is never used for gradient sync since we run our own loop.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(0)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"torch.distributed initialized (rank={rank}, world_size={world_size})")


def load_model(hf_model_path: str) -> torch.nn.Module:
    """Load a Megatron-Core GPTModel from HuggingFace pretrained weights.

    Uses AutoBridge to convert HF weights → MCore model format in-memory.
    The bridge instance is cached for later use in weight conversion.
    Returns the unwrapped model (no DDP).

    Disables gradient_accumulation_fusion since we use a standard PyTorch
    optimizer instead of Megatron's distributed optimizer (which sets up
    main_grad buffers).
    """
    global _bridge_instance

    from megatron.bridge import AutoBridge

    logger.info(f"Loading model via AutoBridge: {hf_model_path}")
    bridge = AutoBridge.from_hf_pretrained(hf_model_path)
    _bridge_instance = bridge

    provider = bridge.to_megatron_provider(load_weights=True)
    if hasattr(provider, "finalize"):
        provider.finalize()

    provider.gradient_accumulation_fusion = False
    provider.async_tensor_model_parallel_allreduce = False

    models = provider.provide_distributed_model(wrap_with_ddp=False)
    model = models[0]

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_bridge():
    """Return the cached AutoBridge instance."""
    if _bridge_instance is None:
        raise RuntimeError("Call load_model() first")
    return _bridge_instance


def export_hf_weights_iter(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (hf_name, tensor) pairs by converting Megatron weights to HF format.

    Handles QKV unfusing, gate+up unfusing, and parameter renaming.
    Tensors stay on their original device (GPU).
    """
    bridge = get_bridge()
    for hf_tuple in bridge.export_hf_weights(model, cpu=False):
        yield (hf_tuple.param_name, hf_tuple.weight)


def get_hf_weight_metadata(model: torch.nn.Module) -> tuple[list[str], list[str], list[list[int]]]:
    """Get HF-format weight metadata (names, dtypes, shapes) for vLLM sync.

    Cached after first call since metadata is static.
    """
    global _hf_weight_meta_cache
    if _hf_weight_meta_cache is not None:
        return _hf_weight_meta_cache

    names, dtype_names, shapes = [], [], []
    for hf_name, weight in export_hf_weights_iter(model):
        names.append(hf_name)
        dtype_names.append(str(weight.dtype).split(".")[-1])
        shapes.append(list(weight.shape))

    _hf_weight_meta_cache = (names, dtype_names, shapes)
    logger.info(f"HF weight metadata cached: {len(names)} tensors")
    return _hf_weight_meta_cache


def save_hf_checkpoint(model: torch.nn.Module, save_dir: str, tokenizer=None) -> None:
    """Export Megatron model to HuggingFace format for eval compatibility.

    Avoids bridge.save_hf_pretrained() which uses distributed barriers —
    the logprob server (rank 1) is in its command loop and can't participate.
    Instead, manually exports weights via export_hf_weights + safetensors.
    """
    import json
    from safetensors.torch import save_file

    bridge = get_bridge()
    os.makedirs(save_dir, exist_ok=True)

    weights = {}
    for hf_name, weight in export_hf_weights_iter(model):
        weights[hf_name] = weight.contiguous().cpu()

    save_file(weights, os.path.join(save_dir, "model.safetensors"))

    hf_config = getattr(bridge.hf_pretrained, "config", bridge.hf_pretrained)
    if hf_config is not None:
        hf_config.save_pretrained(save_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
        # Fix tokenizer_config.json: the NeMo container's transformers saves
        # extra_special_tokens as a list, but host transformers expects a dict.
        import json
        tc_path = os.path.join(save_dir, "tokenizer_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                tc = json.load(f)
            if isinstance(tc.get("extra_special_tokens"), list):
                tc["extra_special_tokens"] = {}
                with open(tc_path, "w") as f:
                    json.dump(tc, f, indent=2)

    logger.info(f"HF checkpoint saved: {save_dir} ({len(weights)} tensors)")


def cleanup() -> None:
    """Destroy process groups."""
    try:
        from megatron.core import parallel_state as mpu
        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        pass
    if dist.is_initialized():
        dist.destroy_process_group()
