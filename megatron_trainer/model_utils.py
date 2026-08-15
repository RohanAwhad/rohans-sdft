"""Megatron model initialization and weight conversion utilities.

Handles:
    - torch.distributed + Megatron parallel state initialization
    - Model loading via AutoBridge
    - Megatron→HF weight format conversion for vLLM sync
    - HF checkpoint export
    - LoRA adapter application + HF PEFT adapter export (TRAIN_MODE=lora)
"""

import os
import socket
from typing import Iterator

import torch
import torch.distributed as dist
from loguru import logger


_bridge_instance = None
_hf_weight_meta_cache = None
_lora_config = None


def _get_free_port() -> int:
    """Get an available port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed_trainer() -> int:
    """Initialize torch.distributed for trainer DDP via torchrun.

    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    Returns local_rank for CUDA device selection.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    logger.info(
        f"torch.distributed initialized (trainer DDP, "
        f"rank={dist.get_rank()}, world_size={dist.get_world_size()}, local_rank={local_rank})"
    )
    return local_rank


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
    provider.async_tensor_parallel_allreduce = False

    models = provider.provide_distributed_model(wrap_with_ddp=False)
    model = models[0]

    # Activation recompute: required for long-completion backward (gpt-oss
    # thinking sequences hit 4k-8k tokens). NOTE: setting these on the
    # provider is a no-op (the bridge LLM builder never maps them into the
    # TransformerConfig) — they must be set on the built model's config, which
    # MCore's TransformerLayer reads at forward time.
    model.config.recompute_granularity = 'full'
    model.config.recompute_method = 'uniform'
    model.config.recompute_num_layers = 1

    # Ensure model-parallel RNG state is initialized (required by
    # TransformerEngine attention's dropout context even in eval mode)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(42)

    # Free HF model from GPU — only metadata is needed for weight conversion.
    # Saves ~16 GB GPU memory (critical for DDP with MegatronDDP main_grad buffers).
    if hasattr(bridge, 'hf_pretrained') and hasattr(bridge.hf_pretrained, 'to'):
        bridge.hf_pretrained.to('cpu')
        torch.cuda.empty_cache()
        logger.info("HF model moved to CPU to free GPU memory.")

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def register_fsdp_module_mappings() -> None:
    """Register MCore FSDP-wrapped module classes with the bridge's AutoMapping.

    torch's fully_shard() dynamically replaces wrapped module classes with
    FSDP-prefixed subclasses (FSDPColumnParallelLinear, FSDPTransformerLayer,
    ...). The bridge's parallelism detection matches on exact class name, so
    register the FSDP variants with the same parallelism types as their
    originals (the subclasses preserve all module attributes/weights).
    """
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    for name, ptype in (
        ("FSDPColumnParallelLinear", "column"),
        ("FSDPRowParallelLinear", "row"),
        ("FSDPLanguageModelEmbedding", "replicated"),
        ("FSDPRotaryEmbedding", "replicated"),
        ("FSDPFloat16Module", "replicated"),
    ):
        AutoMapping.register_module_type(name, ptype)
    logger.info("Registered FSDP-wrapped module types with AutoMapping")


def get_bridge():
    """Return the cached AutoBridge instance."""
    if _bridge_instance is None:
        raise RuntimeError("Call load_model() first")
    return _bridge_instance


def apply_lora_transform(model: torch.nn.Module) -> torch.nn.Module:
    """Apply the Megatron-Bridge LoRA transform to an already-loaded MCore model.

    Freezes the base (requires_grad=False everywhere) and injects trainable
    adapters on the target modules. The adapter params (requires_grad=True)
    are the ONLY trainable params — this is the canonical filter used for the
    optimizer, grad sync, logprob sync, and adapter checkpoints.

    NOTE: called AFTER load_model() and BEFORE any FSDP wrapping. The bridge
    PEFT stack wraps plain MCore modules; our lora mode deliberately skips
    FSDP (replicated base per rank, adapter-grad all_reduce) since bridge
    LoRA + MCore FSDP is unverified upstream.
    """
    global _lora_config

    from megatron_trainer.config import (
        LORA_ALPHA,
        LORA_DIM,
        LORA_DROPOUT,
        LORA_TARGET_MODULES,
    )
    from megatron.bridge.peft.lora import LoRA

    _lora_config = LoRA(
        target_modules=LORA_TARGET_MODULES,
        dim=LORA_DIM,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )
    model = _lora_config(model, training=True)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"LoRA applied: target_modules={LORA_TARGET_MODULES} dim={LORA_DIM} "
        f"alpha={LORA_ALPHA} dropout={LORA_DROPOUT} — "
        f"{n_trainable} trainable tensors, {n_params:,} params"
    )
    return model


def get_lora_config():
    """Return the cached LoRA config (only valid in TRAIN_MODE=lora)."""
    if _lora_config is None:
        raise RuntimeError("Call apply_lora_transform() first")
    return _lora_config


def export_hf_weights_iter(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (hf_name, tensor) pairs by converting Megatron weights to HF format.

    Handles QKV unfusing, gate+up unfusing, and parameter renaming.
    Tensors stay on their original device (GPU).

    NOTE: under FSDP this is a COLLECTIVE — every trainer rank must consume
    the iterator in lockstep (the bridge all-gathers sharded DTensors).
    """
    bridge = get_bridge()
    for hf_tuple in bridge.export_hf_weights(model, cpu=False):
        yield (hf_tuple.param_name, hf_tuple.weight)


def gather_raw_params_iter(model: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Yield full (unsharded) parameter tensors in raw model.parameters() order.

    Used for the logprob-server weight sync, which is order-indexed against
    the server's own model.parameters() iteration.

    COLLECTIVE under FSDP: every trainer rank must consume in lockstep
    (full_tensor() all-gathers on the FSDP device mesh). Under DDP this is a
    plain pass-through of param.data.
    """
    from torch.distributed.tensor import DTensor

    for param in model.parameters():
        if isinstance(param, DTensor):
            yield param.full_tensor()
        else:
            yield param.data


def gather_trainable_params_iter(model: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Yield unsharded tensors for trainable (adapter) params only.

    LoRA mode logprob sync: both sides hold the same frozen base + identical
    LoRA transform, so adapter params appear in the same order in
    model.parameters(). Server side applies the same requires_grad filter.
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param.data


def sync_adapter_grads(model: torch.nn.Module) -> None:
    """All-reduce flattened adapter grads across trainer ranks (LoRA mode).

    Replaces FSDP finish_grad_sync(): grads accumulate locally across micro
    steps, then one flat all_reduce averages them before the optimizer step.
    Adapter grads are tiny (~14 MB at dim=32) — a single NCCL call.
    """
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    flat = torch.cat([g.flatten() for g in grads]) if grads else torch.empty(0, device="cuda")
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    if not grads:
        return
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


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


def save_hf_checkpoint(model: torch.nn.Module, save_dir: str, tokenizer=None, rank: int = 0) -> None:
    """Export Megatron model to HuggingFace format for eval compatibility.

    Avoids bridge.save_hf_pretrained() which uses distributed barriers —
    the logprob server (rank 1) is in its command loop and can't participate.
    Instead, manually exports weights via export_hf_weights + safetensors.

    Under FSDP the export pass is collective — ALL ranks must call this.
    Only rank 0 performs file writes.
    """
    import json
    from safetensors.torch import save_file

    bridge = get_bridge()
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    weights = {}
    for hf_name, weight in export_hf_weights_iter(model):
        if rank == 0:
            weights[hf_name] = weight.contiguous().cpu()

    if rank == 0:
        save_file(weights, os.path.join(save_dir, "model.safetensors"))

        hf_config = getattr(bridge.hf_pretrained, "config", bridge.hf_pretrained)
        if hf_config is not None:
            hf_config.save_pretrained(save_dir)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
            # Fix tokenizer_config.json: the NeMo container's transformers saves
            # extra_special_tokens as a list, but host transformers expects a dict.
            tc_path = os.path.join(save_dir, "tokenizer_config.json")
            if os.path.exists(tc_path):
                with open(tc_path) as f:
                    tc = json.load(f)
                if isinstance(tc.get("extra_special_tokens"), list):
                    tc["extra_special_tokens"] = {}
                    with open(tc_path, "w") as f:
                        json.dump(tc, f, indent=2)

        logger.info(f"HF checkpoint saved: {save_dir} ({len(weights)} tensors)")


def _fix_fused_expert_gate_up_adapter_layout(save_dir: str) -> None:
    """Repair bridge's fused-expert LoRA export layout in place (gpt-oss MoE).

    Bridge's ``save_hf_adapter`` writes the MoE expert adapters with
    ``lora_A`` carrying the fused output dim (2 * intermediate for gate_up)
    and ``lora_B`` carrying the input dim (hidden), both transposed relative
    to the PEFT on-disk layout. This affects BOTH expert projections:
    ``experts.base_layer`` (fused gate_up, non-square → vLLM load 500) and
    ``experts`` (down_proj, square → shapes look fine but values are swapped).
    vLLM's MoE LoRA loader (``_stack_moe_lora_weights``) expects ``lora_A``
    ``(num_experts * rank, hidden)`` and ``lora_B``
    ``(2 * intermediate, num_experts * rank)``; the gate_up mismatch crashes
    ``POST /v1/load_lora_adapter`` with e.g.
    ``The size of tensor a (2880) must match the size of tensor b (5760)
    at non-singleton dimension 2`` (gpt-oss-20b: hidden=2880, intermediate=2880).

    Fix = swap + transpose the pair: ``new_A = old_B.T``, ``new_B = old_A.T``
    (the expert-block layout is preserved by the transpose). Detection is
    shape-based on gate_up (the fused out-side is always the larger dim), so
    already-correct exports (or a future bridge fix) pass through untouched;
    when gate_up is detected as swapped, the same transform is applied to the
    square down_proj pair on the same layer (shape-ambiguous there). Rank-0
    only, after the collective bridge export.
    """
    st_path = os.path.join(save_dir, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        return

    from safetensors.torch import load_file, save_file

    weights = load_file(st_path)
    changed = False

    def _swap_pair(a_key: str, b_key: str) -> None:
        nonlocal changed
        lora_a, lora_b = weights[a_key], weights[b_key]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            return
        weights[a_key] = lora_b.transpose(0, 1).contiguous()
        weights[b_key] = lora_a.transpose(0, 1).contiguous()
        changed = True
        logger.info(
            f"Fixed fused expert adapter layout: {a_key} "
            f"{tuple(lora_a.shape)}/{tuple(lora_b.shape)} -> "
            f"{tuple(weights[a_key].shape)}/{tuple(weights[b_key].shape)}"
        )

    for key in list(weights):
        if not key.endswith(".experts.base_layer.lora_A.weight"):
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in weights:
            continue
        lora_a, lora_b = weights[key], weights[b_key]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            continue
        # Buggy layout: A (E*r, 2*intermediate), B (hidden, E*r).
        # Correct layout: A (E*r, hidden), B (2*intermediate, E*r).
        # The fused out-side (2*intermediate) is always the larger dim, so
        # A carrying it signals the swap (model-agnostic, no config needed).
        if not (lora_a.shape[0] == lora_b.shape[1] and lora_a.shape[1] > lora_b.shape[0]):
            continue
        _swap_pair(key, b_key)
        # Same bug on the square down_proj pair (shapes can't reveal it).
        down_a = key.replace(".base_layer.lora_A", ".lora_A")
        down_b = down_a.replace("lora_A", "lora_B")
        if down_a in weights and down_b in weights:
            _swap_pair(down_a, down_b)
    if changed:
        save_file(weights, st_path)
        logger.info(f"Re-saved {save_dir}/adapter_model.safetensors with fixed expert layout")


def save_hf_adapter_checkpoint(model: torch.nn.Module, save_dir: str, rank: int = 0) -> None:
    """Export LoRA adapter weights as an HF PEFT directory (TRAIN_MODE=lora).

    Writes adapter_config.json + adapter_model.safetensors — the exact format
    vLLM's LoRA loader consumes. Collective: all ranks participate in the
    export (barrier inside save_hf_adapter), only rank 0 writes files.
    """
    from megatron_trainer.config import HF_MODEL_PATH

    bridge = get_bridge()
    lora_cfg = get_lora_config()
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    bridge.save_hf_adapter(
        model,
        save_dir,
        peft_config=lora_cfg,
        base_model_name_or_path=HF_MODEL_PATH,
        show_progress=False,
    )
    if rank == 0:
        _fix_fused_expert_gate_up_adapter_layout(save_dir)
        logger.info(f"LoRA adapter saved: {save_dir}")


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
