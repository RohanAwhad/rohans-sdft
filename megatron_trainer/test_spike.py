"""Minimal spike test: load Qwen3-8B via AutoBridge and run a forward pass.

Run inside NeMo container:
    podman run --rm --device nvidia.com/gpu=1 --ipc=host --network=host \
        -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -v /mnt/nvme0n1/rawhad/self_distillation/rohans-sdft-2:/workspace:z \
        -v ~/.cache/huggingface:/root/.cache/huggingface:z \
        -w /workspace \
        nvcr.io/nvidia/nemo:26.06 \
        python megatron_trainer/test_spike.py
"""

import os
import sys
import time

import torch

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29501")

DEVICE = torch.device("cuda:0")


def test_autobridge_load():
    """Test 1: Load model via AutoBridge without torch.distributed."""
    print("\n=== Test 1: AutoBridge model loading ===")
    from megatron.bridge import AutoBridge

    t0 = time.time()
    bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B")
    print(f"AutoBridge created in {time.time() - t0:.1f}s")
    print(f"  Supported: {bridge.can_handle('Qwen/Qwen3-8B')}")
    return bridge


def test_model_creation(bridge):
    """Test 2: Create Megatron model via load_model (handles PG + config)."""
    print("\n=== Test 2: Megatron model creation ===")

    sys.path.insert(0, "/workspace")
    from megatron_trainer.model_utils import load_model

    t0 = time.time()
    model = load_model("Qwen/Qwen3-8B")
    print(f"  Model created in {time.time() - t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  GPU memory: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB")

    return model


def test_forward(model):
    """Test 3: Forward pass with labels=None (get logits)."""
    print("\n=== Test 3: Forward pass ===")

    seq_len = 32
    input_ids = torch.randint(0, 1000, (1, seq_len), device=DEVICE)
    position_ids = torch.arange(seq_len, device=DEVICE, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        print(f"  Forward pass in {time.time() - t0:.3f}s")
        print(f"  Logits shape: {logits.shape}")  # expect (1, 32, vocab_size)
        print(f"  Logits dtype: {logits.dtype}")
        print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"  GPU memory after forward: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB")


def test_backward(model):
    """Test 4: Backward pass (training mode)."""
    print("\n=== Test 4: Backward pass ===")

    seq_len = 32
    input_ids = torch.randint(0, 1000, (1, seq_len), device=DEVICE)
    position_ids = torch.arange(seq_len, device=DEVICE, dtype=torch.long).unsqueeze(0)

    model.train()
    t0 = time.time()
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    loss = logits.sum()
    loss.backward()
    print(f"  Forward + backward in {time.time() - t0:.3f}s")
    print(f"  Loss: {loss.item():.4f}")

    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total}")


def test_weight_export(bridge, model):
    """Test 5: Export weights in HF format."""
    print("\n=== Test 5: HF weight export ===")

    count = 0
    for hf_tuple in bridge.export_hf_weights(model, cpu=False):
        if count < 5:
            print(f"  {hf_tuple.param_name}: {list(hf_tuple.weight.shape)} {hf_tuple.weight.dtype}")
        count += 1

    print(f"  Total HF-format tensors: {count}")


def test_model_structure(model):
    """Test 6: Inspect model structure."""
    print("\n=== Test 6: Model structure ===")

    for name, _ in model.named_children():
        print(f"  model.{name}")

    if hasattr(model, 'output_layer'):
        print(f"  output_layer type: {type(model.output_layer).__name__}")
    if hasattr(model, 'decoder'):
        print(f"  decoder type: {type(model.decoder).__name__}")
        print(f"  decoder layers: {len(model.decoder.layers)}")
    if hasattr(model, 'embedding'):
        print(f"  embedding type: {type(model.embedding).__name__}")


def main():
    print("=" * 60)
    print("SDFT Megatron Bridge Spike Test")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    bridge = test_autobridge_load()
    model = test_model_creation(bridge)
    test_model_structure(model)
    test_forward(model)
    test_backward(model)
    test_weight_export(bridge, model)

    print("\n=== ALL TESTS PASSED ===")

    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
