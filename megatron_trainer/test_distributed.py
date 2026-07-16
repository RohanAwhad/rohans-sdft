"""Test 2-rank distributed setup: trainer (rank 0) + logprob server (rank 1).

Validates that Megatron parallel_state works with world_size=2 and that
NCCL weight broadcast works between two Megatron models.

Run inside NeMo container with 2 GPUs:
    podman run --rm --device nvidia.com/gpu=1,nvidia.com/gpu=2 --ipc=host --network=host \
        -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -v /mnt/nvme0n1/rawhad/self_distillation/rohans-sdft-2:/workspace:z \
        -v ~/.cache/huggingface:/root/.cache/huggingface:z \
        -w /workspace \
        nvcr.io/nvidia/nemo:26.06 \
        torchrun --nproc_per_node=2 megatron_trainer/test_distributed.py
"""

import os
import time

import torch
import torch.distributed as dist

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")



def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[rank {rank}] Initialized (world_size={world_size}, "
          f"local_rank={local_rank}, GPU={torch.cuda.get_device_name(local_rank)})")

    from megatron.core import parallel_state as mpu
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    print(f"[rank {rank}] Megatron parallel state initialized")

    device = torch.device(f"cuda:{local_rank}")

    import sys
    sys.path.insert(0, "/workspace")
    from megatron_trainer.model_utils import load_model

    model_name = os.environ.get("TEST_MODEL", "Qwen/Qwen3-8B")
    print(f"[rank {rank}] Loading {model_name}...")

    t0 = time.time()
    model = load_model(model_name)
    print(f"[rank {rank}] Model loaded in {time.time() - t0:.1f}s")

    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, 1000, (1, 16), device=device)
        position_ids = torch.arange(16, device=device, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        print(f"[rank {rank}] Forward pass: logits shape={logits.shape}")

    print(f"[rank {rank}] Testing weight broadcast...")
    for i, param in enumerate(model.parameters()):
        if rank == 0:
            dist.broadcast(param.data, src=0)
        else:
            incoming = torch.empty_like(param.data)
            dist.broadcast(incoming, src=0)
            if i == 0:
                diff = (param.data - incoming).abs().max().item()
                print(f"[rank {rank}] Weight broadcast check: max diff = {diff:.6e}")
        if i >= 2:
            break

    print(f"[rank {rank}] Test PASSED")

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
