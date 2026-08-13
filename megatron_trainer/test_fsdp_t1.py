"""T1: Phase 1 FSDP verification (gpt-oss-20b, 4 trainer ranks).

Exercises the exact production code paths added for the FSDP backend:
    1. FSDP wrap + 1 training micro-step (no_sync + finish_grad_sync + AdamW)
    2. Collective HF export -> save_hf_checkpoint() on ALL ranks (no deadlock)
    3. Export tensor hashes == checkpoint tensor hashes (sha256 on sampled weights)
    4. gather_raw_params_iter() (logprob-sync pattern) full tensors ==
       checkpoint values for the first/last raw params (embed + lm_head)
    5. Peak GPU memory per rank

Exit code 0 = all checks passed.

Launch: bash megatron_trainer/test_fsdp_t1.sh
"""

import hashlib
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().to(torch.float32).numpy().tobytes()
    ).hexdigest()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    from loguru import logger

    from megatron_trainer.config import HF_MODEL_PATH
    from megatron_trainer.model_utils import (
        cleanup,
        export_hf_weights_iter,
        gather_raw_params_iter,
        load_model,
        register_fsdp_module_mappings,
        save_hf_checkpoint,
    )
    from megatron.core.distributed import (
        DistributedDataParallelConfig,
        TorchFullyShardedDataParallel,
    )

    ckpt_dir = f"/tmp/opencode/t1_ckpt_r{rank}"
    logger.info(f"=== T1 start rank={rank}/{world_size} model={HF_MODEL_PATH} ===")

    t0 = time.monotonic()
    model = load_model(HF_MODEL_PATH)
    model.train()
    logger.info(f"model loaded in {time.monotonic() - t0:.1f}s")

    unwrapped = model.module if hasattr(model, "module") else model
    vocab_size = unwrapped.vocab_size
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    register_fsdp_module_mappings()
    fsdp_model = TorchFullyShardedDataParallel(unwrapped.config, ddp_config, model)
    logger.info(f"FSDP wrap OK: {type(fsdp_model).__name__}")

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=5e-5)

    # ---- 1. Training micro-step (trainer FSDP pattern) ----
    torch.manual_seed(0)
    seq_len = 512
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    with fsdp_model.no_sync():
        logits = fsdp_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )
        loss = F.cross_entropy(logits[0, -1:], input_ids[0, -1:])
        (loss / 1.0).backward()
    fsdp_model.finish_grad_sync()
    clip_grad_norm_(fsdp_model.parameters(), 10.0)
    optimizer.step()
    logger.info(f"training step OK loss={loss.item():.4f}")

    # ---- 2. Collective checkpoint save (production path, ALL ranks) ----
    t0 = time.monotonic()
    save_hf_checkpoint(model, ckpt_dir, tokenizer=None, rank=rank)
    logger.info(f"collective save_hf_checkpoint OK in {time.monotonic() - t0:.1f}s")

    # ---- 3. Sampling export pass (collective; rank 0 hashes) ----
    export_hashes: dict[str, str] = {}
    export_count = 0
    last_name = None
    for i, (name, w) in enumerate(export_hf_weights_iter(model)):
        export_count += 1
        if rank == 0:
            if i % 100 == 0:
                export_hashes[name] = _sha(w)
            last_name, last_w = name, w
    if rank == 0:
        export_hashes[last_name] = _sha(last_w)
        logger.info(f"export pass OK: {export_count} tensors, {len(export_hashes)} hashed")

    # ---- 4. Raw-order gather pass (logprob-sync pattern, collective) ----
    raw_hashes: dict[int, str] = {}
    raw_count = 0
    last_full = None
    for i, full in enumerate(gather_raw_params_iter(model)):
        raw_count += 1
        if rank == 0:
            if i % 100 == 0:
                raw_hashes[i] = _sha(full)
            last_full = full
    if rank == 0:
        raw_hashes[raw_count - 1] = _sha(last_full)
        logger.info(f"raw-order gather OK: {raw_count} params, {len(raw_hashes)} hashed")

    # ---- 5. Verify: export hashes vs checkpoint; raw-order vs checkpoint ----
    failures: list[str] = []
    if rank == 0:
        from safetensors.torch import load_file

        ck = load_file(f"{ckpt_dir}/model.safetensors")
        for name, h in export_hashes.items():
            if name not in ck:
                failures.append(f"checkpoint missing tensor {name}")
            elif h != _sha(ck[name]):
                failures.append(f"export/checkpoint hash mismatch: {name}")

        # raw-order index 0 = first param (embed_tokens), last = lm_head
        if raw_hashes.get(0) != _sha(ck["model.embed_tokens.weight"]):
            failures.append("raw-order idx 0 != checkpoint model.embed_tokens.weight")
        if raw_hashes.get(raw_count - 1) != _sha(ck["lm_head.weight"]):
            failures.append("raw-order last != checkpoint lm_head.weight")

        if failures:
            logger.error("T1 FAILURES:\n" + "\n".join(failures))
        else:
            logger.info(
                f"ALL HASH CHECKS PASSED (export={len(export_hashes)}, "
                f"raw={len(raw_hashes)}, ckpt={len(ck)})"
            )

    # ---- 6. Peak memory ----
    peak = torch.tensor([torch.cuda.max_memory_allocated(device)], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        logger.info(f"GLOBAL peak memory: {peak.item() / 1e9:.2f} GB")

    dist.barrier()
    if rank == 0:
        n_fail = len(failures)
    else:
        n_fail = 0
    dist.broadcast_object_list([n_fail], src=0)
    cleanup()
    logger.info(f"=== T1 {'PASSED' if n_fail == 0 else f'FAILED ({n_fail})'} ===")
    raise SystemExit(n_fail)


if __name__ == "__main__":
    main()
