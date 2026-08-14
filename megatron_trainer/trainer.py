"""SDFT trainer — Megatron Bridge version with MCore FSDP.

Orchestrates:
    1. Env rollout (rank 0 only, broadcast to all ranks)
       - ENV_TYPE=rag: RagEnv (vLLM generation + optional reflector)
       - ENV_TYPE=api_adapter: ApiAdapterEnv (multi-turn adapter loop)
    2. Student forward pass (Megatron-Core GPTModel via MCore FSDP)
    3. Teacher log-probs (TCP from logprob server, each rank independently)
    4. Chunked reverse KL loss + backward (via MCore output_processor hook,
       with gradient accumulation + no_sync)
    5. Step-level weight sync to both servers (all ranks — FSDP
       export/gather passes are collectives)

With ASYNC_ROLLOUT=1 the rollout moves to a rank-0 producer thread feeding a
bounded queue; the main path consumes one microbatch (world_size samples) at
a time. ASYNC_IN_ORDER=1 keeps the deterministic batch-in-order producer
(Layer 1 plumbing-equivalence mode). See
docs/megatron_trainer/async_rollouts.md.

Launch: torchrun --nproc_per_node=N -m megatron_trainer.trainer
"""

import concurrent.futures
import hashlib
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch
import torch.distributed as dist
from datasets import load_dataset
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import wandb
from megatron_trainer.collator import SDFTCollator
from megatron_trainer.chunked_head import make_kl_processor
from megatron_trainer.config import (
    ASYNC_IN_ORDER,
    ASYNC_ROLLOUT,
    BATCH_SIZE,
    DEBUG_ROLLOUT_HASH,
    EMA_ALPHA,
    ENV_TYPE,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    HF_MODEL_PATH,
    HINDSIGHT_FIELD,
    IS_CAP,
    IS_WEIGHTING,
    LEARNING_RATE,
    LR_SCHEDULER,
    MAX_GRAD_NORM,
    MAX_TOTAL_LEN,
    MODEL_NAME,
    N_ASYNC,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SAVE_EVERY,
    STUDENT_MAX_PROMPT_LEN,
    STUDENT_THINKING,
    TEACHER_MAX_PROMPT_LEN,
    TEACHER_MODEL_PATH,
    THINKING_BUDGET,
    TRAIN_DATA_PATH,
    TRAINER_SEED,
    VLLM_BASE_URL,
    VLLM_BASE_URLS,
    WANDB_PROJECT,
    WANDB_ENTITY,
    WANDB_NAME,
)
from megatron_trainer.env import ApiAdapterEnv, RagEnv
from megatron_trainer.model_utils import (
    cleanup,
    init_distributed_trainer,
    load_model,
    register_fsdp_module_mappings,
    save_hf_checkpoint,
)
from megatron_trainer.logprob_client import (
    init_logprob_weight_engine,
    request_teacher_log_probs_tcp,
    sync_weights_to_logprob_server,
    wait_for_logprob_server,
)
from megatron_trainer.vllm_utils import (
    init_vllm_weight_engine,
    sync_weights_to_vllm,
    wait_for_vllm,
)

# Completed optimizer steps — written by the main thread, read by the rollout
# producer to version-stamp generations (policy_version).
_OPTIMIZER_STEP: int = 0

# End-of-data signal: the producer pushes this as its very last queue item.
# The done signal travels through the queue itself — immune to the
# check-flag-then-block race a separate Event has (producer can set the event
# while the consumer is already blocked in get()).
_ROLLOUT_SENTINEL = object()

# ---------------------------------------------------------------------------
# Rollout helpers (rank 0; produce() is shared by the sync and in-order
# async paths, keeping per-sample stats identical between modes)
# ---------------------------------------------------------------------------


def _build_env(item: dict, success_cache: dict[str, str], tokenizer, vllm_idx: int):
    """Build a rollout env for one dataset item (vLLM round-robins instances)."""
    url = VLLM_BASE_URLS[vllm_idx % len(VLLM_BASE_URLS)]
    if ENV_TYPE == "rag":
        use_reflector = HINDSIGHT_FIELD == "online_feedback"
        return RagEnv(
            prompt_text=item["prompt_texts"][0],
            vllm_base_url=url,
            privileged_information_prompt=item["conditional_texts"][0],
            raw_question=item["raw_questions"][0],
            golden_answer=item["golden_answers"][0],
            normalized_messages=item["normalized_messages"][0],
            tokenizer=tokenizer,
            use_reflector=use_reflector,
            golden_chunk=item["golden_chunks"][0],
        )
    return ApiAdapterEnv(
        prompt_text=item["prompt_texts"][0],
        vllm_base_url=url,
        raw_question=item["raw_questions"][0],
        golden_answer=item["golden_answers"][0],
        tokenizer=tokenizer,
        success_cache=success_cache,
    )


def _sample_meta(env) -> dict:
    """Per-sample rollout metadata for step aggregation and wandb logging."""
    if ENV_TYPE == "api_adapter":
        if env.episode_result is not None:
            pass_value = 1.0 if env.episode_result else 0.0
        else:
            pass_value = None
        return {
            "pass_value": pass_value,
            "reflector_present": False,
            "fallback": False,
            "table": {
                "adapter_history": env.adapter_history,
                "raw_question": env.raw_question,
                "golden_answer": env.golden_answer,
                "verdict": env.verdict,
            },
        }
    reflector_present = env.reflector_result is not None
    if reflector_present:
        pass_value = 1.0 if env.reflector_result["verdict"] == "PASS" else 0.0
    else:
        pass_value = None
    return {
        "pass_value": pass_value,
        "reflector_present": reflector_present,
        "fallback": env.use_reflector and env.reflector_result is None,
        "table": None,
    }


def _aggregate_pass_rate(metas: list[dict]) -> float | None:
    """Batch pass rate from per-sample metas (same math as the sync path)."""
    if ENV_TYPE == "api_adapter":
        vals = [m["pass_value"] for m in metas if m["pass_value"] is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)
    if not metas or not metas[0]["reflector_present"]:
        return None
    vals = [m["pass_value"] for m in metas if m["reflector_present"]]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _log_rollout_hash(batch_id: int, idx: int, env) -> None:
    if not DEBUG_ROLLOUT_HASH:
        return
    h = hashlib.sha256(
        (env.prompt_text + "\x00" + (env.completion_text or "")).encode("utf-8")
    ).hexdigest()[:12]
    logger.info(
        f"ROLLOUT_HASH batch={batch_id} idx={idx} hash={h} "
        f"plen={len(env.prompt_text or '')} clen={len(env.completion_text or '')}"
    )


def _log_consume_hash(step: int, rank: int, micro: int, item_data: dict) -> None:
    if not DEBUG_ROLLOUT_HASH:
        return
    h = hashlib.sha256(
        (item_data["prompt_text"] + "\x00" + (item_data["completion_text"] or "")).encode("utf-8")
    ).hexdigest()[:12]
    logger.info(f"CONSUME_HASH step={step} rank={rank} micro={micro} hash={h}")


def produce(items: list[dict], success_cache: dict[str, str], policy_version: int, tokenizer):
    """Rank-0 rollout for one batch of items.

    Builds envs, runs them, updates success_cache (api_adapter), and returns
    (rollout_data, metas, batch_meta). rollout_data is the broadcast payload;
    metas/batch_meta feed the wandb logging.
    """
    envs = [_build_env(item, success_cache, tokenizer, i) for i, item in enumerate(items)]

    with ThreadPoolExecutor(max_workers=min(32, len(envs))) as executor:
        list(executor.map(lambda e: e.run(), envs))

    for i, env in enumerate(envs):
        _log_rollout_hash(policy_version, i, env)

    if ENV_TYPE == "api_adapter":
        # Cache successful adapter responses
        for env in envs:
            if env.episode_result and env.completion_text:
                parsed_verdict, parsed_feedback = env.parse_adapter_response(env.completion_text)
                if parsed_verdict:
                    cached_text = f"Verdict: {parsed_verdict}\nFeedback: {parsed_feedback}"
                    success_cache[env.raw_question] = cached_text

    rollout_data = [
        {
            "prompt_text": env.prompt_text,
            "completion_text": env.completion_text,
            "completion_log_probs": env.completion_log_probs,
            "privileged_information_prompt": env.privileged_information_prompt,
            "policy_version": policy_version,
        }
        for env in envs
    ]
    metas = [_sample_meta(env) for env in envs]
    batch_meta = {
        "full_pass_rate": _aggregate_pass_rate(metas),
        "fallback_delta": sum(m["fallback"] for m in metas),
        "table": metas[-1]["table"],
    }
    return rollout_data, metas, batch_meta


def _push_sample(rollout_queue: queue.Queue, env, policy_version: int, success_cache: dict[str, str], sample_idx: int = 0) -> None:
    """Post-process one completed env and push its payload (with _meta) to the queue."""
    _log_rollout_hash(policy_version, sample_idx, env)
    if ENV_TYPE == "api_adapter":
        if env.episode_result and env.completion_text:
            parsed_verdict, parsed_feedback = env.parse_adapter_response(env.completion_text)
            if parsed_verdict:
                cached_text = f"Verdict: {parsed_verdict}\nFeedback: {parsed_feedback}"
                success_cache[env.raw_question] = cached_text
    payload = {
        "prompt_text": env.prompt_text,
        "completion_text": env.completion_text,
        "completion_log_probs": env.completion_log_probs,
        "privileged_information_prompt": env.privileged_information_prompt,
        "policy_version": policy_version,
        "_meta": _sample_meta(env),
    }
    rollout_queue.put(payload)


def _produce_in_order(
    rollout_queue: queue.Queue,
    data_iter,
    success_cache: dict[str, str],
    step_done_q: queue.Queue,
    gen_times: deque,
    tokenizer,
    world_size: int,
) -> None:
    """In-order producer (ASYNC_IN_ORDER=1): full GRAD_ACCUM_STEPS batches in
    dataset order, column-major pushes, no overlap with training — the
    plumbing equivalent of the sync path (Layer 1 mode).

    step_done_q is a 1:1 signal queue: the main path puts one token per
    completed optimizer step; the producer consumes one per batch, so the
    next batch's generation never starts before the previous step finished
    (a persistent Event would accumulate stale sets and allow overlap).
    Ends by pushing _ROLLOUT_SENTINEL (no Event-based done flag).
    """
    local_accum = GRAD_ACCUM_STEPS // world_size
    sentinel = object()
    while True:
        items = []
        for _ in range(GRAD_ACCUM_STEPS):
            item = next(data_iter, sentinel)
            if item is sentinel:
                break
            items.append(item)
        if len(items) < GRAD_ACCUM_STEPS:
            rollout_queue.put(_ROLLOUT_SENTINEL)
            return
        t0 = time.monotonic()
        rollout_data, metas, _ = produce(items, success_cache, _OPTIMIZER_STEP, tokenizer)
        gen_times.append(time.monotonic() - t0)
        for k in range(local_accum):
            for r in range(world_size):
                s = r * local_accum + k
                rollout_data[s]["_meta"] = metas[s]
                rollout_queue.put(rollout_data[s])
        step_done_q.get()


def _produce_streaming(
    rollout_queue: queue.Queue,
    data_iter,
    success_cache: dict[str, str],
    gen_times: deque,
    tokenizer,
    steps_per_epoch: int,
) -> None:
    """Streaming producer (ASYNC_IN_ORDER=0): N_ASYNC envs in flight,
    per-sample push on completion — completion order, not dataset order
    (the reordering is the feature). Submits exactly steps_per_epoch *
    GRAD_ACCUM_STEPS samples (the same count the sync path trains), drains
    in-flight work, then pushes the end-of-data sentinel."""
    executor = ThreadPoolExecutor(max_workers=N_ASYNC)
    window: dict = {}
    sentinel = object()
    submit_limit = steps_per_epoch * GRAD_ACCUM_STEPS
    submitted = 0
    while submitted < submit_limit:
        item = next(data_iter, sentinel)
        if item is sentinel:
            break
        env = _build_env(item, success_cache, tokenizer, submitted)
        t_start = time.monotonic()
        fut = executor.submit(env.run)
        window[fut] = (env, t_start, _OPTIMIZER_STEP)
        submitted += 1
        while len(window) >= N_ASYNC:
            done, _ = concurrent.futures.wait(
                window.keys(), return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                env, t_start, version = window.pop(fut)
                fut.result()
                _push_sample(rollout_queue, env, version, success_cache, submitted - len(window) - 1)
                gen_times.append(time.monotonic() - t_start)
    for fut in concurrent.futures.as_completed(window):
        env, t_start, version = window[fut]
        fut.result()
        _push_sample(rollout_queue, env, version, success_cache, submit_limit)
        gen_times.append(time.monotonic() - t_start)
    rollout_queue.put(_ROLLOUT_SENTINEL)


# ---------------------------------------------------------------------------
# Training helpers (shared by sync and async step loops)
# ---------------------------------------------------------------------------


def _train_sample(
    item_data: dict,
    micro_step: int,
    local_accum_steps: int,
    fsdp_model,
    model,
    tokenizer,
    vocab_size: int,
    device: torch.device,
) -> dict | None:
    """One sample: teacher log-probs → student forward → reverse-KL backward.

    Returns per-sample results (None when the completion is empty — skipped).
    """
    completion_ids: list[int] = tokenizer.encode(
        item_data["completion_text"], add_special_tokens=False,
    )
    if len(completion_ids) == 0:
        logger.warning(f"Empty completion, skipping micro_step {micro_step}")
        return None
    completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

    # Rollout (vLLM) log-probs for importance sampling; NaN marks
    # never-sampled tokens (excluded from the IS weight).
    rollout_log_probs = None
    if IS_WEIGHTING:
        lp_list = item_data.get("completion_log_probs")
        if lp_list:
            lp_list = lp_list[: len(completion_ids)]
            rollout_log_probs = torch.tensor(
                [float("nan") if v is None else float(v) for v in lp_list],
                dtype=torch.float32,
                device=device,
            )
            if rollout_log_probs.size(0) != len(completion_ids):
                logger.warning(
                    f"logprobs length {rollout_log_probs.size(0)} != "
                    f"completion length {len(completion_ids)}; masking trailing"
                )
                pad = torch.full(
                    (len(completion_ids) - rollout_log_probs.size(0),),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                )
                rollout_log_probs = torch.cat([rollout_log_probs, pad])

    # Teacher log-probs via TCP (each rank independently)
    t0 = time.monotonic()
    cond_ids: list[int] = tokenizer.encode(
        item_data["privileged_information_prompt"],
        add_special_tokens=False, truncation=True, max_length=TEACHER_MAX_PROMPT_LEN,
    )
    teacher_log_probs = request_teacher_log_probs_tcp(
        token_ids=cond_ids + completion_ids,
        prompt_len=len(cond_ids),
        vocab_size=vocab_size,
        device=device,
    )  # (C, V)
    t_teacher = time.monotonic() - t0

    # Student forward + chunked reverse-KL loss via MCore
    # output_processor hook (head GEMM on local vocab shard only)
    t0 = time.monotonic()
    prompt_enc = tokenizer(
        item_data["prompt_text"],
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=STUDENT_MAX_PROMPT_LEN,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"][0]
    prompt_len = prompt_ids.size(0)
    comp_ids_t = torch.tensor(completion_ids, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)
    position_ids = torch.arange(
        input_ids.size(1), device=device, dtype=torch.long
    ).unsqueeze(0)

    kl_processor = make_kl_processor(
        prompt_len=prompt_len,
        completion_ids=completion_ids,
        teacher_log_probs=teacher_log_probs,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        rollout_log_probs=rollout_log_probs,
        is_weighting=IS_WEIGHTING,
        is_cap=IS_CAP,
    )
    loss, step_metrics = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        output_processor=kl_processor,
    )
    t_student = time.monotonic() - t0

    # Reverse-KL loss backward
    t0 = time.monotonic()
    # no_sync on non-final micro-steps (skip allreduce)
    is_final = (micro_step == local_accum_steps - 1)
    ctx = nullcontext() if is_final else fsdp_model.no_sync()
    with ctx:
        scaled_loss = loss / local_accum_steps
        scaled_loss.backward()
    t_loss_bwd = time.monotonic() - t0

    return {
        "loss": loss.item(),
        "comp_len": len(completion_ids),
        "metrics": step_metrics,
        "t_teacher": t_teacher,
        "t_student": t_student,
        "t_loss_bwd": t_loss_bwd,
    }


def _pull_microbatch(
    rank: int,
    world_size: int,
    rollout_queue: queue.Queue,
    device: torch.device,
) -> tuple[list[dict], bool]:
    """Collective microbatch pull: rank 0 pops world_size samples (blocking),
    broadcasts them. Returns (microbatch, epoch_done) on all ranks. The
    producer's end-of-data sentinel is the only epoch-done signal."""
    if rank == 0:
        first = rollout_queue.get()
        if first is _ROLLOUT_SENTINEL:
            microbatch = [None] * world_size
            epoch_done = True
        else:
            microbatch = [first] + [rollout_queue.get() for _ in range(world_size - 1)]
            epoch_done = False
    else:
        microbatch = [None] * world_size
        epoch_done = False
    done_tensor = torch.tensor([1 if epoch_done else 0], dtype=torch.int, device=device)
    dist.broadcast(done_tensor, src=0)
    if done_tensor.item():
        return microbatch, True
    dist.broadcast_object_list(microbatch, src=0)
    return microbatch, False


def _drain_gen_times(gen_times: deque) -> float:
    total = 0.0
    while gen_times:
        total += gen_times.popleft()
    return total


def _step_tail(
    optimizer,
    scheduler,
    fsdp_model,
    model,
    logprob_comm,
    vllm_group,
    tokenizer,
    *,
    rank: int,
    device: torch.device,
    epoch: int,
    optimizer_step: int,
    accum_loss_sum: float,
    accum_samples: int,
    accum_comp_len_sum: int,
    accum_metrics: dict,
    full_pass_rate: float | None,
    table_meta: dict | None,
    policy_lags: list[int],
    t_teacher_sum: float,
    t_student_sum: float,
    t_loss_bwd_sum: float,
    t_step_start: float,
    t_generation: float,
    t_producer_wait: float,
) -> int:
    """Optimizer step → aggregate → rank-0 wandb log → weight sync → barrier.

    Collective across all ranks. Returns the new optimizer_step.
    """
    global _OPTIMIZER_STEP

    t0 = time.monotonic()
    fsdp_model.finish_grad_sync()
    grad_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer_step += 1
    _OPTIMIZER_STEP = optimizer_step
    t_optimizer = time.monotonic() - t0

    # ---- Aggregate loss + grad norm across ranks ----
    t0 = time.monotonic()
    agg_tensor = torch.tensor(
        [accum_loss_sum, float(accum_samples), grad_norm.item() ** 2], device=device,
    )
    dist.all_reduce(agg_tensor)
    t_allreduce = time.monotonic() - t0

    t_weight_sync: float = 0.0
    if rank == 0:
        total_loss = agg_tensor[0].item()
        total_samples = max(agg_tensor[1].item(), 1)
        avg_loss = total_loss / total_samples
        global_grad_norm = agg_tensor[2].item() ** 0.5
        avg_comp_len = accum_comp_len_sum / max(accum_samples, 1)

        log_dict: dict = {
            "train/loss": avg_loss,
            "train/completion_length": avg_comp_len,
            "train/grad_norm": global_grad_norm,
            "train/epoch": epoch,
            "train/lr": scheduler.get_last_lr()[0] if scheduler is not None else LEARNING_RATE,
        }
        if full_pass_rate is not None:
            key = "reflector/pass_rate" if ENV_TYPE == "rag" else "episode/pass_rate"
            log_dict[key] = full_pass_rate
        for k, vals in accum_metrics.items():
            log_dict[k] = sum(vals) / len(vals)
        if policy_lags:
            log_dict["async/policy_lag_mean"] = sum(policy_lags) / len(policy_lags)
            log_dict["async/policy_lag_max"] = max(policy_lags)

        if optimizer_step % 10 == 0 and table_meta is not None:
            table = wandb.Table(
                columns=["step", "question", "golden_answer", "num_turns", "verdict", "conversation"],
            )
            conversation = "\n".join(str(msg) for msg in table_meta["adapter_history"])
            table.add_data(
                optimizer_step, table_meta["raw_question"], table_meta["golden_answer"],
                len(table_meta["adapter_history"]), table_meta["verdict"], conversation,
            )
            log_dict["episode/sample"] = table

        wandb.log(log_dict, step=optimizer_step)
        metrics_str = " ".join(f"{k}={sum(vals) / len(vals):.4f}" for k, vals in sorted(accum_metrics.items()))
        lag_str = ""
        if policy_lags:
            lag_str = f" lag_mean={sum(policy_lags) / len(policy_lags):.1f} lag_max={max(policy_lags)}"
        logger.info(
            f"opt_step={optimizer_step} loss={avg_loss:.4f} comp_len={avg_comp_len:.0f} "
            f"grad_norm={global_grad_norm:.4f}{lag_str} {metrics_str}".rstrip()
        )

    # ---- Sync weights + checkpoint (all ranks — FSDP export/gather
    #      passes are collectives) ----
    t0 = time.monotonic()
    if not TEACHER_MODEL_PATH:
        sync_weights_to_logprob_server(model, logprob_comm, rank=rank)
    sync_weights_to_vllm(model, device, vllm_group, rank=rank)
    t_weight_sync = time.monotonic() - t0

    if optimizer_step % SAVE_EVERY == 0:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
        save_hf_checkpoint(model, ckpt_dir, tokenizer, rank=rank)

    # All ranks wait for rank 0 weight sync before next step
    t0 = time.monotonic()
    dist.barrier()
    t_barrier = time.monotonic() - t0

    t_total = time.monotonic() - t_step_start
    if rank == 0:
        timing = (
            f"TIMING step={optimizer_step} | total={t_total:.1f}s gen={t_generation:.1f}s "
            f"teacher={t_teacher_sum:.1f}s student={t_student_sum:.1f}s "
            f"loss_bwd={t_loss_bwd_sum:.1f}s optim={t_optimizer:.1f}s wsync={t_weight_sync:.1f}s"
        )
        if ASYNC_ROLLOUT:
            timing += f" producer_wait={t_producer_wait:.1f}s gen_overlap={t_generation - t_producer_wait:.1f}s"
            if policy_lags:
                timing += f" lag_mean={sum(policy_lags) / len(policy_lags):.1f} lag_max={max(policy_lags)}"
        logger.info(timing)
    return optimizer_step


def _crash_hard_on_thread_error(args: threading.ExceptHookArgs) -> None:
    logger.error(f"Thread {args.thread.name} crashed with {args.exc_type.__name__}: {args.exc_value}")
    print(
        f"FATAL: thread {args.thread.name} crashed: {args.exc_value}",
        file=sys.stderr,
        flush=True,
    )
    os._exit(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add("logs/trainer.log", level=log_level)

    logger.info("=== SDFT Megatron Trainer Starting ===")

    # ---- Initialize torch.distributed via torchrun ----
    local_rank = init_distributed_trainer()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    assert GRAD_ACCUM_STEPS % world_size == 0, (
        f"GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}) must be divisible by "
        f"num_trainers ({world_size})"
    )
    local_accum_steps = GRAD_ACCUM_STEPS // world_size

    logger.info(f"FSDP: rank={rank}/{world_size}, local_rank={local_rank}, "
                f"local_accum_steps={local_accum_steps}")

    # ---- Model + tokenizer ----
    # Two tokenizer instances: the Rust tokenizers library is NOT thread-safe
    # (overlapping mutating calls raise "Already borrowed"). In async mode the
    # producer thread builds envs + runs the collator, while the main thread
    # encodes in _train_sample — so the producer side gets its own instance.
    logger.info(f"Loading model: {HF_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    producer_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    model = load_model(HF_MODEL_PATH)
    model.train()

    # Use padded vocab size from model (Megatron pads for TP alignment)
    unwrapped = model.module if hasattr(model, 'module') else model
    vocab_size = unwrapped.vocab_size
    logger.info(
        f"Model loaded. vocab_size={vocab_size} (tokenizer={tokenizer.vocab_size}) "
        f"GPU mem after load: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
    )

    # ---- MCore FSDP wrapping ----
    from megatron.core.distributed import (
        DistributedDataParallelConfig,
        TorchFullyShardedDataParallel,
    )

    register_fsdp_module_mappings()
    fsdp_model = TorchFullyShardedDataParallel(
        unwrapped.config,
        DistributedDataParallelConfig(use_distributed_optimizer=False),
        model,
    )
    logger.info("MCore FSDP wrapping complete.")
    # The Megatron bridge closes stray file descriptors during model load,
    # silently killing the loguru file sink opened at boot (writes fail after
    # "Loading model"). Re-open it so logs/trainer.log covers the training loop.
    logger.add("logs/trainer.log", level=log_level)

    # ---- Optimizer (FSDP: torch AdamW) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.01)
    logger.info(f"torch AdamW optimizer ready. LR={LEARNING_RATE}")

    # ---- Dataset (all ranks load, only rank 0 iterates) ----
    logger.info(f"Loading dataset: {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    collator = SDFTCollator(tokenizer=producer_tokenizer, hindsight_field=HINDSIGHT_FIELD)
    # Drop examples whose protected set (system + hint) exceeds the budgets —
    # logged as warnings, never trained on. Deterministic across ranks.
    dataset = collator.filter_dataset(dataset, rank=rank)
    if TRAINER_SEED is not None:
        torch.manual_seed(TRAINER_SEED)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, drop_last=True,
    )
    steps_per_epoch = len(dataset) // GRAD_ACCUM_STEPS
    logger.info(f"Dataset: {len(dataset)} examples, {steps_per_epoch} steps/epoch")

    # ---- LR scheduler (total steps known only after dataset load) ----
    total_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = 0
    scheduler = None
    if LR_SCHEDULER == "cosine":
        warmup_steps = min(int(0.1 * total_train_steps), 100)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )
        logger.info(
            f"LR scheduler: cosine, warmup={warmup_steps} over {total_train_steps} "
            f"optimizer steps"
        )
    else:
        logger.info("LR scheduler: constant")

    # ---- Rank 0 only: wandb, vLLM, logprob server ----
    vllm_group = None
    logprob_comm = None
    success_cache: dict[str, str] = {}

    if rank == 0:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_NAME or f"sdft-megatron-{MODEL_NAME.split('/')[-1]}-e{NUM_EPOCHS}",
            config={
                "model": MODEL_NAME,
                "teacher_model": TEACHER_MODEL_PATH,
                "env_type": ENV_TYPE,
                "backend": "fsdp",
                "learning_rate": LEARNING_RATE,
                "lr_scheduler": LR_SCHEDULER,
                "warmup_steps": warmup_steps,
                "max_grad_norm": MAX_GRAD_NORM,
                "ema_alpha": EMA_ALPHA,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
                "num_trainers": world_size,
                "local_accum_steps": local_accum_steps,
                "num_epochs": NUM_EPOCHS,
                "max_total_len": MAX_TOTAL_LEN,
                "student_max_prompt_len": STUDENT_MAX_PROMPT_LEN,
                "teacher_max_prompt_len": TEACHER_MAX_PROMPT_LEN,
                "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
                "student_thinking": STUDENT_THINKING,
                "thinking_budget": THINKING_BUDGET,
                "is_weighting": IS_WEIGHTING,
                "is_cap": IS_CAP,
                "async_rollout": ASYNC_ROLLOUT,
                "async_in_order": ASYNC_IN_ORDER,
                "n_async": N_ASYNC,
                "loss": "reverse_kl",
                "dataset": TRAIN_DATA_PATH,
                "hindsight_field": HINDSIGHT_FIELD,
            },
        )

        logger.info("Waiting for vLLM server...")
        wait_for_vllm()
        logger.info("Initializing vLLM weight transfer engine...")
        vllm_group = init_vllm_weight_engine(device)
        logger.info("vLLM weight engine ready.")

        logger.info("Waiting for logprob server...")
        wait_for_logprob_server()
        # External frozen teacher (TEACHER_MODEL_PATH set): no NCCL weight sync —
        # the teacher keeps its downloaded weights.
        if not TEACHER_MODEL_PATH:
            logger.info("Initializing logprob weight transfer engine...")
            logprob_comm = init_logprob_weight_engine(device)
            logger.info("Logprob weight engine ready.")

    # Barrier: all ranks wait for rank 0 to finish setup
    dist.barrier()

    # ---- Training loop ----
    global _OPTIMIZER_STEP
    _OPTIMIZER_STEP = 0
    optimizer_step = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        epoch_loss_sum: float = 0.0
        epoch_samples: int = 0
        reflector_fallback_count: int = 0

        data_iter = iter(dataloader) if rank == 0 else None

        rollout_queue: queue.Queue | None = None
        step_done_q: queue.Queue | None = None
        gen_times: deque | None = None

        if ASYNC_ROLLOUT and rank == 0:
            threading.excepthook = _crash_hard_on_thread_error
            rollout_queue = queue.Queue(maxsize=N_ASYNC + world_size + 1)
            step_done_q = queue.Queue(maxsize=1)
            gen_times = deque()
            if ASYNC_IN_ORDER:
                target = _produce_in_order
                args = (rollout_queue, data_iter, success_cache, step_done_q, gen_times, producer_tokenizer, world_size)
            else:
                target = _produce_streaming
                args = (rollout_queue, data_iter, success_cache, gen_times, producer_tokenizer, steps_per_epoch)
            producer_thread = threading.Thread(
                target=target,
                args=args,
                name="rollout-producer",
                daemon=False,
            )
            producer_thread.start()

        if ASYNC_ROLLOUT:
            # Streaming consumer: pull microbatches until the epoch drains
            # (producer exhausted the dataset + queue empty).
            while True:
                t_step_start = time.monotonic()
                t_producer_wait: float = 0.0
                t_generation: float = 0.0
                step_metas: list[dict] = []
                policy_lags: list[int] = []

                optimizer.zero_grad()
                accum_loss_sum: float = 0.0
                accum_samples: int = 0
                accum_comp_len_sum: int = 0
                accum_metrics: dict[str, list[float]] = {}
                t_teacher_sum: float = 0.0
                t_student_sum: float = 0.0
                t_loss_bwd_sum: float = 0.0

                epoch_done = False
                for micro in range(local_accum_steps):
                    t0 = time.monotonic()
                    microbatch, done = _pull_microbatch(
                        rank, world_size, rollout_queue, device,
                    )
                    if done:
                        epoch_done = True
                        break
                    t_producer_wait += time.monotonic() - t0
                    if rank == 0:
                        t_generation += _drain_gen_times(gen_times)
                    item_data = microbatch[rank]
                    if rank == 0:
                        step_metas.append(item_data["_meta"])
                        policy_lags.append(_OPTIMIZER_STEP - item_data["policy_version"])
                    _log_consume_hash(_OPTIMIZER_STEP, rank, micro, item_data)
                    result = _train_sample(
                        item_data, micro, local_accum_steps, fsdp_model, model,
                        tokenizer, vocab_size, device,
                    )
                    if result is None:
                        continue
                    accum_loss_sum += result["loss"]
                    accum_samples += 1
                    accum_comp_len_sum += result["comp_len"]
                    epoch_loss_sum += result["loss"]
                    epoch_samples += 1
                    for k, v in result["metrics"].items():
                        accum_metrics.setdefault(k, []).append(v)
                    t_teacher_sum += result["t_teacher"]
                    t_student_sum += result["t_student"]
                    t_loss_bwd_sum += result["t_loss_bwd"]

                if epoch_done:
                    break

                if rank == 0:
                    full_pass_rate = _aggregate_pass_rate(step_metas)
                    reflector_fallback_count += sum(m["fallback"] for m in step_metas)
                    table_meta = step_metas[-1]["table"]
                else:
                    full_pass_rate = None
                    table_meta = None

                optimizer_step = _step_tail(
                    optimizer, scheduler, fsdp_model, model, logprob_comm, vllm_group, tokenizer,
                    rank=rank, device=device, epoch=epoch, optimizer_step=optimizer_step,
                    accum_loss_sum=accum_loss_sum, accum_samples=accum_samples,
                    accum_comp_len_sum=accum_comp_len_sum, accum_metrics=accum_metrics,
                    full_pass_rate=full_pass_rate, table_meta=table_meta,
                    policy_lags=policy_lags, t_teacher_sum=t_teacher_sum,
                    t_student_sum=t_student_sum, t_loss_bwd_sum=t_loss_bwd_sum,
                    t_step_start=t_step_start, t_generation=t_generation,
                    t_producer_wait=t_producer_wait,
                )
                if rank == 0 and ASYNC_IN_ORDER:
                    step_done_q.put(True)
        else:
            for _step in range(steps_per_epoch):
                t_step_start = time.monotonic()

                # ---- Rank 0: rollout + build broadcast data ----
                t_gen_start = time.monotonic()
                if rank == 0:
                    items = [next(data_iter) for _ in range(GRAD_ACCUM_STEPS)]
                    rollout_data, metas, batch_meta = produce(
                        items, success_cache, _OPTIMIZER_STEP, producer_tokenizer,
                    )
                else:
                    rollout_data = [None] * GRAD_ACCUM_STEPS
                t_generation = time.monotonic() - t_gen_start

                # ---- Broadcast rollout data to all ranks ----
                t_bcast_start = time.monotonic()
                dist.broadcast_object_list(rollout_data, src=0)
                t_broadcast = time.monotonic() - t_bcast_start

                # ---- Each rank slices its portion ----
                my_items = rollout_data[rank * local_accum_steps : (rank + 1) * local_accum_steps]

                # ---- Gradient accumulation loop ----
                optimizer.zero_grad()
                accum_loss_sum: float = 0.0
                accum_samples: int = 0
                accum_comp_len_sum: int = 0
                accum_metrics: dict[str, list[float]] = {}
                t_teacher_sum: float = 0.0
                t_student_sum: float = 0.0
                t_loss_bwd_sum: float = 0.0

                for micro_step, item_data in enumerate(my_items):
                    _log_consume_hash(optimizer_step, rank, micro_step, item_data)
                    result = _train_sample(
                        item_data, micro_step, local_accum_steps, fsdp_model, model,
                        tokenizer, vocab_size, device,
                    )
                    if result is None:
                        continue
                    accum_loss_sum += result["loss"]
                    accum_samples += 1
                    accum_comp_len_sum += result["comp_len"]
                    epoch_loss_sum += result["loss"]
                    epoch_samples += 1
                    for k, v in result["metrics"].items():
                        accum_metrics.setdefault(k, []).append(v)
                    t_teacher_sum += result["t_teacher"]
                    t_student_sum += result["t_student"]
                    t_loss_bwd_sum += result["t_loss_bwd"]

                if rank == 0:
                    full_pass_rate = batch_meta["full_pass_rate"]
                    reflector_fallback_count += batch_meta["fallback_delta"]
                    table_meta = batch_meta["table"]
                else:
                    full_pass_rate = None
                    table_meta = None

                optimizer_step = _step_tail(
                    optimizer, scheduler, fsdp_model, model, logprob_comm, vllm_group, tokenizer,
                    rank=rank, device=device, epoch=epoch, optimizer_step=optimizer_step,
                    accum_loss_sum=accum_loss_sum, accum_samples=accum_samples,
                    accum_comp_len_sum=accum_comp_len_sum, accum_metrics=accum_metrics,
                    full_pass_rate=full_pass_rate, table_meta=table_meta,
                    policy_lags=[], t_teacher_sum=t_teacher_sum,
                    t_student_sum=t_student_sum, t_loss_bwd_sum=t_loss_bwd_sum,
                    t_step_start=t_step_start, t_generation=t_generation,
                    t_producer_wait=0.0,
                )

        # ---- Epoch summary ----
        avg_epoch_loss = epoch_loss_sum / max(epoch_samples, 1)
        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} done. "
            f"avg_loss={avg_epoch_loss:.4f} samples={epoch_samples}"
        )
        if reflector_fallback_count > 0:
            logger.info(f"reflector fallback: {reflector_fallback_count}/{epoch_samples} envs")
        if rank == 0:
            wandb.log(
                {"epoch/avg_loss": avg_epoch_loss, "epoch/number": epoch + 1},
                step=optimizer_step,
            )
        # FSDP export is collective — all ranks save
        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
        save_hf_checkpoint(model, ckpt_dir, tokenizer, rank=rank)

    # ---- Final checkpoint + shutdown ----
    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
    save_hf_checkpoint(model, ckpt_dir, tokenizer, rank=rank)
    if rank == 0:
        wandb.finish()

    cleanup()
    logger.info("Training complete.")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    train()
