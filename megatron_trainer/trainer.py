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
a time. See docs/megatron_trainer/async_rollouts.md.

Launch: torchrun --nproc_per_node=N -m megatron_trainer.trainer
"""

import concurrent.futures
import os
import queue
import shutil
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
from megatron_trainer.chunked_head import make_grpo_processor, make_kl_processor
from megatron_trainer.config import (
    ASYNC_ROLLOUT,
    BATCH_SIZE,
    EMA_ALPHA,
    ENV_TYPE,
    GEN_MAX_NEW_TOKENS,
    GRAD_ACCUM_STEPS,
    GRPO_ADV,
    GRPO_CLIP_HIGH,
    GRPO_CLIP_LOW,
    GRPO_GRAD_CLIP,
    GRPO_GROUPS,
    GRPO_IS_C_MAX,
    GRPO_KL_COEF,
    GRPO_LR,
    GRPO_LR_WARMUP_STEPS,
    GRPO_MASK_TRUNCATED,
    GRPO_OLD_LOGPS,
    HF_MODEL_PATH,
    HINDSIGHT_FIELD,
    IS_CAP,
    IS_WEIGHTING,
    LEARNING_RATE,
    LORA_ADAPTER_NAME,
    LORA_ALPHA,
    LORA_DIM,
    LORA_TARGET_MODULES,
    LOSS_TYPE,
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
    TRAIN_MODE,
    TRAINER_SEED,
    USE_LOGPROB_SERVER,
    VLLM_BASE_URL,
    VLLM_BASE_URLS,
    WANDB_PROJECT,
    WANDB_ENTITY,
    WANDB_NAME,
)
from megatron_trainer.env import ApiAdapterEnv, RagEnv
from megatron_trainer.model_utils import (
    apply_lora_transform,
    cleanup,
    init_distributed_trainer,
    load_model,
    register_fsdp_module_mappings,
    save_hf_adapter_checkpoint,
    save_hf_checkpoint,
    sync_adapter_grads,
)
from megatron_trainer.logprob_client import (
    init_logprob_weight_engine,
    request_teacher_log_probs_tcp,
    sync_weights_to_logprob_server,
    wait_for_logprob_server,
)
from megatron_trainer.vllm_utils import (
    init_vllm_weight_engine,
    push_lora_adapter,
    sync_weights_to_vllm,
    wait_for_vllm,
)

# Completed optimizer steps — written by the main thread, read by the rollout
# producer to version-stamp generations (policy_version).
_OPTIMIZER_STEP: int = 0

# Transient (non-SAVE_EVERY) LoRA adapter dirs awaiting deletion — see the
# TRAIN_MODE == "lora" branch of _step_tail for why deletion is delayed.
# Margin of 3 is comfortably above vLLM's --max-cpu-loras=2 (train_full.sh):
# a 1-step delay still crashed the engine (observed twice, reproducibly, ~8s
# after the first real rmtree each time) — the CPU-side LRU adapter cache
# apparently keeps more than just the immediately-previous generation alive.
_PENDING_DELETE_DIRS: deque[str] = deque()
_DELETE_DELAY_STEPS = 3

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
        # GRPO always needs the reflector verdict (it's the reward) regardless
        # of HINDSIGHT_FIELD — the privileged-prompt text it also builds is
        # simply unused by the grpo training path (no teacher, no hindsight).
        use_reflector = HINDSIGHT_FIELD == "online_feedback" or LOSS_TYPE == "grpo"
        return RagEnv(
            prompt_text=item["prompt_texts"][0],
            vllm_base_url=url,
            privileged_information_prompt=item["conditional_texts"][0],
            raw_question=item["raw_questions"][0],
            golden_answer=item["golden_answers"][0],
            normalized_messages=item["normalized_messages"][0],
            tokenizer=tokenizer,
            use_reflector=use_reflector,
            reflector_verdict_only=(LOSS_TYPE == "grpo"),
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


def produce(items: list[dict], success_cache: dict[str, str], policy_version: int, tokenizer):
    """Rank-0 rollout for one batch of items.

    Builds envs, runs them, updates success_cache (api_adapter), and returns
    (rollout_data, metas, batch_meta). rollout_data is the broadcast payload;
    metas/batch_meta feed the wandb logging.
    """
    envs = [_build_env(item, success_cache, tokenizer, i) for i, item in enumerate(items)]

    with ThreadPoolExecutor(max_workers=min(32, len(envs))) as executor:
        list(executor.map(lambda e: e.run(), envs))

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


def _push_sample(rollout_queue: queue.Queue, env, policy_version: int, success_cache: dict[str, str]) -> None:
    """Post-process one completed env and push its payload (with _meta) to the queue."""
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


def _produce_streaming(
    rollout_queue: queue.Queue,
    data_iter,
    success_cache: dict[str, str],
    gen_times: deque,
    tokenizer,
    steps_per_epoch: int,
) -> None:
    """Streaming producer: N_ASYNC envs in flight, per-sample push on
    completion — completion order, not dataset order (the reordering is the
    feature). Submits exactly steps_per_epoch * GRAD_ACCUM_STEPS samples (the
    same count the sync path trains), drains in-flight work, then pushes the
    end-of-data sentinel."""
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
                _push_sample(rollout_queue, env, version, success_cache)
                gen_times.append(time.monotonic() - t_start)
    for fut in concurrent.futures.as_completed(window):
        env, t_start, version = window[fut]
        fut.result()
        _push_sample(rollout_queue, env, version, success_cache)
        gen_times.append(time.monotonic() - t_start)
    rollout_queue.put(_ROLLOUT_SENTINEL)


def _push_group(rollout_queue: queue.Queue, envs: list, policy_version: int, success_cache: dict[str, str]) -> None:
    """Post-process one completed GROUP (G rollouts of the same prompt): compute
    rewards (reflector/episode verdict), group-relative advantages, and
    degeneracy, then push the whole group as ONE queue item (list of G
    payloads) — group-atomic, since the advantage needs every member's
    reward before any of them can train."""
    if ENV_TYPE == "api_adapter":
        for env in envs:
            if env.episode_result and env.completion_text:
                parsed_verdict, parsed_feedback = env.parse_adapter_response(env.completion_text)
                if parsed_verdict:
                    cached_text = f"Verdict: {parsed_verdict}\nFeedback: {parsed_feedback}"
                    success_cache[env.raw_question] = cached_text

    metas = [_sample_meta(env) for env in envs]
    rewards = [m["pass_value"] if m["pass_value"] is not None else 0.0 for m in metas]
    mean_r = sum(rewards) / len(rewards)
    advantages = [r - mean_r for r in rewards]
    is_degenerate = len(set(rewards)) == 1
    group_total_tokens = sum(len(env.completion_log_probs or []) for env in envs)

    group = [
        {
            "prompt_text": env.prompt_text,
            "completion_text": env.completion_text,
            "completion_log_probs": env.completion_log_probs,
            "finish_reason": env.finish_reason,
            "policy_version": policy_version,
            "reward": rewards[i],
            "advantage": advantages[i],
            "group_total_tokens": group_total_tokens,
            "is_degenerate": is_degenerate,
            "_meta": metas[i],
        }
        for i, env in enumerate(envs)
    ]
    rollout_queue.put(group)


def _produce_streaming_grpo(
    rollout_queue: queue.Queue,
    data_iter,
    success_cache: dict[str, str],
    gen_times: deque,
    tokenizer,
    steps_per_epoch: int,
    group_size: int,
) -> None:
    """Streaming producer for GRPO: the GROUP (not the individual rollout) is
    the unit of async overlap. Up to N_ASYNC // group_size groups in flight;
    each group's `group_size` completions of the SAME prompt run concurrently
    and are pushed to the queue together only once all finish (group-atomic —
    see _push_group). Submits exactly steps_per_epoch * (GRAD_ACCUM_STEPS //
    group_size) groups (the same count the training loop consumes), drains
    in-flight work, then pushes the end-of-data sentinel."""
    n_groups_async = max(1, N_ASYNC // group_size)
    executor = ThreadPoolExecutor(max_workers=n_groups_async)
    window: dict = {}
    sentinel = object()
    submit_limit = steps_per_epoch * (GRAD_ACCUM_STEPS // group_size)
    submitted = 0

    def run_group(item: dict) -> list:
        envs = [_build_env(item, success_cache, tokenizer, g) for g in range(group_size)]
        with ThreadPoolExecutor(max_workers=group_size) as inner:
            list(inner.map(lambda e: e.run(), envs))
        return envs

    while submitted < submit_limit:
        item = next(data_iter, sentinel)
        if item is sentinel:
            break
        t_start = time.monotonic()
        fut = executor.submit(run_group, item)
        window[fut] = (t_start, _OPTIMIZER_STEP)
        submitted += 1
        while len(window) >= n_groups_async:
            done, _ = concurrent.futures.wait(
                window.keys(), return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                t_start, version = window.pop(fut)
                envs = fut.result()
                _push_group(rollout_queue, envs, version, success_cache)
                gen_times.append(time.monotonic() - t_start)
    for fut in concurrent.futures.as_completed(window):
        t_start, version = window[fut]
        envs = fut.result()
        _push_group(rollout_queue, envs, version, success_cache)
        gen_times.append(time.monotonic() - t_start)
    rollout_queue.put(_ROLLOUT_SENTINEL)
    executor.shutdown(wait=False)


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
    # no_sync on non-final micro-steps (skip allreduce). LoRA mode has no FSDP
    # wrapper (replicated base per rank) — plain accumulation, single
    # all_reduce at step end.
    is_final = (micro_step == local_accum_steps - 1)
    ctx = nullcontext() if (is_final or fsdp_model is None) else fsdp_model.no_sync()
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


def _train_sample_grpo(
    item_data: dict,
    micro_step: int,
    local_accum_steps: int,
    gpg_rescale: float,
    fsdp_model,
    model,
    tokenizer,
    device: torch.device,
) -> dict | None:
    """One GRPO sample: student forward (attached sampled-token logp) →
    clipped-surrogate policy-gradient backward. No teacher call at all
    (GRPO_KL_COEF=0 — the only mode implemented so far).
    """
    completion_ids: list[int] = tokenizer.encode(
        item_data["completion_text"], add_special_tokens=False,
    )
    if len(completion_ids) == 0:
        logger.warning(f"Empty completion, skipping micro_step {micro_step}")
        return None
    completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]
    C = len(completion_ids)

    lp_list = item_data.get("completion_log_probs")
    if not lp_list:
        logger.warning(f"Missing rollout log-probs, skipping micro_step {micro_step}")
        return None
    lp_list = lp_list[:C]
    rollout_log_probs = torch.tensor(
        [float("nan") if v is None else float(v) for v in lp_list],
        dtype=torch.float32,
        device=device,
    )
    if rollout_log_probs.size(0) != C:
        pad = torch.full(
            (C - rollout_log_probs.size(0),), float("nan"), dtype=torch.float32, device=device,
        )
        rollout_log_probs = torch.cat([rollout_log_probs, pad])

    mask_all = GRPO_MASK_TRUNCATED and item_data.get("finish_reason") == "length"

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

    grpo_processor = make_grpo_processor(
        prompt_len=prompt_len,
        completion_ids=completion_ids,
        advantage=item_data["advantage"],
        group_total_tokens=item_data["group_total_tokens"],
        gpg_rescale=gpg_rescale,
        device=device,
        rollout_log_probs=rollout_log_probs,
        clip_low=GRPO_CLIP_LOW,
        clip_high=GRPO_CLIP_HIGH,
        is_c_max=GRPO_IS_C_MAX,
        old_logps_mode=GRPO_OLD_LOGPS,
        mask_all=mask_all,
    )
    loss, step_metrics = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        output_processor=grpo_processor,
    )
    t_student = time.monotonic() - t0

    t0 = time.monotonic()
    is_final = (micro_step == local_accum_steps - 1)
    ctx = nullcontext() if (is_final or fsdp_model is None) else fsdp_model.no_sync()
    with ctx:
        scaled_loss = loss / local_accum_steps
        scaled_loss.backward()
    t_loss_bwd = time.monotonic() - t0

    return {
        "loss": loss.item(),
        "comp_len": C,
        "metrics": step_metrics,
        "t_teacher": 0.0,
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
    if fsdp_model is not None:
        fsdp_model.finish_grad_sync()
    else:
        # LoRA mode: flattened all_reduce over adapter grads replaces FSDP
        # grad sync (no wrapper exists).
        sync_adapter_grads(model)
    grad_norm = clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        GRPO_GRAD_CLIP if LOSS_TYPE == "grpo" else MAX_GRAD_NORM,
    )
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
    if TRAIN_MODE == "lora":
        # Adapter hot-swap into vLLM (drain barrier + load_inplace) +
        # adapter-only EMA sync to the logprob server (frozen base both sides).
        # USE_LOGPROB_SERVER=False (grpo, GRPO_KL_COEF=0) skips this entirely —
        # there's no reference/teacher forward pass, so no logprob server was
        # ever initialized to sync to (matches the TRAIN_MODE=full branch below).
        if not TEACHER_MODEL_PATH and USE_LOGPROB_SERVER:
            sync_weights_to_logprob_server(
                model, logprob_comm, rank=rank, trainable_only=True
            )
        adapter_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
        push_lora_adapter(model, adapter_dir, rank=rank)
        if rank == 0:
            # push_lora_adapter always writes adapter_dir to disk (vLLM's
            # hot-swap loads the new adapter from a local path). The on-disk
            # copy is only needed long-term for SAVE_EVERY-cadence "real"
            # checkpoints — every other optimizer step's dir is transient
            # (disk pressure, and auto_eval_poller.sh would try to eval every
            # single one) and gets deleted.
            #
            # BUT: vLLM's HTTP response confirms the load request was
            # accepted, not that every internal reference to the old file is
            # gone (safetensors loads can be mmap-backed and page in lazily;
            # --max-cpu-loras=2 also means vLLM's LRU cache can hold more
            # than just the immediately-previous adapter generation).
            # Deleting too early risks a use-after-free crash in the engine.
            # Keep a rolling window of the last _DELETE_DELAY_STEPS transient
            # dirs alive and only delete once a dir falls out of that window
            # (i.e. that many subsequent pushes have succeeded since).
            if optimizer_step % SAVE_EVERY != 0:
                _PENDING_DELETE_DIRS.append(adapter_dir)
            while len(_PENDING_DELETE_DIRS) > _DELETE_DELAY_STEPS:
                shutil.rmtree(_PENDING_DELETE_DIRS.popleft(), ignore_errors=True)
    else:
        if not TEACHER_MODEL_PATH and USE_LOGPROB_SERVER:
            sync_weights_to_logprob_server(model, logprob_comm, rank=rank)
        sync_weights_to_vllm(model, device, vllm_group, rank=rank)
    t_weight_sync = time.monotonic() - t0

    if optimizer_step % SAVE_EVERY == 0:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
        if TRAIN_MODE == "lora":
            save_hf_adapter_checkpoint(model, ckpt_dir, rank=rank)
        else:
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
    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_level = os.environ.get("LOGGING_LEVEL", "DEBUG")
    logger.add(os.path.join(log_dir, "trainer.log"), level=log_level)

    logger.info("=== SDFT Megatron Trainer Starting ===")
    logger.info(
        f"Config: TRAIN_MODE={TRAIN_MODE} LORA_DIM={LORA_DIM} LORA_ALPHA={LORA_ALPHA} "
        f"LORA_TARGET_MODULES={LORA_TARGET_MODULES} LORA_ADAPTER_NAME={LORA_ADAPTER_NAME}"
    )

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
    if LOSS_TYPE == "grpo":
        assert local_accum_steps % GRPO_GROUPS == 0, (
            f"local_accum_steps ({local_accum_steps} = GRAD_ACCUM_STEPS/world_size) must "
            f"be divisible by GRPO_GROUPS ({GRPO_GROUPS}) — groups are rank-local"
        )
        if ASYNC_ROLLOUT:
            n_groups_async = max(1, N_ASYNC // GRPO_GROUPS)
            assert n_groups_async >= world_size, (
                f"N_ASYNC ({N_ASYNC}) // GRPO_GROUPS ({GRPO_GROUPS}) = {n_groups_async} "
                f"in-flight producer groups, but world_size ({world_size}) groups are "
                f"needed concurrently (1/rank/step) — a rank's group would queue behind "
                f"another, adding a full extra generation round that risks the NCCL "
                f"collective timeout. Set N_ASYNC >= world_size * GRPO_GROUPS "
                f"({world_size * GRPO_GROUPS})."
            )

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
    if TRAIN_MODE == "lora":
        # Freeze base + inject adapters (bridge PEFT). MUST happen before any
        # distributed wrapping — lora mode skips FSDP entirely.
        model = apply_lora_transform(model)
    model.train()

    # Use padded vocab size from model (Megatron pads for TP alignment)
    unwrapped = model.module if hasattr(model, 'module') else model
    vocab_size = unwrapped.vocab_size
    logger.info(
        f"Model loaded. vocab_size={vocab_size} (tokenizer={tokenizer.vocab_size}) "
        f"GPU mem after load: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
    )

    # ---- MCore FSDP wrapping (full mode only) ----
    # LoRA mode: replicated frozen base per rank, no FSDP wrapper (bridge LoRA
    # + MCore FSDP unverified upstream; adapter grads are tiny so a flat
    # all_reduce at step end replaces finish_grad_sync). fsdp_model=None
    # signals lora mode to _train_sample / _step_tail.
    fsdp_model = None
    if TRAIN_MODE == "full":
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
    else:
        logger.info("LoRA mode: no FSDP wrapping (replicated base + adapter-grad all_reduce).")
    # The Megatron bridge closes stray file descriptors during model load,
    # silently killing the loguru file sink opened at boot (writes fail after
    # "Loading model"). Re-open it so the trainer log covers the training loop.
    logger.add(os.path.join(log_dir, "trainer.log"), level=log_level)

    # ---- Optimizer (FSDP: torch AdamW; LoRA: adapter params only) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_lr = GRPO_LR if LOSS_TYPE == "grpo" else LEARNING_RATE
    optimizer = torch.optim.AdamW(
        trainable_params, lr=optimizer_lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    logger.info(
        f"torch AdamW optimizer ready. LR={optimizer_lr} "
        f"trainable_params={sum(p.numel() for p in trainable_params):,}"
    )

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
    # GRPO consumes one unique prompt per GROUP (not per rollout) — a "step
    # unit" is GRAD_ACCUM_STEPS//GRPO_GROUPS unique prompts, each expanded
    # into G rollouts by the producer.
    step_unit = GRAD_ACCUM_STEPS // GRPO_GROUPS if LOSS_TYPE == "grpo" else GRAD_ACCUM_STEPS
    steps_per_epoch = len(dataset) // step_unit
    unit_label = "unique prompts" if LOSS_TYPE == "grpo" else "examples"
    logger.info(f"Dataset: {len(dataset)} examples, {steps_per_epoch} steps/epoch ({step_unit} {unit_label}/step)")

    # ---- LR scheduler (total steps known only after dataset load) ----
    total_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = 0
    scheduler = None
    if LOSS_TYPE == "grpo":
        warmup_steps = GRPO_LR_WARMUP_STEPS
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(warmup_steps, 1))
        )
        logger.info(f"LR scheduler: grpo linear warmup ({warmup_steps} steps) then constant at {GRPO_LR}")
    elif LR_SCHEDULER == "cosine":
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
                "backend": "fsdp" if TRAIN_MODE == "full" else "lora",
                "train_mode": TRAIN_MODE,
                "lora_dim": LORA_DIM,
                "lora_alpha": LORA_ALPHA,
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
                "n_async": N_ASYNC,
                "loss": LOSS_TYPE if LOSS_TYPE == "grpo" else "reverse_kl",
                "dataset": TRAIN_DATA_PATH,
                "hindsight_field": HINDSIGHT_FIELD,
                **(
                    {
                        "grpo_groups": GRPO_GROUPS,
                        "grpo_adv": GRPO_ADV,
                        "grpo_clip_low": GRPO_CLIP_LOW,
                        "grpo_clip_high": GRPO_CLIP_HIGH,
                        "grpo_old_logps": GRPO_OLD_LOGPS,
                        "grpo_is_c_max": GRPO_IS_C_MAX,
                        "grpo_kl_coef": GRPO_KL_COEF,
                        "grpo_lr": GRPO_LR,
                        "grpo_lr_warmup_steps": GRPO_LR_WARMUP_STEPS,
                        "grpo_grad_clip": GRPO_GRAD_CLIP,
                        "grpo_mask_truncated": GRPO_MASK_TRUNCATED,
                    }
                    if LOSS_TYPE == "grpo"
                    else {}
                ),
            },
        )

        logger.info("Waiting for vLLM server...")
        wait_for_vllm()
        if TRAIN_MODE == "lora":
            # No NCCL weight transfer engine — adapters are hot-swapped via
            # HTTP (POST /v1/load_lora_adapter).
            logger.info("LoRA mode: skipping vLLM weight transfer engine init.")
        else:
            logger.info("Initializing vLLM weight transfer engine...")
            vllm_group = init_vllm_weight_engine(device)
            logger.info("vLLM weight engine ready.")

        if USE_LOGPROB_SERVER:
            logger.info("Waiting for logprob server...")
            wait_for_logprob_server()
            # External frozen teacher (TEACHER_MODEL_PATH set): no NCCL weight
            # sync — the teacher keeps its downloaded weights.
            if not TEACHER_MODEL_PATH:
                logger.info("Initializing logprob weight transfer engine...")
                logprob_comm = init_logprob_weight_engine(device)
                logger.info("Logprob weight engine ready.")
        else:
            logger.info("LOSS_TYPE=grpo with GRPO_KL_COEF=0: skipping logprob server entirely.")

    # Barrier: all ranks wait for rank 0 to finish setup
    dist.barrier()

    if TRAIN_MODE == "lora":
        # Bootstrap push (collective): the first rollout wave starts before any
        # step-end adapter push, and vLLM 404s on unknown adapter names (no
        # silent base fallback). Push the zero-init adapter — lora_B=0 makes it
        # mathematically equivalent to the base model.
        adapter_dir = os.path.join(OUTPUT_DIR, "step_0")
        push_lora_adapter(model, adapter_dir, rank=rank)

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
        gen_times: deque | None = None

        if ASYNC_ROLLOUT and rank == 0:
            threading.excepthook = _crash_hard_on_thread_error
            if LOSS_TYPE == "grpo":
                rollout_queue = queue.Queue(maxsize=max(N_ASYNC // GRPO_GROUPS, 1) + world_size + 1)
                gen_times = deque()
                producer_thread = threading.Thread(
                    target=_produce_streaming_grpo,
                    args=(rollout_queue, data_iter, success_cache, gen_times, producer_tokenizer, steps_per_epoch, GRPO_GROUPS),
                    name="rollout-producer-grpo",
                    daemon=False,
                )
            else:
                rollout_queue = queue.Queue(maxsize=N_ASYNC + world_size + 1)
                gen_times = deque()
                producer_thread = threading.Thread(
                    target=_produce_streaming,
                    args=(rollout_queue, data_iter, success_cache, gen_times, producer_tokenizer, steps_per_epoch),
                    name="rollout-producer",
                    daemon=False,
                )
            producer_thread.start()

        if LOSS_TYPE == "grpo":
            # Group-atomic streaming consumer: pull whole groups (G rollouts
            # of the same prompt) until the epoch drains. Each optimizer step
            # buffers all of this rank's groups first (so gpg_rescale is known
            # before any forward pass), then trains member-by-member.
            groups_per_step = local_accum_steps // GRPO_GROUPS
            while True:
                t_step_start = time.monotonic()
                t_producer_wait: float = 0.0
                t_generation: float = 0.0
                policy_lags: list[int] = []

                optimizer.zero_grad()
                accum_loss_sum: float = 0.0
                accum_samples: int = 0
                accum_comp_len_sum: int = 0
                accum_metrics: dict[str, list[float]] = {}
                t_teacher_sum: float = 0.0
                t_student_sum: float = 0.0
                t_loss_bwd_sum: float = 0.0

                groups_for_step: list[list[dict]] = []
                epoch_done = False
                for _ in range(groups_per_step):
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
                    groups_for_step.append(microbatch[rank])

                if epoch_done:
                    break

                num_groups_local = len(groups_for_step)
                num_nondeg_local = sum(1 for g in groups_for_step if not g[0]["is_degenerate"])
                # GPG rescale must reflect the whole step's population of
                # groups across all ranks, not just this rank's local slice.
                # With the common config (groups_per_step=1, i.e. world_size
                # groups/step, 1/rank), a per-rank-local count can only ever
                # be 0/1 or 1/1 -- making the rescale a no-op (1.0, or a
                # no-op 10000x of an already-zero degenerate loss) instead of
                # the intended cross-rank compensation (up-weight surviving
                # non-degenerate groups' gradient by however many degenerate
                # groups were dropped this step, so the effective per-step
                # gradient magnitude doesn't shrink as the degenerate rate
                # rises). All-reduce the local counts to get the true step-
                # wide totals before computing the rescale factor.
                counts = torch.tensor(
                    [float(num_groups_local), float(num_nondeg_local)], device=device,
                )
                dist.all_reduce(counts)
                num_groups_global, num_nondeg_global = counts[0].item(), counts[1].item()
                gpg_rescale = num_groups_global / max(num_nondeg_global, 1e-4)
                if rank == 0:
                    policy_lags = [_OPTIMIZER_STEP - g[0]["policy_version"] for g in groups_for_step]

                flat_members = [m for g in groups_for_step for m in g]
                for micro, item_data in enumerate(flat_members):
                    result = _train_sample_grpo(
                        item_data, micro, local_accum_steps, gpg_rescale, fsdp_model, model,
                        tokenizer, device,
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
                    step_metas = [m["_meta"] for g in groups_for_step for m in g]
                    full_pass_rate = _aggregate_pass_rate(step_metas)
                    reflector_fallback_count += sum(m["fallback"] for m in step_metas)
                    table_meta = step_metas[-1]["table"] if step_metas else None
                    accum_metrics.setdefault("grpo/frac_reward_zero_std", []).append(
                        1.0 - (num_nondeg_global / max(num_groups_global, 1))
                    )
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
        elif ASYNC_ROLLOUT:
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
        if TRAIN_MODE == "lora":
            save_hf_adapter_checkpoint(model, ckpt_dir, rank=rank)
        else:
            save_hf_checkpoint(model, ckpt_dir, tokenizer, rank=rank)

    # ---- Final checkpoint + shutdown ----
    ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{optimizer_step}")
    if TRAIN_MODE == "lora":
        save_hf_adapter_checkpoint(model, ckpt_dir, rank=rank)
    else:
        save_hf_checkpoint(model, ckpt_dir, tokenizer, rank=rank)
    if rank == 0:
        wandb.finish()

    cleanup()
    logger.info("Training complete.")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    train()
