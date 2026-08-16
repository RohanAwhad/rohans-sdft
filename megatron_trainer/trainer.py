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
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

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
    GRPO_FILTER_GROUPS,
    GRPO_GRAD_CLIP,
    GRPO_GROUPS,
    GRPO_IS_C_MAX,
    GRPO_IS_MODE,
    GRPO_KL_COEF,
    GRPO_LR,
    GRPO_LR_WARMUP_STEPS,
    GRPO_MASK_TRUNCATED,
    GRPO_MAX_GEN_BATCHES,
    GRPO_OLD_LOGPS,
    HF_MODEL_PATH,
    HINDSIGHT_FIELD,
    IS_CAP,
    IS_WEIGHTING,
    LEARNING_RATE,
    LR_SCHEDULER,
    LOSS_TYPE,
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
from megatron_trainer.grpo import (
    grpo_async_queue_order,
    grpo_kl_special_token_ids,
    grpo_rollouts_per_prompt,
    grpo_seed_offset,
    prepare_grpo_group,
    stamp_grpo_loss_scales,
)
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


def _build_env(
    item: dict,
    success_cache: dict[str, str],
    tokenizer,
    vllm_idx: int,
    seed_offset: int = 0,
):
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
            seed_offset=seed_offset,
            reward_only=LOSS_TYPE == "grpo",
        )
    return ApiAdapterEnv(
        prompt_text=item["prompt_texts"][0],
        vllm_base_url=url,
        raw_question=item["raw_questions"][0],
        golden_answer=item["golden_answers"][0],
        tokenizer=tokenizer,
        success_cache=success_cache,
        seed_offset=seed_offset,
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


def _update_success_cache(envs: list, success_cache: dict[str, str]) -> None:
    if ENV_TYPE == "api_adapter":
        for env in envs:
            if env.episode_result and env.completion_text:
                parsed_verdict, parsed_feedback = env.parse_adapter_response(env.completion_text)
                if parsed_verdict:
                    cached_text = f"Verdict: {parsed_verdict}\nFeedback: {parsed_feedback}"
                    success_cache[env.raw_question] = cached_text


def _run_envs(envs: list, *, max_workers: int = 32) -> None:
    with ThreadPoolExecutor(max_workers=min(max_workers, len(envs))) as executor:
        list(executor.map(lambda env: env.run(), envs))


def _produce_sdft(
    items: list[dict],
    success_cache: dict[str, str],
    policy_version: int,
    tokenizer,
):
    """Existing one-completion-per-prompt SDFT rollout path."""
    envs = [_build_env(item, success_cache, tokenizer, i) for i, item in enumerate(items)]
    _run_envs(envs)
    _update_success_cache(envs, success_cache)

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


def _grpo_payload(env, policy_version: int, tokenizer) -> dict:
    if env.completion_text is None:
        raise ValueError("GRPO env returned no completion_text")
    meta = _sample_meta(env)
    completion_ids = tokenizer.encode(env.completion_text, add_special_tokens=False)
    return {
        "prompt_text": env.prompt_text,
        "completion_text": env.completion_text,
        "completion_log_probs": env.completion_log_probs,
        "finish_reason": env.finish_reason,
        "privileged_information_prompt": env.privileged_information_prompt,
        "policy_version": policy_version,
        "reward": meta["pass_value"],
        "active_token_count": min(len(completion_ids), GEN_MAX_NEW_TOKENS),
        "_meta": meta,
    }


def _produce_grpo(
    items: list[dict],
    success_cache: dict[str, str],
    policy_version: int,
    tokenizer,
    world_size: int,
    rollout_batch: int,
):
    """Generate, score, and normalize complete prompt-local GRPO groups."""
    generated_per_prompt = grpo_rollouts_per_prompt(GRPO_GROUPS, GRPO_ADV)
    pending = {slot: item for slot, item in enumerate(items)}
    accepted: dict[int, list[dict]] = {}
    collected: dict[int, dict[int, dict]] = {slot: {} for slot in pending}
    pending_reasons: dict[int, str] = {}

    for attempt in range(GRPO_MAX_GEN_BATCHES):
        if not pending:
            break
        envs: list = []
        group_envs: dict[int, dict[int, object]] = {}
        for slot, item in pending.items():
            group_envs[slot] = {}
            for group_index in range(generated_per_prompt):
                if group_index in collected[slot]:
                    continue
                seed_offset = grpo_seed_offset(
                    rollout_batch=rollout_batch,
                    attempt=attempt,
                    prompt_slot=slot,
                    group_index=group_index,
                    prompts_per_step=len(items),
                    rollouts_per_prompt=generated_per_prompt,
                    max_gen_batches=GRPO_MAX_GEN_BATCHES,
                )
                env = _build_env(
                    item,
                    success_cache,
                    tokenizer,
                    len(envs),
                    seed_offset=seed_offset,
                )
                envs.append(env)
                group_envs[slot][group_index] = env

        _run_envs(envs, max_workers=1)
        _update_success_cache(envs, success_cache)

        next_pending: dict[int, dict] = {}
        for slot, item in pending.items():
            for group_index, env in group_envs[slot].items():
                payload = _grpo_payload(env, policy_version, tokenizer)
                if payload["reward"] is None:
                    logger.info(
                        f"GRPO retry: prompt_slot={slot} group_index={group_index} "
                        f"attempt={attempt + 1} has no reward"
                    )
                elif payload["active_token_count"] == 0:
                    logger.info(
                        f"GRPO retry: prompt_slot={slot} group_index={group_index} "
                        f"attempt={attempt + 1} has an empty completion"
                    )
                else:
                    collected[slot][group_index] = payload
            if len(collected[slot]) != generated_per_prompt:
                next_pending[slot] = item
                pending_reasons[slot] = "missing reward or empty completion"
                continue
            payloads = [
                collected[slot][group_index]
                for group_index in range(generated_per_prompt)
            ]
            group = prepare_grpo_group(
                payloads,
                group_id=rollout_batch * len(items) + slot,
                advantage_type=GRPO_ADV,
                mask_truncated=GRPO_MASK_TRUNCATED,
            )
            if len(group) != GRPO_GROUPS:
                raise ValueError(
                    f"Prepared GRPO group has {len(group)} rollouts, "
                    f"expected {GRPO_GROUPS}"
                )
            if GRPO_FILTER_GROUPS and group[0]["group_degenerate"]:
                next_pending[slot] = item
                pending_reasons[slot] = "degenerate rewards"
                collected[slot].clear()
            else:
                accepted[slot] = group
                collected.pop(slot)
                pending_reasons.pop(slot, None)
        pending = next_pending

    if pending:
        reasons = ", ".join(
            f"{reason}: {sum(value == reason for value in pending_reasons.values())}"
            for reason in sorted(set(pending_reasons.values()))
        )
        raise RuntimeError(
            f"GRPO could not produce {len(pending)} complete groups after "
            f"{GRPO_MAX_GEN_BATCHES} batches ({reasons})"
        )

    rollout_data = [item for slot in sorted(accepted) for item in accepted[slot]]
    stamp_grpo_loss_scales(
        rollout_data,
        world_size=world_size,
        group_size=GRPO_GROUPS,
    )
    metas = [item["_meta"] for item in rollout_data]
    batch_meta = {
        "full_pass_rate": _aggregate_pass_rate(metas),
        "fallback_delta": sum(meta["fallback"] for meta in metas),
        "table": metas[-1]["table"],
    }
    return rollout_data, metas, batch_meta


def produce(
    items: list[dict],
    success_cache: dict[str, str],
    policy_version: int,
    tokenizer,
    world_size: int = 1,
    rollout_batch: int | None = None,
):
    """Rank-0 rollout for one optimizer-step input batch."""
    if LOSS_TYPE == "grpo":
        return _produce_grpo(
            items,
            success_cache,
            policy_version,
            tokenizer,
            world_size,
            policy_version if rollout_batch is None else rollout_batch,
        )
    return _produce_sdft(items, success_cache, policy_version, tokenizer)


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
    world_size: int,
) -> None:
    """Streaming producer: N_ASYNC envs in flight, per-sample push on
    completion — completion order, not dataset order (the reordering is the
    feature). Submits exactly steps_per_epoch * GRAD_ACCUM_STEPS samples (the
    same count the sync path trains), drains in-flight work, then pushes the
    end-of-data sentinel."""
    if LOSS_TYPE == "grpo":
        prompts_per_step = GRAD_ACCUM_STEPS // GRPO_GROUPS
        rollout_batch_start = _OPTIMIZER_STEP
        for batch_index in range(steps_per_epoch):
            items = [next(data_iter) for _ in range(prompts_per_step)]
            policy_version = _OPTIMIZER_STEP
            t_start = time.monotonic()
            rollout_data, _, _ = produce(
                items,
                success_cache,
                policy_version,
                tokenizer,
                world_size=world_size,
                rollout_batch=rollout_batch_start + batch_index,
            )
            for payload in grpo_async_queue_order(
                rollout_data,
                world_size=world_size,
                group_size=GRPO_GROUPS,
            ):
                rollout_queue.put(payload)
            gen_times.append(time.monotonic() - t_start)
        rollout_queue.put(_ROLLOUT_SENTINEL)
        return

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
    """Train one completion with the configured SDFT or GRPO objective."""
    completion_ids: list[int] = tokenizer.encode(
        item_data["completion_text"], add_special_tokens=False,
    )
    if len(completion_ids) == 0:
        logger.warning(f"Empty completion, skipping micro_step {micro_step}")
        return None
    completion_ids = completion_ids[:GEN_MAX_NEW_TOKENS]

    # Rollout log-probs; NaN marks deterministic inserted/template tokens.
    rollout_log_probs = None
    needs_rollout_logps = (
        IS_WEIGHTING if LOSS_TYPE == "sdft" else GRPO_OLD_LOGPS == "vllm"
    )
    if needs_rollout_logps:
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

    # SDFT always uses privileged teacher log-probs. GRPO requests a plain-
    # prompt reference only when its KL coefficient is non-zero.
    t0 = time.monotonic()
    reference_log_probs = None
    if LOSS_TYPE == "sdft":
        cond_ids: list[int] = tokenizer.encode(
            item_data["privileged_information_prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=TEACHER_MAX_PROMPT_LEN,
        )
        reference_log_probs = request_teacher_log_probs_tcp(
            token_ids=cond_ids + completion_ids,
            prompt_len=len(cond_ids),
            vocab_size=vocab_size,
            device=device,
        )
    elif GRPO_KL_COEF > 0:
        reference_log_probs = request_teacher_log_probs_tcp(
            token_ids=prompt_ids.tolist() + completion_ids,
            prompt_len=prompt_len,
            vocab_size=vocab_size,
            device=device,
        )
    t_teacher = time.monotonic() - t0

    if LOSS_TYPE == "sdft":
        loss_processor = make_kl_processor(
            prompt_len=prompt_len,
            completion_ids=completion_ids,
            teacher_log_probs=reference_log_probs,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            rollout_log_probs=rollout_log_probs,
            is_weighting=IS_WEIGHTING,
            is_cap=IS_CAP,
        )
    else:
        loss_processor = make_grpo_processor(
            prompt_len=prompt_len,
            completion_ids=completion_ids,
            advantage=item_data["advantage"],
            loss_scale=item_data["grpo_loss_scale"],
            rollout_log_probs=rollout_log_probs,
            old_logps_type=GRPO_OLD_LOGPS,
            clip_low=GRPO_CLIP_LOW,
            clip_high=GRPO_CLIP_HIGH,
            is_c_max=GRPO_IS_C_MAX,
            is_mode=GRPO_IS_MODE,
            reference_log_probs=reference_log_probs,
            kl_coef=GRPO_KL_COEF,
            special_token_ids=grpo_kl_special_token_ids(tokenizer),
            device=device,
        )

    t0 = time.monotonic()
    loss, step_metrics = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        output_processor=loss_processor,
    )
    t_student = time.monotonic() - t0

    if LOSS_TYPE == "grpo" and not torch.isfinite(loss).item():
        raise FloatingPointError(
            f"Non-finite GRPO loss at micro_step={micro_step}: {loss.item()}"
        )

    t0 = time.monotonic()
    is_final = (micro_step == local_accum_steps - 1)
    ctx = nullcontext() if is_final else fsdp_model.no_sync()
    with ctx:
        scaled_loss = loss / local_accum_steps if LOSS_TYPE == "sdft" else loss
        scaled_loss.backward()
    t_loss_bwd = time.monotonic() - t0

    if LOSS_TYPE == "grpo":
        step_metrics.update(
            {
                "grpo/mean_length": len(completion_ids),
                "grpo/frac_reward_zero_std": item_data["frac_reward_zero_std"],
                "grpo/adv_mean_std": item_data["group_advantage_std"],
                "grpo/gpg_rescale": item_data["gpg_rescale"],
            }
        )

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
    """Pop and broadcast one sample/rank (SDFT) or one group/rank (GRPO)."""
    samples_per_rank = GRPO_GROUPS if LOSS_TYPE == "grpo" else 1
    microbatch_size = world_size * samples_per_rank
    if rank == 0:
        first = rollout_queue.get()
        if first is _ROLLOUT_SENTINEL:
            microbatch = [None] * microbatch_size
            epoch_done = True
        else:
            microbatch = [first] + [
                rollout_queue.get() for _ in range(microbatch_size - 1)
            ]
            epoch_done = False
    else:
        microbatch = [None] * microbatch_size
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
    max_grad_norm = GRPO_GRAD_CLIP if LOSS_TYPE == "grpo" else MAX_GRAD_NORM
    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    if LOSS_TYPE == "grpo" and not torch.isfinite(grad_norm).item():
        raise FloatingPointError(
            f"Non-finite GRPO gradient norm before optimizer step: {grad_norm.item()}"
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
        avg_loss = (
            total_loss / dist.get_world_size()
            if LOSS_TYPE == "grpo"
            else total_loss / total_samples
        )
        global_grad_norm = agg_tensor[2].item() ** 0.5
        avg_comp_len = accum_comp_len_sum / max(accum_samples, 1)

        log_dict: dict = {
            "train/loss": avg_loss,
            "train/completion_length": avg_comp_len,
            "train/grad_norm": global_grad_norm,
            "train/epoch": epoch,
            "train/lr": (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else (GRPO_LR if LOSS_TYPE == "grpo" else LEARNING_RATE)
            ),
        }
        if full_pass_rate is not None:
            if LOSS_TYPE == "grpo":
                key = "grpo/pass_rate"
            else:
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
        if LOSS_TYPE == "grpo" and full_pass_rate is not None:
            metrics_str = f"grpo/pass_rate={full_pass_rate:.4f} {metrics_str}"
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
    needs_logprob_server = LOSS_TYPE == "sdft" or GRPO_KL_COEF > 0
    if needs_logprob_server and not TEACHER_MODEL_PATH:
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

    logger.info(f"=== {LOSS_TYPE.upper()} Megatron Trainer Starting ===")

    # ---- Initialize torch.distributed via torchrun ----
    local_rank = init_distributed_trainer()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if LOSS_TYPE == "grpo":
        assert GRAD_ACCUM_STEPS % (world_size * GRPO_GROUPS) == 0, (
            f"GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}) must be divisible by "
            f"num_trainers * GRPO_GROUPS ({world_size} * {GRPO_GROUPS})"
        )
    else:
        assert GRAD_ACCUM_STEPS % world_size == 0, (
            f"GRAD_ACCUM_STEPS ({GRAD_ACCUM_STEPS}) must be divisible by "
            f"num_trainers ({world_size})"
        )
    local_accum_steps = GRAD_ACCUM_STEPS // world_size
    local_group_steps = (
        local_accum_steps // GRPO_GROUPS if LOSS_TYPE == "grpo" else local_accum_steps
    )

    logger.info(f"FSDP: rank={rank}/{world_size}, local_rank={local_rank}, "
                f"local_accum_steps={local_accum_steps} "
                f"local_group_steps={local_group_steps}")

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
    active_learning_rate = GRPO_LR if LOSS_TYPE == "grpo" else LEARNING_RATE
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    logger.info(f"torch AdamW optimizer ready. LR={active_learning_rate}")

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
    prompts_per_step = (
        GRAD_ACCUM_STEPS // GRPO_GROUPS
        if LOSS_TYPE == "grpo"
        else GRAD_ACCUM_STEPS
    )
    steps_per_epoch = len(dataset) // prompts_per_step
    logger.info(
        f"Dataset: {len(dataset)} examples, {steps_per_epoch} steps/epoch, "
        f"prompts_per_step={prompts_per_step}"
    )

    # ---- LR scheduler (total steps known only after dataset load) ----
    total_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = 0
    scheduler = None
    if LOSS_TYPE == "grpo":
        warmup_steps = min(GRPO_LR_WARMUP_STEPS, total_train_steps)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
        logger.info(
            f"LR scheduler: GRPO constant, warmup={warmup_steps} over "
            f"{total_train_steps} optimizer steps"
        )
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
                "backend": "fsdp",
                "learning_rate": active_learning_rate,
                "lr_scheduler": "constant_with_warmup" if LOSS_TYPE == "grpo" else LR_SCHEDULER,
                "warmup_steps": warmup_steps,
                "max_grad_norm": GRPO_GRAD_CLIP if LOSS_TYPE == "grpo" else MAX_GRAD_NORM,
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
                "loss": "reverse_kl" if LOSS_TYPE == "sdft" else "grpo",
                "grpo_groups": GRPO_GROUPS if LOSS_TYPE == "grpo" else None,
                "grpo_adv": GRPO_ADV if LOSS_TYPE == "grpo" else None,
                "grpo_clip_low": GRPO_CLIP_LOW if LOSS_TYPE == "grpo" else None,
                "grpo_clip_high": GRPO_CLIP_HIGH if LOSS_TYPE == "grpo" else None,
                "grpo_old_logps": GRPO_OLD_LOGPS if LOSS_TYPE == "grpo" else None,
                "grpo_is_c_max": GRPO_IS_C_MAX if LOSS_TYPE == "grpo" else None,
                "grpo_is_mode": GRPO_IS_MODE if LOSS_TYPE == "grpo" else None,
                "grpo_kl_coef": GRPO_KL_COEF if LOSS_TYPE == "grpo" else None,
                "grpo_filter_groups": GRPO_FILTER_GROUPS if LOSS_TYPE == "grpo" else None,
                "grpo_mask_truncated": GRPO_MASK_TRUNCATED if LOSS_TYPE == "grpo" else None,
                "dataset": TRAIN_DATA_PATH,
                "hindsight_field": HINDSIGHT_FIELD,
            },
        )

        logger.info("Waiting for vLLM server...")
        wait_for_vllm()
        logger.info("Initializing vLLM weight transfer engine...")
        vllm_group = init_vllm_weight_engine(device)
        logger.info("vLLM weight engine ready.")

        needs_logprob_server = LOSS_TYPE == "sdft" or GRPO_KL_COEF > 0
        if needs_logprob_server:
            logger.info("Waiting for logprob server...")
            wait_for_logprob_server()
            # External frozen teacher (TEACHER_MODEL_PATH set): no NCCL
            # weight sync; the teacher keeps its downloaded weights.
            if not TEACHER_MODEL_PATH:
                logger.info("Initializing logprob weight transfer engine...")
                logprob_comm = init_logprob_weight_engine(device)
                logger.info("Logprob weight engine ready.")
        else:
            logger.info("GRPO KL disabled; skipping logprob server setup.")

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
        gen_times: deque | None = None

        if ASYNC_ROLLOUT and rank == 0:
            threading.excepthook = _crash_hard_on_thread_error
            rollout_queue = queue.Queue(
                maxsize=max(
                    N_ASYNC + world_size + 1,
                    GRAD_ACCUM_STEPS + 1 if LOSS_TYPE == "grpo" else 0,
                )
            )
            gen_times = deque()
            producer_thread = threading.Thread(
                target=_produce_streaming,
                args=(
                    rollout_queue,
                    data_iter,
                    success_cache,
                    gen_times,
                    producer_tokenizer,
                    steps_per_epoch,
                    world_size,
                ),
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
                accum_micro_step = 0
                for _group_step in range(local_group_steps):
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
                        meta_items = (
                            microbatch if LOSS_TYPE == "grpo" else [microbatch[0]]
                        )
                        for queued_item in meta_items:
                            step_metas.append(queued_item["_meta"])
                            policy_lags.append(
                                _OPTIMIZER_STEP - queued_item["policy_version"]
                            )
                    if LOSS_TYPE == "grpo":
                        first = rank * GRPO_GROUPS
                        rank_items = microbatch[first : first + GRPO_GROUPS]
                    else:
                        rank_items = [microbatch[rank]]
                    for item_data in rank_items:
                        result = _train_sample(
                            item_data,
                            accum_micro_step,
                            local_accum_steps,
                            fsdp_model,
                            model,
                            tokenizer,
                            vocab_size,
                            device,
                        )
                        accum_micro_step += 1
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
                    items = [next(data_iter) for _ in range(prompts_per_step)]
                    rollout_data, metas, batch_meta = produce(
                        items,
                        success_cache,
                        _OPTIMIZER_STEP,
                        producer_tokenizer,
                        world_size=world_size,
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
        avg_epoch_loss = (
            epoch_loss_sum / max(steps_per_epoch, 1)
            if LOSS_TYPE == "grpo"
            else epoch_loss_sum / max(epoch_samples, 1)
        )
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
