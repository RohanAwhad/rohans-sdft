"""Pure GRPO group transformations used by rollout and loss plumbing."""

import math
import statistics


def grpo_kl_special_token_ids(tokenizer) -> set[int]:
    """Chat, EOS, and Qwen thinking delimiters excluded from reference KL."""
    token_ids = set(tokenizer.all_special_ids)
    for token in ("<think>", "</think>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            token_ids.add(token_id)
    return token_ids


def grpo_rollouts_per_prompt(group_size: int, advantage_type: str) -> int:
    """Number of generated rollouts; median/MAD drops one before training."""
    return group_size + 1 if advantage_type == "median" else group_size


def grpo_seed_offset(
    *,
    rollout_batch: int,
    attempt: int,
    prompt_slot: int,
    group_index: int,
    prompts_per_step: int,
    rollouts_per_prompt: int,
    max_gen_batches: int,
) -> int:
    """Unique deterministic seed offset across rollout and repair batches."""
    batch_width = prompts_per_step * rollouts_per_prompt
    return (
        (rollout_batch * max_gen_batches + attempt) * batch_width
        + prompt_slot * rollouts_per_prompt
        + group_index
    )


def compute_group_advantages(
    rewards: list[float],
    advantage_type: str,
) -> tuple[list[float], int | None]:
    """Compute group-relative advantages and an optional rollout index to drop."""
    if len(rewards) < 2:
        raise ValueError("GRPO advantage computation requires at least 2 rewards")
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError(f"GRPO rewards must be finite, got {rewards}")

    if advantage_type == "mean":
        center = sum(rewards) / len(rewards)
        return [reward - center for reward in rewards], None

    if advantage_type == "zscore":
        center = sum(rewards) / len(rewards)
        variance = sum((reward - center) ** 2 for reward in rewards) / len(rewards)
        scale = math.sqrt(variance) + 1e-4
        return [(reward - center) / scale for reward in rewards], None

    if advantage_type == "median":
        center = statistics.median(rewards)
        scale = statistics.median(abs(reward - center) for reward in rewards) + 1e-4
        advantages = [(reward - center) / scale for reward in rewards]
        drop_index = min(
            range(len(rewards)),
            key=lambda index: (abs(rewards[index] - center), index),
        )
        return advantages, drop_index

    raise ValueError(f"Unknown GRPO advantage type: {advantage_type!r}")


def prepare_grpo_group(
    rollouts: list[dict],
    *,
    group_id: int,
    advantage_type: str,
    mask_truncated: bool,
) -> list[dict]:
    """Attach reward/advantage/group state and drop the MC-GRPO median sample."""
    rewards: list[float] = []
    for rollout in rollouts:
        reward = rollout.get("reward")
        if reward is None:
            raise ValueError(
                "GRPO rollout has no reward; use ENV_TYPE=api_adapter or "
                "ENV_TYPE=rag with HINDSIGHT_FIELD=online_feedback"
            )
        truncated = rollout.get("finish_reason") == "length"
        rewards.append(0.0 if mask_truncated and truncated else float(reward))

    advantages, drop_index = compute_group_advantages(rewards, advantage_type)
    prepared: list[dict] = []
    for index, (rollout, reward, advantage) in enumerate(
        zip(rollouts, rewards, advantages)
    ):
        if index == drop_index:
            continue
        item = dict(rollout)
        truncated = item.get("finish_reason") == "length"
        item.update(
            {
                "group_id": group_id,
                "group_index": len(prepared),
                "reward": reward,
                "advantage": advantage,
                "truncated": truncated,
            }
        )
        if mask_truncated and truncated:
            item["active_token_count"] = 0
        prepared.append(item)

    is_degenerate = max(rewards) == min(rewards)
    group_token_count = sum(item["active_token_count"] for item in prepared)
    prepared_advantages = [item["advantage"] for item in prepared]
    advantage_mean = sum(prepared_advantages) / len(prepared_advantages)
    advantage_std = math.sqrt(
        sum((advantage - advantage_mean) ** 2 for advantage in prepared_advantages)
        / len(prepared_advantages)
    )
    for item in prepared:
        item["group_size"] = len(prepared)
        item["group_degenerate"] = is_degenerate
        item["group_token_count"] = group_token_count
        item["group_advantage_std"] = advantage_std
    return prepared


def stamp_grpo_loss_scales(
    rollouts: list[dict],
    *,
    world_size: int,
    group_size: int,
) -> None:
    """Stamp DAPO token normalization and global GPG rescaling in place."""
    if len(rollouts) % group_size != 0:
        raise ValueError(
            f"GRPO rollout count {len(rollouts)} is not divisible by G={group_size}"
        )
    num_groups = len(rollouts) // group_size
    if num_groups % world_size != 0:
        raise ValueError(
            f"GRPO group count {num_groups} is not divisible by world_size={world_size}"
        )

    groups_per_rank = num_groups // world_size
    groups = [
        rollouts[group * group_size : (group + 1) * group_size]
        for group in range(num_groups)
    ]
    nondegenerate = sum(not group[0]["group_degenerate"] for group in groups)
    gpg_rescale = num_groups / nondegenerate if nondegenerate else 1.0
    zero_std_fraction = 1.0 - nondegenerate / num_groups

    for rank in range(world_size):
        first_group = rank * groups_per_rank
        rank_groups = groups[first_group : first_group + groups_per_rank]

        for group in rank_groups:
            token_denominator = max(group[0]["group_token_count"], 1)
            for item in group:
                item["grpo_loss_scale"] = (
                    item["active_token_count"]
                    / token_denominator
                    / groups_per_rank
                    * gpg_rescale
                )
                item["gpg_rescale"] = gpg_rescale
                item["frac_reward_zero_std"] = zero_std_fraction


def grpo_async_queue_order(
    rollouts: list[dict],
    *,
    world_size: int,
    group_size: int,
) -> list[dict]:
    """Column-major queue order preserving sync-mode rank group assignment."""
    groups = [
        rollouts[index : index + group_size]
        for index in range(0, len(rollouts), group_size)
    ]
    if not groups or any(len(group) != group_size for group in groups):
        raise ValueError("GRPO async ordering requires complete groups")
    if len(groups) % world_size != 0:
        raise ValueError(
            f"GRPO group count {len(groups)} is not divisible by world_size={world_size}"
        )
    groups_per_rank = len(groups) // world_size
    return [
        item
        for local_group in range(groups_per_rank)
        for rank in range(world_size)
        for item in groups[rank * groups_per_rank + local_group]
    ]
