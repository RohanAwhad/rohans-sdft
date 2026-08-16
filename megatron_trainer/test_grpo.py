"""CPU numeric and dataflow checks for the GRPO implementation."""

import math
import unittest

import torch

from megatron_trainer.chunked_head import (
    ChunkedRowKL,
    ChunkedSelectedLogp,
    compute_grpo_loss_from_logps,
    make_grpo_processor,
)
from megatron_trainer.grpo import (
    compute_group_advantages,
    grpo_async_queue_order,
    grpo_kl_special_token_ids,
    grpo_rollouts_per_prompt,
    grpo_seed_offset,
    prepare_grpo_group,
    stamp_grpo_loss_scales,
)


def _rollout(
    reward: float,
    *,
    active_tokens: int = 2,
    finish_reason: str = "stop",
) -> dict:
    return {
        "reward": reward,
        "active_token_count": active_tokens,
        "finish_reason": finish_reason,
    }


class GroupMathTest(unittest.TestCase):
    def test_seed_offsets_are_unique_across_steps_and_retries(self) -> None:
        offsets = {
            grpo_seed_offset(
                rollout_batch=policy_version,
                attempt=attempt,
                prompt_slot=prompt_slot,
                group_index=group_index,
                prompts_per_step=2,
                rollouts_per_prompt=8,
                max_gen_batches=3,
            )
            for policy_version in range(2)
            for attempt in range(3)
            for prompt_slot in range(2)
            for group_index in range(8)
        }
        self.assertEqual(len(offsets), 2 * 3 * 2 * 8)

    def test_grpo_kl_masks_thinking_delimiters(self) -> None:
        class Tokenizer:
            all_special_ids = [1, 2]
            unk_token_id = 0

            def convert_tokens_to_ids(self, token: str) -> int:
                return {"<think>": 3, "</think>": 4}.get(token, 0)

        self.assertEqual(grpo_kl_special_token_ids(Tokenizer()), {1, 2, 3, 4})

    def test_mean_and_zscore_advantages(self) -> None:
        mean_advantages, drop_index = compute_group_advantages(
            [1.0, 0.0, 1.0, 0.0], "mean",
        )
        self.assertEqual(mean_advantages, [0.5, -0.5, 0.5, -0.5])
        self.assertIsNone(drop_index)

        zscore_advantages, _ = compute_group_advantages([1.0, 0.0], "zscore")
        expected = 0.5 / (0.5 + 1e-4)
        self.assertAlmostEqual(zscore_advantages[0], expected)
        self.assertAlmostEqual(zscore_advantages[1], -expected)

    def test_median_advantage_generates_one_extra_and_drops_one(self) -> None:
        self.assertEqual(grpo_rollouts_per_prompt(4, "median"), 5)
        group = prepare_grpo_group(
            [_rollout(reward) for reward in [0.0, 0.0, 1.0, 1.0, 1.0]],
            group_id=3,
            advantage_type="median",
            mask_truncated=True,
        )
        self.assertEqual(len(group), 4)
        self.assertEqual([item["reward"] for item in group], [0.0, 0.0, 1.0, 1.0])
        self.assertTrue(all(item["group_id"] == 3 for item in group))

    def test_truncated_rollout_has_zero_reward_and_zero_active_tokens(self) -> None:
        group = prepare_grpo_group(
            [
                _rollout(1.0, active_tokens=4, finish_reason="length"),
                _rollout(0.0),
                _rollout(1.0),
                _rollout(0.0),
            ],
            group_id=0,
            advantage_type="mean",
            mask_truncated=True,
        )
        self.assertEqual(group[0]["reward"], 0.0)
        self.assertEqual(group[0]["active_token_count"], 0)
        self.assertEqual(group[0]["group_token_count"], 6)

    def test_global_gpg_rescale_and_async_assignment(self) -> None:
        rewards = ([1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0])
        rollouts = [
            item
            for group_id, group_rewards in enumerate(rewards)
            for item in prepare_grpo_group(
                [_rollout(reward) for reward in group_rewards],
                group_id=group_id,
                advantage_type="mean",
                mask_truncated=True,
            )
        ]
        stamp_grpo_loss_scales(rollouts, world_size=2, group_size=2)

        expected_rescale = 4.0 / 3.0
        self.assertEqual(rollouts[0]["gpg_rescale"], expected_rescale)
        self.assertEqual(rollouts[0]["frac_reward_zero_std"], 0.25)
        self.assertAlmostEqual(rollouts[0]["grpo_loss_scale"], 1.0 / 3.0)
        self.assertEqual(rollouts[4]["gpg_rescale"], expected_rescale)
        self.assertAlmostEqual(rollouts[4]["grpo_loss_scale"], 1.0 / 3.0)

        queued = grpo_async_queue_order(rollouts, world_size=2, group_size=2)
        queued_group_ids = [queued[index]["group_id"] for index in range(0, 8, 2)]
        self.assertEqual(queued_group_ids, [0, 2, 1, 3])
        rank_0 = [queued[index]["group_id"] for index in (0, 4)]
        rank_1 = [queued[index]["group_id"] for index in (2, 6)]
        self.assertEqual(rank_0, [0, 1])
        self.assertEqual(rank_1, [2, 3])


class LossMathTest(unittest.TestCase):
    def test_grpo_processor_calls_head_once_and_backpropagates(self) -> None:
        torch.manual_seed(2)
        hidden = torch.randn(5, 1, 4, requires_grad=True)
        head_weight = torch.randn(6, 4, requires_grad=True)
        calls = 0

        def output_layer(hidden_rows, weight=None):
            nonlocal calls
            calls += 1
            return hidden_rows @ head_weight.t(), None

        processor = make_grpo_processor(
            prompt_len=2,
            completion_ids=[1, 2, 3],
            advantage=0.5,
            loss_scale=1.0,
            rollout_log_probs=None,
            old_logps_type="detached",
            clip_low=0.2,
            clip_high=0.28,
            is_c_max=3.0,
            is_mode="truncate",
            reference_log_probs=None,
            kl_coef=0.0,
            special_token_ids=set(),
            device=torch.device("cpu"),
            row_chunk=2,
        )
        loss, metrics = processor(hidden, output_layer)
        loss.backward()

        self.assertEqual(calls, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(hidden.grad.abs().sum().item(), 0.0)
        self.assertIn("grpo/policy_loss", metrics)

    def test_selected_logp_matches_autograd(self) -> None:
        torch.manual_seed(0)
        token_ids = torch.tensor([1, 3, 0])
        custom_logits = torch.randn(3, 5, requires_grad=True)
        reference_logits = custom_logits.detach().clone().requires_grad_(True)

        custom = ChunkedSelectedLogp.apply(custom_logits, token_ids).sum()
        reference = reference_logits.log_softmax(-1).gather(
            1, token_ids.unsqueeze(1),
        ).sum()
        custom.backward()
        reference.backward()

        torch.testing.assert_close(custom, reference)
        torch.testing.assert_close(custom_logits.grad, reference_logits.grad)

    def test_sdft_chunked_kl_regression(self) -> None:
        torch.manual_seed(1)
        token_ids = torch.tensor([1, 2, 3])
        teacher = torch.randn(3, 7).log_softmax(-1)
        custom_logits = torch.randn(3, 7, requires_grad=True)
        reference_logits = custom_logits.detach().clone().requires_grad_(True)

        custom_loss, _, _ = ChunkedRowKL.apply(custom_logits, teacher, token_ids)
        reference_logp = reference_logits.log_softmax(-1)
        reference_loss = (
            reference_logp.exp() * (reference_logp - teacher)
        ).sum(-1).sum()
        custom_loss.backward()
        reference_loss.backward()

        torch.testing.assert_close(custom_loss, reference_loss)
        torch.testing.assert_close(
            custom_logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-6,
        )

    def test_detached_old_logps_gradient_equals_plain_policy_gradient(self) -> None:
        advantage = 0.5
        policy = torch.tensor([-2.0, -1.0, -0.5], requires_grad=True)
        old = policy.detach().clone()
        loss, _ = compute_grpo_loss_from_logps(
            policy,
            old,
            advantage=advantage,
            clip_low=0.2,
            clip_high=0.28,
            is_c_max=3.0,
            is_mode="truncate",
        )
        loss.backward()
        grpo_gradient = policy.grad.clone()

        plain_policy = policy.detach().clone().requires_grad_(True)
        plain_loss = -(plain_policy * advantage).mean()
        plain_loss.backward()
        torch.testing.assert_close(grpo_gradient, plain_policy.grad, rtol=0, atol=0)

    def test_vllm_ratio_activates_clip_and_tis(self) -> None:
        policy = torch.full((2,), math.log(2.0), requires_grad=True)
        old = torch.zeros(2)
        loss, metrics = compute_grpo_loss_from_logps(
            policy,
            old,
            advantage=1.0,
            clip_low=0.2,
            clip_high=0.28,
            is_c_max=1.5,
            is_mode="truncate",
        )
        self.assertAlmostEqual(loss.item(), -1.28 * 1.5, places=6)
        self.assertEqual(metrics["grpo/clip_frac"].item(), 1.0)
        self.assertAlmostEqual(metrics["grpo/is_ratio"].item(), 2.0, places=6)
        self.assertEqual(metrics["grpo/is_clip_frac"].item(), 1.0)

        masked_loss, _ = compute_grpo_loss_from_logps(
            policy,
            old,
            advantage=1.0,
            clip_low=0.2,
            clip_high=0.28,
            is_c_max=1.5,
            is_mode="mask",
        )
        self.assertEqual(masked_loss.item(), 0.0)

    def test_k3_pp_clamp_and_special_token_mask(self) -> None:
        policy = torch.zeros(1, requires_grad=True)
        old = torch.zeros(1)
        ref = torch.full((1,), 25.0)
        loss, metrics = compute_grpo_loss_from_logps(
            policy,
            old,
            advantage=0.0,
            clip_low=0.2,
            clip_high=0.28,
            is_c_max=3.0,
            is_mode="truncate",
            ref_logp=ref,
            kl_coef=1.0,
            special_token_mask=torch.tensor([False]),
        )
        expected = math.exp(20.0) - 21.0
        self.assertAlmostEqual(loss.item() / expected, 1.0, places=6)
        self.assertAlmostEqual(metrics["grpo/kl"].item() / expected, 1.0, places=6)

        masked_loss, masked_metrics = compute_grpo_loss_from_logps(
            policy,
            old,
            advantage=0.0,
            clip_low=0.2,
            clip_high=0.28,
            is_c_max=3.0,
            is_mode="truncate",
            ref_logp=ref,
            kl_coef=1.0,
            special_token_mask=torch.tensor([True]),
        )
        self.assertEqual(masked_loss.item(), 0.0)
        self.assertEqual(masked_metrics["grpo/kl"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
