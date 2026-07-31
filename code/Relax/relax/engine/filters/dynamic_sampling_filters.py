# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = []
    for sample in samples:
        try:
            reward = sample.get_reward_value(args)
        except Exception:
            return DynamicFilterOutput(keep=False, reason="missing_reward")
        if reward is None:
            return DynamicFilterOutput(keep=False, reason="missing_reward")
        rewards.append(reward)

    if not rewards:
        return DynamicFilterOutput(keep=False, reason="empty_group")

    keep = torch.tensor(rewards, dtype=torch.float).std(unbiased=False) > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
