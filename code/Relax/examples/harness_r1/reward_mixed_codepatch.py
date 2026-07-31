# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Mixed ALFWorld/WebShop/DBBench reward dispatcher.

This module intentionally keeps the single-benchmark reward functions unchanged.
Each sample is routed by ``sample.metadata["benchmark"]`` and scored with the
corresponding same-batch rerun reward.
"""

from __future__ import annotations

import asyncio
from typing import Any

from relax.utils.types import Sample

from examples.harness_r1 import (
    reward_alfworld_patch,
    reward_dbbench_codepatch,
    reward_webshop_patch,
)


def _getattr_int(args: Any, name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default) or default)
    except (TypeError, ValueError):
        return default


def _benchmark(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw = str(metadata.get("benchmark") or "").strip().lower()
    if raw in {"alfworld", "webshop", "dbbench"}:
        return raw

    schema_style = str(metadata.get("schema_style") or "").strip().lower()
    if "alfworld" in schema_style:
        return "alfworld"
    if "webshop" in schema_style:
        return "webshop"
    if "dbbench" in schema_style:
        return "dbbench"

    prompt = sample.prompt
    if isinstance(prompt, list):
        text = "\n".join(str(item.get("content", "")) for item in prompt if isinstance(item, dict)).lower()
    else:
        text = str(prompt).lower()
    if "alfworld" in text:
        return "alfworld"
    if "webshop" in text:
        return "webshop"
    if "dbbench" in text:
        return "dbbench"
    return raw


def _no_patch_reward(sample: Sample, reason: str, detail: str = "") -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    baseline_pass = int(metadata.get("baseline_pass") or 0)
    batch_size = int(metadata.get("batch_size") or max(1, int(metadata.get("end", 0)) - int(metadata.get("start", 0))))
    baseline_rewards = metadata.get("baseline_rewards")
    baseline_average_reward = 0.0
    if isinstance(baseline_rewards, dict) and baseline_rewards:
        values = []
        for value in baseline_rewards.values():
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        if values:
            baseline_average_reward = sum(values) / len(values)
    else:
        baseline_average_reward = baseline_pass / max(1, batch_size)
    return {
        "score": 0.0,
        "delta_score": 0.0,
        "delta_pass": 0,
        "delta_pass_rate": 0.0,
        "delta_average_reward": 0.0,
        "baseline_pass": baseline_pass,
        "patched_pass": baseline_pass,
        "baseline_average_reward": baseline_average_reward,
        "patched_average_reward": baseline_average_reward,
        "batch_size": batch_size,
        "valid_patch": False,
        "eval_status": reason,
        "detail": detail[:1000],
        "benchmark": metadata.get("benchmark"),
    }


async def _score_one(args: Any, sample: Sample) -> dict[str, Any]:
    bench = _benchmark(sample)
    if bench == "alfworld":
        result = await reward_alfworld_patch.reward_func(args, sample)
    elif bench == "webshop":
        result = await reward_webshop_patch.reward_func(args, sample)
    elif bench == "dbbench":
        result = await reward_dbbench_codepatch.reward_func(args, sample)
    else:
        return _no_patch_reward(sample, "unknown_benchmark", f"benchmark={bench!r}")

    if isinstance(result, dict):
        result.setdefault("benchmark", bench)
        return result
    return _no_patch_reward(sample, "bad_reward_result", f"reward returned {type(result).__name__}")


async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
    del kwargs
    if not isinstance(samples, list):
        return await _score_one(args, samples)

    configured_limit = _getattr_int(args, "harness_r1_mixed_dispatch_concurrency", 0)
    harness_limit = _getattr_int(args, "harness_r1_reward_max_concurrency", 0)
    runtime_limit = _getattr_int(args, "reward_max_concurrency", 2)
    limit = max(1, configured_limit, harness_limit, runtime_limit)
    semaphore = asyncio.Semaphore(limit)

    async def run_one(sample: Sample) -> dict[str, Any]:
        async with semaphore:
            return await _score_one(args, sample)

    return await asyncio.gather(*(run_one(sample) for sample in samples))
