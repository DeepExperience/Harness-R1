# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from relax.utils import tracking_utils
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.timer import Timer


logger = get_logger(__name__)


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    seq_lens = getattr(timer_instance, "seq_lens", None)
    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None) and seq_lens:
        total_fwd_flops = compute_total_fwd_flops(seq_lens=seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(seq_lens) / log_dict["perf/actor_train_time"]
    elif ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        logger.warning(
            "Skip token-based perf metrics for rollout_id=%s because Timer.seq_lens is unavailable",
            rollout_id,
        )

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
