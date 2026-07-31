# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Same-batch WebShop delta-pass@1 reward for Harness-R1 patches."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from relax.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTBENCH_SCRIPTS = REPO_ROOT / "code/life-harness/AgentBench/scripts"
sys.path.insert(0, str(AGENTBENCH_SCRIPTS))

from harness_r1_patch import (  # noqa: E402
    PatchValidationError,
    extract_json_object,
    extract_prefilled_think_patch_json_object,
    extract_think_patch_json_object,
    normalize_patch,
    require_code_hook_only_patch,
)
from harness_r1_webshop_identity import (  # noqa: E402
    WebShopIdentityError,
    assert_matching_task_hashes,
    task_manifest_hashes_from_rows,
    validate_identity_metadata,
)


REWARD_CACHE_PROTOCOL_VERSION = "webshop_delta_pass_identity_v3_20260715"
LOCAL_JAVA_HOME = REPO_ROOT / ".local/jdk"
DEFAULT_JAVA_HOME = Path("/usr/lib/jvm/java-11-openjdk-amd64")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed
    raise TypeError(f"expected dict or JSON object string, got {type(value).__name__}")


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _finish_timing(timing: dict[str, Any], t0: float) -> dict[str, Any]:
    timing["total_sec"] = round(time.monotonic() - t0, 3)
    timing["end_time"] = _now_text()
    return timing


def _attach_timing(result: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    result["timing"] = timing
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _reward_of(row: dict[str, Any]) -> float:
    result = ((row.get("output") or {}).get("result") or {})
    try:
        return float(result.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_java_home() -> Path | None:
    candidates = [
        os.environ.get("HARNESS_R1_JAVA_HOME"),
        os.environ.get("JAVA_HOME"),
        str(LOCAL_JAVA_HOME),
        str(DEFAULT_JAVA_HOME),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        java_home = Path(candidate)
        if (java_home / "bin/javac").exists():
            return java_home
    return None


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    no_proxy = (
        env.get("HARNESS_R1_NO_PROXY")
        or env.get("NO_PROXY")
        or "127.0.0.1,localhost,0.0.0.0"
    )
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    java_home = _resolve_java_home()
    if java_home is not None:
        env["JAVA_HOME"] = str(java_home)
        env["HARNESS_R1_JAVA_HOME"] = str(java_home)
        java_bin = str(java_home / "bin")
        path = env.get("PATH", "")
        if java_bin not in path.split(":"):
            env["PATH"] = f"{java_bin}:{path}" if path else java_bin
    return env


def _port_is_free(port: int) -> bool:
    if port <= 1024 or port >= 65535:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ephemeral_port_range() -> tuple[int, int] | None:
    path = Path("/proc/sys/net/ipv4/ip_local_port_range")
    try:
        low_text, high_text = path.read_text(encoding="utf-8").split()
        return int(low_text), int(high_text)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _port_is_ephemeral(port: int) -> bool:
    port_range = _ephemeral_port_range()
    if port_range is None:
        return False
    low, high = port_range
    return low <= port <= high


def _ports_are_safe_candidates(ports: list[int], *, avoid_ephemeral: bool) -> bool:
    if any(port <= 1024 or port >= 65535 for port in ports):
        return False
    if avoid_ephemeral and any(_port_is_ephemeral(port) for port in ports):
        return False
    return True


def _release_port_locks(handles: list[Any]) -> None:
    for handle in handles:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass


def _try_lock_ports(ports: list[int]) -> list[Any] | None:
    lock_dir = Path(os.environ.get("HARNESS_R1_PORT_LOCK_DIR", "/tmp/harness_r1_port_locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles: list[Any] = []
    for port in sorted(set(ports)):
        handle = (lock_dir / f"{port}.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            _release_port_locks(handles)
            return None
        handles.append(handle)
    return handles


def _select_free_port_pair(
    *,
    controller_base: int,
    worker_base: int,
    seed: int,
    num_workers: int = 1,
    span: int = 40000,
    avoid_ephemeral: bool = False,
) -> tuple[int, int, list[Any]]:
    """Pick and reserve controller/worker ports for one WebShop reward subprocess."""
    span = max(1, min(int(span), 60000))
    start = seed % span
    for probe in range(span):
        offset = (start + probe) % span
        controller_port = controller_base + offset
        worker_port_base = worker_base + offset
        ports = [controller_port] + [worker_port_base + idx for idx in range(num_workers)]
        if len(set(ports)) != len(ports):
            continue
        if not _ports_are_safe_candidates(ports, avoid_ephemeral=avoid_ephemeral):
            continue
        port_locks = _try_lock_ports(ports)
        if port_locks is None:
            continue
        if all(_port_is_free(port) for port in ports):
            return controller_port, worker_port_base, port_locks
        _release_port_locks(port_locks)
    raise RuntimeError(
        "could not find free WebShop controller/worker ports "
        f"from controller_base={controller_base}, worker_base={worker_base}, span={span}"
    )


def _retryable_eval_failure(log_path: Path, output_root: Path) -> bool:
    needles = (
        "address already in use",
        "worker did not register",
        "controller did not start",
        "connection refused",
    )
    texts: list[str] = []
    for path in (
        log_path,
        output_root / "controller.log",
        output_root / "webshop/batch_000/status.json",
        output_root / "webshop/batch_000/logs/worker.log",
        output_root / "webshop/batch_000/logs/assigner.log",
    ):
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace").lower())
        except OSError:
            pass
    joined = "\n".join(texts)
    return any(needle in joined for needle in needles)


def _no_patch_reward(metadata: dict[str, Any], reason: str, detail: str = "") -> dict[str, Any]:
    baseline_pass = int(metadata.get("baseline_pass") or 0)
    batch_size = int(metadata.get("batch_size") or max(1, int(metadata.get("end", 0)) - int(metadata.get("start", 0))))
    baseline_average_reward = _baseline_average_reward(metadata, batch_size=batch_size)
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
    }


def _response_protocol(args: Any) -> str:
    protocol = str(getattr(args, "harness_r1_response_protocol", "") or "").strip()
    if protocol:
        return protocol
    if bool(getattr(args, "harness_r1_require_think_patch", False)):
        return "full_think_patch"
    return "json_patch"


def _parse_patch(
    response: str,
    *,
    response_protocol: str = "json_patch",
    require_code_hook_only: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    try:
        if response_protocol == "full_think_patch":
            raw = extract_think_patch_json_object(response or "")
        elif response_protocol == "prefill_think_patch":
            raw = extract_prefilled_think_patch_json_object(response or "")
        elif response_protocol == "json_patch":
            raw = extract_json_object(response or "")
        else:
            raise PatchValidationError(f"unsupported response protocol: {response_protocol!r}")
        patch = normalize_patch(raw, bench="webshop")
        if require_code_hook_only:
            require_code_hook_only_patch(patch)
        return patch, ""
    except (json.JSONDecodeError, PatchValidationError, TypeError, ValueError) as exc:
        return None, repr(exc)


def _reward_metric(args: Any) -> str:
    metric = str(getattr(args, "harness_r1_reward_metric", "delta_pass_rate") or "delta_pass_rate").strip()
    aliases = {
        "pass": "delta_pass_rate",
        "pass_at_1": "delta_pass_rate",
        "delta_pass": "delta_pass_rate",
        "continuous": "delta_average_reward",
        "average_reward": "delta_average_reward",
        "delta_reward": "delta_average_reward",
    }
    metric = aliases.get(metric, metric)
    if metric not in {"delta_pass_rate", "delta_average_reward"}:
        raise ValueError(
            f"unsupported harness_r1_reward_metric={metric!r}; "
            "expected delta_pass_rate or delta_average_reward"
        )
    return metric


def _webshop_runtime_incompatible_reason(patch: dict[str, Any]) -> str:
    """Return why a valid patch is a WebShop runtime no-op under the overlay executor."""
    for idx, action in enumerate(patch.get("actions") or []):
        action_type = action.get("type")
        if action_type == "add_guard_rule":
            trigger = action.get("trigger")
            effect_kind = (action.get("effect") or {}).get("kind")
            if trigger != "before_action":
                return f"action {idx} guard trigger {trigger!r} is not executed by WebShop overlay guards"
            if effect_kind not in {"block_and_prompt", "force_action", "rewrite_action"}:
                return f"action {idx} guard effect {effect_kind!r} is not executed by WebShop overlay guards"
        elif action_type == "add_recovery_rule":
            trigger = action.get("trigger")
            effect_kind = (action.get("effect") or {}).get("kind")
            if trigger != "post_step":
                return f"action {idx} recovery trigger {trigger!r} is not executed by WebShop overlay recovery"
            if effect_kind not in {"inject_hint", "force_action"}:
                return f"action {idx} recovery effect {effect_kind!r} is not executed by WebShop overlay recovery"
        elif action_type == "add_code_hook":
            if action.get("hook") not in {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}:
                return f"action {idx} code hook {action.get('hook')!r} is not executed by WebShop code hooks"
    return ""


def _patch_hash(
    batch_tag: str,
    patch: dict[str, Any],
    *,
    valid_bonus: float = 0.0,
    reward_context: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "cache_protocol_version": REWARD_CACHE_PROTOCOL_VERSION,
            "batch_tag": batch_tag,
            "patch": patch,
            "valid_bonus": float(valid_bonus),
            "reward_context": reward_context or {},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _reward_context(
    args: Any,
    metadata: dict[str, Any],
    patch: dict[str, Any],
    *,
    identity: dict[str, Any] | None,
    target_urls: list[str],
    target_model: str,
    agent_name: str,
    start: int,
    end: int,
    batch_size: int,
    baseline_pass: int,
    threshold: float,
    valid_bonus: float,
    rollout_concurrency: int,
    startup_timeout: int,
    rollout_timeout: int,
    reward_timeout: int,
    rollout_max_tokens: int,
    rollout_http_timeout: int,
    webshop_rounds: int,
    rollout_tool_choice: str,
    rollout_chat_template_kwargs: dict[str, Any],
    rollout_system_prefix: str,
    auto_retry: bool,
    require_think_patch: bool,
    response_protocol: str,
    reject_runtime_noop_patch: bool,
    require_code_hook_only: bool,
    reward_metric: str,
    retry_incomplete_eval: bool,
    eval_max_attempts: int,
) -> dict[str, Any]:
    """Return all protocol knobs that can change cached reward semantics."""
    return {
        "reward_func": f"reward_webshop_patch.{reward_metric}",
        "reward_func_version": str(
            getattr(args, "harness_r1_reward_cache_protocol", REWARD_CACHE_PROTOCOL_VERSION)
        ),
        "benchmark": "webshop",
        "patch_schema_version": str(patch.get("schema_version") or "unknown"),
        "target": {
            "model": target_model,
            "urls": target_urls,
            "agent_name": agent_name,
        },
        "batch": {
            "start": start,
            "end": end,
            "batch_size": batch_size,
            "baseline_pass": baseline_pass,
            "metadata_target_model": str(metadata.get("target_model") or ""),
            "metadata_prompt_version": str(metadata.get("prompt_version") or ""),
            "metadata_source_rollout_dir": str(metadata.get("source_rollout_dir") or ""),
            "webshop_identity_protocol": (identity or {}).get("protocol"),
            "webshop_goal_seed": (identity or {}).get("goal_seed"),
            "webshop_task_manifest_sha256": (identity or {}).get("sha256"),
        },
        "reward_policy": {
            "threshold": threshold,
            "valid_bonus": valid_bonus,
            "invalid_patch": "no_patch",
            "metric": reward_metric,
            "require_think_patch": require_think_patch,
            "response_protocol": response_protocol,
            "reject_runtime_noop_patch": reject_runtime_noop_patch,
            "require_code_hook_only": require_code_hook_only,
            "retry_incomplete_eval": retry_incomplete_eval,
        },
        "rollout": {
            "temperature": 0.0,
            "max_tokens": rollout_max_tokens,
            "http_timeout": rollout_http_timeout,
            "tool_choice": rollout_tool_choice,
            "chat_template_kwargs": rollout_chat_template_kwargs,
            "system_prefix": rollout_system_prefix,
            "webshop_rounds": webshop_rounds,
            "concurrency": rollout_concurrency,
            "auto_retry": auto_retry,
            "startup_timeout": startup_timeout,
            "rollout_timeout": rollout_timeout,
            "reward_timeout": reward_timeout,
        },
        "infra": {
            "controller_port_base": int(getattr(args, "harness_r1_controller_port_base", 21000)),
            "worker_port_base": int(getattr(args, "harness_r1_worker_port_base", 33000)),
            "port_probe_span": int(getattr(args, "harness_r1_port_probe_span", 40000)),
            "avoid_ephemeral_ports": _as_bool(getattr(args, "harness_r1_avoid_ephemeral_ports", False)),
            "eval_max_attempts": eval_max_attempts,
            "eval_retry_on_infra_failure": int(getattr(args, "harness_r1_eval_retry_on_infra_failure", 0) or 0),
        },
    }


def _patched_pass(
    eval_output_root: Path,
    threshold: float,
    *,
    goal_seed: int,
) -> tuple[int, int, dict[int, float], dict[int, str]]:
    rewards: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_output_root.glob("webshop/batch_*/rollout/*/*/runs.jsonl")):
        path_rows = _read_jsonl(path)
        rows.extend(path_rows)
        for row in path_rows:
            idx = row.get("index")
            if isinstance(idx, int):
                rewards[idx] = _reward_of(row)
    task_hashes = task_manifest_hashes_from_rows(rows, goal_seed=goal_seed)
    return (
        sum(value >= threshold for value in rewards.values()),
        len(rewards),
        rewards,
        task_hashes,
    )


def _average_reward(rewards: dict[int, float] | dict[str, float], *, batch_size: int | None = None) -> float:
    values = [float(value) for value in rewards.values()]
    if values:
        return sum(values) / len(values)
    if batch_size:
        return 0.0
    return 0.0


def _baseline_rewards(metadata: dict[str, Any]) -> dict[int, float]:
    raw = metadata.get("baseline_rewards")
    if not isinstance(raw, dict):
        return {}
    rewards: dict[int, float] = {}
    for key, value in raw.items():
        try:
            rewards[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return rewards


def _baseline_average_reward(metadata: dict[str, Any], *, batch_size: int) -> float:
    rewards = _baseline_rewards(metadata)
    if rewards:
        return _average_reward(rewards)
    baseline_pass = int(metadata.get("baseline_pass") or 0)
    return baseline_pass / max(1, batch_size)


def _wait_for_cache_or_lock(lock_path: Path, reward_path: Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if reward_path.exists():
                return False
            time.sleep(2)
    return False


def _score_sync(args: Any, sample: Sample) -> dict[str, Any]:
    t0 = time.monotonic()
    timing: dict[str, Any] = {"start_time": _now_text(), "cache_hit": False}
    port_locks: list[Any] = []
    t_parse = time.monotonic()
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    require_identity = _as_bool(
        getattr(args, "harness_r1_require_webshop_identity", True),
        default=True,
    )
    identity: dict[str, Any] | None = None
    if require_identity or metadata.get("webshop_identity_protocol"):
        identity = validate_identity_metadata(metadata)
    configured_goal_seed = getattr(args, "harness_r1_webshop_goal_seed", None)
    if identity is not None and configured_goal_seed is not None:
        if int(configured_goal_seed) != int(identity["goal_seed"]):
            raise WebShopIdentityError(
                "configured webshop goal seed does not match baseline identity: "
                f"{configured_goal_seed} != {identity['goal_seed']}"
            )
    require_think_patch = bool(getattr(args, "harness_r1_require_think_patch", False))
    response_protocol = _response_protocol(args)
    require_code_hook_only = bool(getattr(args, "harness_r1_require_code_hook_only", False))
    patch, parse_error = _parse_patch(
        sample.response,
        response_protocol=response_protocol,
        require_code_hook_only=require_code_hook_only,
    )
    timing["parse_sec"] = round(time.monotonic() - t_parse, 3)
    if patch is None:
        return _attach_timing(
            _no_patch_reward(metadata, "invalid_patch_treated_as_no_patch", parse_error),
            _finish_timing(timing, t0),
        )
    reject_runtime_noop_patch = bool(getattr(args, "harness_r1_reject_runtime_noop_patch", False))
    if reject_runtime_noop_patch:
        incompatible_reason = _webshop_runtime_incompatible_reason(patch)
        if incompatible_reason:
            return _attach_timing(
                _no_patch_reward(metadata, "runtime_noop_patch_treated_as_no_patch", incompatible_reason),
                _finish_timing(timing, t0),
            )

    batch_tag = str(metadata.get("batch_tag") or f"{metadata.get('start', 'unknown')}_{metadata.get('end', 'unknown')}")
    valid_bonus = float(getattr(args, "harness_r1_valid_bonus", 0.0) or 0.0)
    target_urls = _as_list(getattr(args, "harness_r1_target_urls", None)) or [
        str(getattr(args, "harness_r1_target_url", "http://127.0.0.1:8110/v1"))
    ]
    target_model = str(getattr(args, "harness_r1_target_model", metadata.get("target_model", "Qwen2.5-7B-Instruct")))
    agent_name = str(getattr(args, "harness_r1_agent_name", metadata.get("target_agent_name", "qwen25-7b-rollout")))
    start = int(metadata["start"])
    end = int(metadata["end"])
    batch_size = int(metadata.get("batch_size") or (end - start))
    baseline_pass = int(metadata.get("baseline_pass") or 0)
    if identity is not None and (start, end) != (identity["start"], identity["end"]):
        raise WebShopIdentityError(
            f"reward batch range [{start}, {end}) does not match identity "
            f"[{identity['start']}, {identity['end']})"
        )
    webshop_goal_seed = int(
        identity["goal_seed"] if identity is not None else metadata.get("webshop_goal_seed", 233)
    )
    threshold_value = metadata.get("reward_threshold")
    threshold = float(
        threshold_value if threshold_value is not None else getattr(args, "harness_r1_reward_threshold", 1.0)
    )
    reward_metric = _reward_metric(args)
    rollout_concurrency = int(getattr(args, "harness_r1_rollout_concurrency", 4))
    startup_timeout = int(getattr(args, "harness_r1_startup_timeout", 300))
    rollout_timeout = int(getattr(args, "harness_r1_rollout_timeout", 1800))
    reward_timeout = int(getattr(args, "harness_r1_reward_timeout", 2400))
    rollout_max_tokens = int(getattr(args, "harness_r1_rollout_max_tokens", 2048))
    rollout_http_timeout = int(getattr(args, "harness_r1_rollout_http_timeout", 120))
    webshop_rounds = int(getattr(args, "harness_r1_webshop_rounds", 20))
    rollout_tool_choice = str(getattr(args, "harness_r1_rollout_tool_choice", "") or "")
    rollout_chat_template_kwargs = _as_dict(getattr(args, "harness_r1_rollout_chat_template_kwargs", None))
    rollout_system_prefix = str(getattr(args, "harness_r1_rollout_system_prefix", "") or "")
    auto_retry = bool(getattr(args, "harness_r1_auto_retry", False))
    retry_incomplete_eval = _as_bool(getattr(args, "harness_r1_retry_incomplete_eval", False))
    retry_count = int(getattr(args, "harness_r1_eval_retry_on_infra_failure", 0) or 0)
    eval_max_attempts = max(
        1,
        int(getattr(args, "harness_r1_eval_max_attempts", 2) or 1),
        retry_count + 1,
    )
    reward_context = _reward_context(
        args,
        metadata,
        patch,
        identity=identity,
        target_urls=target_urls,
        target_model=target_model,
        agent_name=agent_name,
        start=start,
        end=end,
        batch_size=batch_size,
        baseline_pass=baseline_pass,
        threshold=threshold,
        valid_bonus=valid_bonus,
        rollout_concurrency=rollout_concurrency,
        startup_timeout=startup_timeout,
        rollout_timeout=rollout_timeout,
        rollout_http_timeout=rollout_http_timeout,
        reward_timeout=reward_timeout,
        rollout_max_tokens=rollout_max_tokens,
        webshop_rounds=webshop_rounds,
        rollout_tool_choice=rollout_tool_choice,
        rollout_chat_template_kwargs=rollout_chat_template_kwargs,
        rollout_system_prefix=rollout_system_prefix,
        auto_retry=auto_retry,
        require_think_patch=require_think_patch,
        response_protocol=response_protocol,
        reject_runtime_noop_patch=reject_runtime_noop_patch,
        require_code_hook_only=require_code_hook_only,
        reward_metric=reward_metric,
        retry_incomplete_eval=retry_incomplete_eval,
        eval_max_attempts=eval_max_attempts,
    )
    eval_key = _patch_hash(batch_tag, patch, valid_bonus=valid_bonus, reward_context=reward_context)
    eval_root = Path(getattr(args, "harness_r1_eval_root", REPO_ROOT / "outputs/relax_harness_r1/reward_cache"))
    eval_dir = eval_root / batch_tag / eval_key
    reward_path = eval_dir / "reward.json"
    lock_path = eval_dir / ".lock"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if reward_path.exists():
        result = json.loads(reward_path.read_text(encoding="utf-8"))
        result.setdefault("timing", {})["cache_hit"] = True
        result["timing"]["cache_return_sec"] = round(time.monotonic() - t0, 3)
        return result

    t_lock = time.monotonic()
    lock_owner = _wait_for_cache_or_lock(
        lock_path,
        reward_path,
        float(getattr(args, "harness_r1_cache_lock_timeout", 7200)),
    )
    timing["lock_wait_sec"] = round(time.monotonic() - t_lock, 3)
    if not lock_owner and reward_path.exists():
        result = json.loads(reward_path.read_text(encoding="utf-8"))
        result.setdefault("timing", {})["cache_hit"] = True
        result["timing"]["cache_return_sec"] = round(time.monotonic() - t0, 3)
        return result
    if not lock_owner:
        return _attach_timing(
            _no_patch_reward(metadata, "cache_lock_timeout"),
            _finish_timing(timing, t0),
        )

    try:
        if reward_path.exists():
            result = json.loads(reward_path.read_text(encoding="utf-8"))
            result.setdefault("timing", {})["cache_hit"] = True
            result["timing"]["cache_return_sec"] = round(time.monotonic() - t0, 3)
            return result

        t_prepare = time.monotonic()
        java_home = _resolve_java_home()
        if java_home is None:
            result = _no_patch_reward(
                metadata,
                "java_home_missing",
                "WebShop reward requires a JDK with bin/javac. Set HARNESS_R1_JAVA_HOME or install .local/jdk.",
            )
            result.update({"valid_patch": True})
            _attach_timing(result, _finish_timing(timing, t0))
            _json_dump(reward_path, result)
            return result
        timing["java_home"] = str(java_home)

        patch_path = eval_dir / "patch.json"
        _json_dump(patch_path, patch)
        _json_dump(
            eval_dir / "cache_key_payload.json",
            {
                "cache_protocol_version": REWARD_CACHE_PROTOCOL_VERSION,
                "eval_key": eval_key,
                "batch_tag": batch_tag,
                "patch": patch,
                "valid_bonus": valid_bonus,
                "reward_context": reward_context,
            },
        )

        target_url = target_urls[int(eval_key[:6], 16) % len(target_urls)]
        run_id = f"relax_webshop_{batch_tag}_{eval_key}"
        script = REPO_ROOT / "code/life-harness/AgentBench/scripts/harness_r1_batch_debug.py"
        agentbench_dir = Path(
            getattr(
                args,
                "harness_r1_agentbench_dir",
                REPO_ROOT / "external/harness_evolution/life-harness/AgentBench",
            )
        )
        default_agentbench_python = (
            agentbench_dir / ".venv/bin/python"
            if (agentbench_dir / ".venv/bin/python").exists()
            else REPO_ROOT / "external/harness_evolution/life-harness/AgentBench/.venv/bin/python"
        )
        agentbench_python = Path(
            getattr(args, "harness_r1_agentbench_python", default_agentbench_python)
        )
        webshop_worker_python = str(getattr(args, "harness_r1_webshop_worker_python", "") or "").strip()

        eval_dir.mkdir(parents=True, exist_ok=True)
        timing["prepare_eval_sec"] = round(time.monotonic() - t_prepare, 3)
        timing["target_url"] = target_url
        timing["batch_tag"] = batch_tag
        timing["eval_key"] = eval_key
        max_eval_attempts = eval_max_attempts
        port_probe_span = int(getattr(args, "harness_r1_port_probe_span", 40000))
        avoid_ephemeral_ports = _as_bool(getattr(args, "harness_r1_avoid_ephemeral_ports", False))

        proc: subprocess.CompletedProcess[str] | None = None
        patched_pass = 0
        patched_n = 0
        patched_rewards: dict[int, float] = {}
        patched_task_hashes: dict[int, str] = {}
        output_root = eval_dir / "batch_debug"
        log_path = eval_dir / "eval.log"
        eval_attempts: list[dict[str, Any]] = []
        t_eval_all = time.monotonic()
        for attempt in range(max_eval_attempts):
            if port_locks:
                _release_port_locks(port_locks)
                port_locks = []
            controller_port, worker_port_base, port_locks = _select_free_port_pair(
                controller_base=int(getattr(args, "harness_r1_controller_port_base", 21000)),
                worker_base=int(getattr(args, "harness_r1_worker_port_base", 33000)),
                seed=int(eval_key[:8], 16) + attempt * 104729,
                num_workers=1,
                span=port_probe_span,
                avoid_ephemeral=avoid_ephemeral_ports,
            )
            output_root = eval_dir / ("batch_debug" if attempt == 0 else f"batch_debug_retry_{attempt}")
            log_path = eval_dir / ("eval.log" if attempt == 0 else f"eval_retry_{attempt}.log")
            cmd = [
                str(agentbench_python),
                str(script),
                "--agentbench-dir",
                str(agentbench_dir),
                "--agentbench-python",
                str(agentbench_python),
                "--bench",
                "webshop",
                "--run-id",
                f"{run_id}_a{attempt}",
                "--start",
                str(start),
                "--batch-size",
                str(batch_size),
                "--num-batches",
                "1",
                "--max-parallel-batches",
                "1",
                "--rollout-concurrency",
                str(rollout_concurrency),
                "--skip-adb",
                "--controller-port",
                str(controller_port),
                "--worker-port-base",
                str(worker_port_base),
                "--startup-timeout",
                str(startup_timeout),
                "--rollout-timeout",
                str(rollout_timeout),
                "--rollout-base-url",
                target_url,
                "--rollout-model",
                target_model,
                "--agent-name",
                agent_name,
                "--rollout-temperature",
                "0.0",
                "--rollout-max-tokens",
                str(rollout_max_tokens),
                "--rollout-http-timeout",
                str(rollout_http_timeout),
                "--webshop-rounds",
                str(webshop_rounds),
                "--webshop-goal-seed",
                str(webshop_goal_seed),
                "--output-root",
                str(output_root),
                "--harness-patch",
                str(patch_path),
            ]
            if rollout_tool_choice:
                cmd.extend(["--rollout-tool-choice", rollout_tool_choice])
            if rollout_chat_template_kwargs:
                cmd.extend(
                    [
                        "--rollout-chat-template-kwargs",
                        json.dumps(rollout_chat_template_kwargs, ensure_ascii=False),
                    ]
                )
            if rollout_system_prefix:
                cmd.extend(["--rollout-system-prefix", rollout_system_prefix])
            if webshop_worker_python:
                cmd.extend(["--webshop-worker-python", webshop_worker_python])
            if auto_retry:
                cmd.append("--auto-retry")

            t_eval = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=_clean_env(),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=reward_timeout,
                    check=False,
                )
            patched_pass, patched_n, patched_rewards, patched_task_hashes = _patched_pass(
                output_root,
                threshold,
                goal_seed=webshop_goal_seed,
            )
            if identity is not None:
                assert_matching_task_hashes(
                    identity["task_hashes"],
                    patched_task_hashes,
                    require_complete=patched_n >= batch_size,
                )
            retryable_infra = _retryable_eval_failure(log_path, output_root)
            retry_reason = ""
            if proc.returncode != 0 and retryable_infra:
                retry_reason = "infra_failure"
            if patched_n < batch_size and retry_incomplete_eval:
                retry_reason = "incomplete_eval"
            retryable = (
                attempt + 1 < max_eval_attempts
                and (proc.returncode != 0 or patched_n < batch_size)
                and bool(retry_reason)
            )
            eval_attempts.append(
                {
                    "attempt": attempt,
                    "returncode": proc.returncode,
                    "patched_n": patched_n,
                    "retryable_eval_failure": retryable,
                    "retryable_infra_failure": retryable and retry_reason == "infra_failure",
                    "retry_reason": retry_reason if retryable else "",
                    "controller_port": controller_port,
                    "worker_port_base": worker_port_base,
                    "log": str(log_path),
                    "elapsed_sec": round(time.monotonic() - t_eval, 3),
                }
            )
            timing["controller_port"] = controller_port
            timing["worker_port_base"] = worker_port_base
            if not retryable:
                break
        timing["eval_subprocess_sec"] = round(time.monotonic() - t_eval_all, 3)
        timing["eval_attempts"] = eval_attempts

        t_collect = time.monotonic()
        baseline_average_reward = _baseline_average_reward(metadata, batch_size=batch_size)
        patched_average_reward = _average_reward(patched_rewards, batch_size=batch_size)
        delta_average_reward = patched_average_reward - baseline_average_reward
        timing["collect_reward_sec"] = round(time.monotonic() - t_collect, 3)
        if proc is None or proc.returncode != 0 or patched_n < batch_size:
            result = _no_patch_reward(
                metadata,
                "eval_failed_treated_as_no_patch",
                f"returncode={None if proc is None else proc.returncode}; patched_n={patched_n}; log={log_path}",
            )
            result.update({"valid_patch": True, "patch_path": str(patch_path), "eval_log": str(log_path)})
        else:
            delta_pass = patched_pass - baseline_pass
            delta_pass_rate = delta_pass / max(1, batch_size)
            delta_score = delta_average_reward if reward_metric == "delta_average_reward" else delta_pass_rate
            result = {
                "score": delta_score + valid_bonus,
                "delta_score": delta_score,
                "valid_bonus": valid_bonus,
                "reward_metric": reward_metric,
                "delta_pass": delta_pass,
                "delta_pass_rate": delta_pass_rate,
                "delta_average_reward": delta_average_reward,
                "baseline_pass": baseline_pass,
                "patched_pass": patched_pass,
                "baseline_average_reward": baseline_average_reward,
                "patched_average_reward": patched_average_reward,
                "batch_size": batch_size,
                "patched_n": patched_n,
                "valid_patch": True,
                "eval_status": "ok",
                "patch_path": str(patch_path),
                "eval_log": str(log_path),
                "target_url": target_url,
                "patched_rewards": patched_rewards,
                "webshop_identity_protocol": (identity or {}).get("protocol"),
                "webshop_goal_seed": webshop_goal_seed,
                "baseline_task_manifest_sha256": (identity or {}).get("sha256"),
                "patched_task_manifests": {
                    str(index): digest for index, digest in patched_task_hashes.items()
                },
            }
        _attach_timing(result, _finish_timing(timing, t0))
        _json_dump(reward_path, result)
        return result
    except subprocess.TimeoutExpired as exc:
        result = _no_patch_reward(metadata, "eval_timeout_treated_as_no_patch", repr(exc))
        _attach_timing(result, _finish_timing(timing, t0))
        _json_dump(reward_path, result)
        return result
    except WebShopIdentityError:
        raise
    except Exception as exc:
        result = _no_patch_reward(metadata, "reward_error_treated_as_no_patch", traceback.format_exc())
        _attach_timing(result, _finish_timing(timing, t0))
        _json_dump(reward_path, result)
        return result
    finally:
        if port_locks:
            _release_port_locks(port_locks)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
    del kwargs
    if isinstance(samples, list):
        configured_limit = int(getattr(args, "harness_r1_reward_max_concurrency", 0) or 0)
        runtime_limit = int(getattr(args, "reward_max_concurrency", 2) or 2)
        limit = max(configured_limit, runtime_limit)
        semaphore = asyncio.Semaphore(max(1, limit))

        async def run_one(sample: Sample) -> dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(_score_sync, args, sample)

        tasks = [run_one(sample) for sample in samples]
        return await asyncio.gather(*tasks)
    return await asyncio.to_thread(_score_sync, args, samples)
