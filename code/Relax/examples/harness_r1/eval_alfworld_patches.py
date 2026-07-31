# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Evaluate Harness-R1 patching on fixed ALFWorld batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTBENCH_DIR = REPO_ROOT / "external/harness_evolution/life-harness/AgentBench"
AGENTBENCH_PYTHON = AGENTBENCH_DIR / ".venv/bin/python"
AGENTBENCH_SCRIPTS = REPO_ROOT / "code/life-harness/AgentBench/scripts"
ALFWORLD_WORKER_PYTHON = AGENTBENCH_DIR / ".venvs/alfworld-worker/bin/python"
sys.path.insert(0, str(AGENTBENCH_SCRIPTS))

from harness_r1_patch import (  # noqa: E402
    PatchValidationError,
    extract_json_object,
    extract_prefilled_think_patch_json_object,
    extract_think_patch_json_object,
    normalize_patch,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean_env() -> dict[str, str]:
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
    return env


def engineer_retry_delay(args: argparse.Namespace, exc: Exception, attempt: int) -> float:
    """Return a bounded exponential delay, honoring numeric Retry-After headers."""
    if isinstance(exc, urllib.error.HTTPError):
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return min(float(retry_after), args.engineer_retry_max_sleep)
            except ValueError:
                pass
    return min(args.engineer_retry_sleep * (2**attempt), args.engineer_retry_max_sleep)


def engineer_error_is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return True


def engineer_error_is_fatal(args: argparse.Namespace, exc: Exception) -> bool:
    return (
        isinstance(exc, urllib.error.HTTPError)
        and exc.code in args.engineer_fail_http_statuses
    )


def parse_http_statuses(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        statuses = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HTTP statuses must be comma-separated integers") from exc
    if any(status < 100 or status > 599 for status in statuses):
        raise argparse.ArgumentTypeError("HTTP statuses must be between 100 and 599")
    return statuses


def message_text(message: dict[str, Any], response_protocol: str) -> str:
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    if response_protocol in {"prefill_think_patch", "prefill_think_patch_relaxed"} and reasoning.strip():
        return f"{reasoning.strip()}\n</think>\n{content.lstrip()}"
    if content.strip() and reasoning.strip() and "<think" not in content.lower():
        return f"<think>\n{reasoning.strip()}\n</think>\n{content}"
    if content.strip():
        return content
    return reasoning


def call_chat(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    request_messages = messages
    if args.engineer_assistant_prefill:
        request_messages = [
            *messages,
            {"role": "assistant", "content": args.engineer_assistant_prefill},
        ]
    payload = {
        "model": args.engineer_model,
        "messages": request_messages,
        "temperature": args.engineer_temperature,
        "top_p": args.engineer_top_p,
        "max_tokens": args.engineer_max_tokens,
    }
    if args.engineer_assistant_prefill:
        payload["continue_final_message"] = True
    if args.engineer_chat_template_kwargs:
        payload["chat_template_kwargs"] = args.engineer_chat_template_kwargs
    if args.engineer_reasoning_effort:
        payload["reasoning_effort"] = args.engineer_reasoning_effort
    reserved_extra_keys = {"model", "messages"} & set(args.engineer_extra_body)
    if reserved_extra_keys:
        raise ValueError(
            "--engineer-extra-body cannot override reserved keys: "
            + ", ".join(sorted(reserved_extra_keys))
        )
    payload.update(args.engineer_extra_body)
    req = urllib.request.Request(
        args.engineer_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.engineer_api_key}",
        },
    )
    opener = (
        urllib.request.build_opener()
        if args.engineer_use_env_proxy
        else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    )
    last_error: Exception | None = None
    for attempt in range(args.engineer_retries + 1):
        try:
            with opener.open(req, timeout=args.engineer_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                error = data.get("error") if isinstance(data, dict) else None
                raise ValueError(
                    "engineer response is missing a non-empty choices list; "
                    f"keys={sorted(data) if isinstance(data, dict) else type(data).__name__}, "
                    f"error={str(error)[:500]}"
                )
            return data
        except (
            ConnectionError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt >= args.engineer_retries or not engineer_error_is_retryable(exc):
                raise
            delay = engineer_retry_delay(args, exc, attempt)
            print(
                f"[engineer-retry] model={args.engineer_model} "
                f"attempt={attempt + 1}/{args.engineer_retries + 1} "
                f"error={type(exc).__name__} sleep={delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"engineer request failed: {last_error!r}")


def parse_patch(response: str, response_protocol: str) -> tuple[dict[str, Any] | None, str]:
    try:
        if response_protocol == "full_think_patch":
            raw = extract_think_patch_json_object(response or "")
        elif response_protocol == "full_think_patch_relaxed":
            try:
                raw = extract_think_patch_json_object(response or "")
            except PatchValidationError:
                patch_matches = list(
                    re.finditer(r"<patch>\s*(.*?)\s*</patch>", response or "", re.DOTALL | re.IGNORECASE)
                )
                raw = extract_json_object(patch_matches[-1].group(1) if patch_matches else response or "")
        elif response_protocol == "prefill_think_patch":
            raw = extract_prefilled_think_patch_json_object(response or "")
        elif response_protocol == "prefill_think_patch_relaxed":
            raw = extract_json_object(response or "")
        else:
            raw = extract_json_object(response or "")
        return normalize_patch(raw, bench="alfworld"), ""
    except (json.JSONDecodeError, PatchValidationError, RecursionError, TypeError, ValueError) as exc:
        return None, repr(exc)


def reward_of(row: dict[str, Any]) -> float:
    result = ((row.get("output") or {}).get("result") or {})
    try:
        return float(result.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def collect_rewards(eval_root: Path) -> dict[int, float]:
    rewards: dict[int, float] = {}
    for path in sorted(eval_root.glob("alfworld/batch_*/rollout/*/*/runs.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row.get("index")
            if isinstance(idx, int):
                rewards[idx] = reward_of(row)
    return rewards


def baseline_rewards(metadata: dict[str, Any]) -> dict[int, float]:
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


def average_reward(rewards: dict[int, float], fallback_pass: int, batch_size: int) -> float:
    if rewards:
        return sum(float(value) for value in rewards.values()) / len(rewards)
    return fallback_pass / max(1, batch_size)


def evaluate_patch(
    args: argparse.Namespace,
    record: dict[str, Any],
    patch_path: Path,
    batch_dir: Path,
    ordinal: int,
) -> dict[str, Any]:
    metadata = record["metadata"]
    start = int(metadata["start"])
    task_ids = metadata.get("task_ids")
    if task_ids is not None:
        task_ids = [int(item) for item in task_ids]
    batch_size = int(metadata["batch_size"])
    if task_ids is not None and len(task_ids) != batch_size:
        raise ValueError(
            f"metadata.task_ids has {len(task_ids)} entries, expected batch_size={batch_size}"
        )
    baseline_pass = int(metadata["baseline_pass"])
    threshold = float(metadata.get("reward_threshold") or 1.0)
    base_rewards = baseline_rewards(metadata)
    baseline_average_reward = average_reward(base_rewards, baseline_pass, batch_size)
    eval_root = batch_dir / "eval"
    run_id = f"{args.run_id}_{metadata['batch_tag']}"
    controller_port = args.controller_port_base + ordinal
    worker_port_base = args.worker_port_base + ordinal * 20
    log_path = batch_dir / "eval.log"
    task_ids_path: Path | None = None
    if task_ids is not None:
        task_ids_path = batch_dir / "task_ids.json"
        dump_json(task_ids_path, task_ids)
    cmd = [
        str(args.agentbench_python),
        str(AGENTBENCH_SCRIPTS / "harness_r1_batch_debug.py"),
        "--agentbench-dir",
        str(args.agentbench_dir),
        "--agentbench-python",
        str(args.agentbench_python),
        "--bench",
        "alfworld",
        "--run-id",
        run_id,
        "--alfworld-split",
        str(metadata.get("split") or args.alfworld_split),
        "--alfworld-max-step",
        str(args.alfworld_max_step),
        "--alfworld-worker-python",
        str(args.alfworld_worker_python),
        "--start",
        str(start),
        "--batch-size",
        str(batch_size),
        "--num-batches",
        "1",
        "--max-parallel-batches",
        "1",
        "--rollout-concurrency",
        str(args.rollout_concurrency),
    ]
    if args.rollout_auto_retry:
        cmd.append("--auto-retry")
    cmd.extend(
        [
            "--skip-adb",
            "--controller-port",
            str(controller_port),
            "--worker-port-base",
            str(worker_port_base),
            "--startup-timeout",
            str(args.startup_timeout),
            "--rollout-timeout",
            str(args.rollout_timeout),
            "--rollout-base-url",
            args.target_base_url,
            "--rollout-model",
            args.target_model,
            "--agent-name",
            args.target_agent_name,
            "--rollout-temperature",
            "0.0",
            "--rollout-max-tokens",
            str(args.rollout_max_tokens),
            "--rollout-tool-choice",
            args.rollout_tool_choice,
            "--output-root",
            str(eval_root),
            "--harness-patch",
            str(patch_path),
        ]
    )
    if args.rollout_chat_template_kwargs:
        cmd.extend(["--rollout-chat-template-kwargs", json.dumps(args.rollout_chat_template_kwargs)])
    if args.rollout_disable_parallel_tool_calls:
        cmd.append("--rollout-disable-parallel-tool-calls")
    if args.rollout_single_tool_call_only:
        cmd.append("--rollout-single-tool-call-only")
    if task_ids_path is not None:
        cmd.extend(["--task-ids-file", str(task_ids_path)])
    with log_path.open("w", encoding="utf-8") as log_f:
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=clean_env(),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.eval_timeout,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            log_f.write(f"\n[harness-r1] eval timed out after {args.eval_timeout}s\n")
            returncode = 124
    rewards = collect_rewards(eval_root)
    patched_pass = sum(value >= threshold for value in rewards.values())
    status = "ok" if returncode == 0 and len(rewards) >= batch_size else "eval_failed_treated_as_no_patch"
    if status != "ok":
        patched_pass = baseline_pass
        rewards = {}
    patched_average_reward = (
        average_reward(rewards, patched_pass, batch_size) if status == "ok" else baseline_average_reward
    )
    return {
        "eval_status": status,
        "eval_returncode": returncode,
        "eval_log": str(log_path),
        "eval_root": str(eval_root),
        "patched_rewards": rewards,
        "patched_n": len(rewards),
        "baseline_pass": baseline_pass,
        "patched_pass": patched_pass,
        "baseline_average_reward": baseline_average_reward,
        "patched_average_reward": patched_average_reward,
        "batch_size": batch_size,
        "delta_pass": patched_pass - baseline_pass,
        "delta_pass_rate": (patched_pass - baseline_pass) / max(1, batch_size),
        "delta_average_reward": patched_average_reward - baseline_average_reward,
        "score": (patched_pass - baseline_pass) / max(1, batch_size),
    }


def run_one(args: argparse.Namespace, indexed_record: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    ordinal, record = indexed_record
    metadata = record["metadata"]
    batch_tag = str(metadata["batch_tag"])
    batch_dir = args.output_root / batch_tag
    result_path = batch_dir / "result.json"
    if args.resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            args.resume_rerun_eval_failed
            and existing.get("valid_patch") is True
            and existing.get("eval_status") == "eval_failed_treated_as_no_patch"
        ):
            patch_path = Path(existing.get("patch_path") or batch_dir / "patch.json")
            if not patch_path.is_file():
                raise FileNotFoundError(
                    f"cannot rerun {batch_tag}: existing patch is missing: {patch_path}"
                )
            shutil.rmtree(batch_dir / "eval", ignore_errors=True)
            print(f"[eval-alfworld] rerunning existing valid patch for {batch_tag}", flush=True)
            result = {
                **existing,
                "patch_path": str(patch_path),
                **evaluate_patch(args, record, patch_path, batch_dir, ordinal),
            }
            dump_json(result_path, result)
            return result
        retry_statuses = set(args.resume_retry_patch_status)
        if existing.get("patch_status") not in retry_statuses:
            return existing
        print(
            f"[eval-alfworld] retrying {batch_tag} with patch_status="
            f"{existing.get('patch_status')}",
            flush=True,
        )

    batch_dir.mkdir(parents=True, exist_ok=True)
    dump_json(batch_dir / "metadata.json", metadata)
    base_rewards = baseline_rewards(metadata)
    baseline_average_reward = average_reward(
        base_rewards,
        fallback_pass=int(metadata["baseline_pass"]),
        batch_size=int(metadata["batch_size"]),
    )
    if args.patch_source_root is not None:
        source_patch = args.patch_source_root / batch_tag / "patch.json"
        if not source_patch.exists():
            result = {
                "batch_tag": batch_tag,
                "start": metadata["start"],
                "end": metadata["end"],
                "valid_patch": False,
                "patch_status": "missing_source_patch_treated_as_no_patch",
                "source_patch_path": str(source_patch),
                "baseline_pass": int(metadata["baseline_pass"]),
                "patched_pass": int(metadata["baseline_pass"]),
                "baseline_average_reward": baseline_average_reward,
                "patched_average_reward": baseline_average_reward,
                "batch_size": int(metadata["batch_size"]),
                "delta_pass": 0,
                "delta_pass_rate": 0.0,
                "delta_average_reward": 0.0,
                "score": 0.0,
            }
            dump_json(result_path, result)
            return result
        patch_path = batch_dir / "patch.json"
        patch = json.loads(source_patch.read_text(encoding="utf-8"))
        dump_json(patch_path, patch)
        result = {
            "batch_tag": batch_tag,
            "start": metadata["start"],
            "end": metadata["end"],
            "valid_patch": True,
            "patch_status": "valid_reused",
            "source_patch_path": str(source_patch),
            "patch_path": str(patch_path),
            **evaluate_patch(args, record, patch_path, batch_dir, ordinal),
        }
        dump_json(result_path, result)
        return result
    try:
        response_data = call_chat(args, record["prompt"])
    except Exception as exc:  # Keep long teacher runs alive; invalid means no-patch.
        if engineer_error_is_fatal(args, exc):
            raise RuntimeError(
                f"fatal engineer HTTP {exc.code} for batch {batch_tag}; aborting evaluation"
            ) from exc
        result = {
            "batch_tag": batch_tag,
            "start": metadata["start"],
            "end": metadata["end"],
            "valid_patch": False,
            "patch_status": "engineer_request_failed_treated_as_no_patch",
            "parse_error": repr(exc),
            "baseline_pass": int(metadata["baseline_pass"]),
            "patched_pass": int(metadata["baseline_pass"]),
            "baseline_average_reward": baseline_average_reward,
            "patched_average_reward": baseline_average_reward,
            "batch_size": int(metadata["batch_size"]),
            "delta_pass": 0,
            "delta_pass_rate": 0.0,
            "delta_average_reward": 0.0,
            "score": 0.0,
        }
        dump_json(result_path, result)
        return result
    dump_json(batch_dir / "response.json", response_data)
    message = response_data["choices"][0].get("message") or {}
    response = message_text(message, args.response_protocol)
    dump_text(batch_dir / "raw_response.txt", response)
    if message.get("reasoning_content") is not None:
        dump_text(batch_dir / "reasoning_content.txt", message.get("reasoning_content") or "")
    patch, parse_error = parse_patch(response, args.response_protocol)
    if patch is None:
        result = {
            "batch_tag": batch_tag,
            "start": metadata["start"],
            "end": metadata["end"],
            "valid_patch": False,
            "patch_status": "invalid_patch_treated_as_no_patch",
            "parse_error": parse_error,
            "baseline_pass": int(metadata["baseline_pass"]),
            "patched_pass": int(metadata["baseline_pass"]),
            "baseline_average_reward": baseline_average_reward,
            "patched_average_reward": baseline_average_reward,
            "batch_size": int(metadata["batch_size"]),
            "delta_pass": 0,
            "delta_pass_rate": 0.0,
            "delta_average_reward": 0.0,
            "score": 0.0,
        }
        dump_json(result_path, result)
        return result

    patch_path = batch_dir / "patch.json"
    dump_json(patch_path, patch)
    if args.generate_only:
        result = {
            "batch_tag": batch_tag,
            "start": metadata["start"],
            "end": metadata["end"],
            "valid_patch": True,
            "patch_status": "generated_only",
            "patch_path": str(patch_path),
            "baseline_pass": int(metadata["baseline_pass"]),
            "patched_pass": int(metadata["baseline_pass"]),
            "baseline_average_reward": baseline_average_reward,
            "patched_average_reward": baseline_average_reward,
            "batch_size": int(metadata["batch_size"]),
            "delta_pass": 0,
            "delta_pass_rate": 0.0,
            "delta_average_reward": 0.0,
            "score": 0.0,
        }
    else:
        result = {
            "batch_tag": batch_tag,
            "start": metadata["start"],
            "end": metadata["end"],
            "valid_patch": True,
            "patch_status": "valid",
            "patch_path": str(patch_path),
            **evaluate_patch(args, record, patch_path, batch_dir, ordinal),
        }
    dump_json(result_path, result)
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_batches = len(results)
    total_tasks = sum(int(r.get("batch_size", 0)) for r in results)
    baseline_pass = sum(int(r.get("baseline_pass", 0)) for r in results)
    patched_pass = sum(int(r.get("patched_pass", 0)) for r in results)
    baseline_reward_sum = sum(float(r.get("baseline_average_reward", 0.0) or 0.0) * int(r.get("batch_size", 0)) for r in results)
    patched_reward_sum = sum(float(r.get("patched_average_reward", 0.0) or 0.0) * int(r.get("batch_size", 0)) for r in results)
    valid = sum(bool(r.get("valid_patch")) for r in results)
    eval_ok = sum(r.get("eval_status") == "ok" for r in results)
    return {
        "total_batches": total_batches,
        "total_tasks": total_tasks,
        "valid_patches": valid,
        "eval_ok_batches": eval_ok,
        "baseline_pass": baseline_pass,
        "patched_pass": patched_pass,
        "delta_pass": patched_pass - baseline_pass,
        "baseline_pass_rate": baseline_pass / max(1, total_tasks),
        "patched_pass_rate": patched_pass / max(1, total_tasks),
        "delta_pass_rate": (patched_pass - baseline_pass) / max(1, total_tasks),
        "baseline_average_reward": baseline_reward_sum / max(1, total_tasks),
        "patched_average_reward": patched_reward_sum / max(1, total_tasks),
        "delta_average_reward": (patched_reward_sum - baseline_reward_sum) / max(1, total_tasks),
        "invalid_or_failed_batches": total_batches - eval_ok,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--engineer-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--engineer-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--engineer-api-key", default="EMPTY")
    parser.add_argument("--engineer-api-key-env", default="")
    parser.add_argument("--engineer-timeout", type=int, default=900)
    parser.add_argument("--engineer-max-tokens", type=int, default=8192)
    parser.add_argument("--engineer-concurrency", type=int, default=1)
    parser.add_argument("--engineer-temperature", type=float, default=0.0)
    parser.add_argument("--engineer-top-p", type=float, default=1.0)
    parser.add_argument("--engineer-chat-template-kwargs", type=json.loads, default={})
    parser.add_argument(
        "--engineer-extra-body",
        type=json.loads,
        default={},
        help="Additional JSON request fields, such as OpenRouter provider routing preferences.",
    )
    parser.add_argument(
        "--engineer-use-env-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY for external engineer APIs; disabled by default for local endpoints.",
    )
    parser.add_argument(
        "--engineer-assistant-prefill",
        default="",
        help=(
            "Append this assistant message before generation and continue it. "
            "Use '<think>\\n' for Qwen3.5 prefill-think checkpoints."
        ),
    )
    parser.add_argument("--engineer-reasoning-effort", choices=["low", "medium", "high"], default="")
    parser.add_argument("--engineer-retries", type=int, default=0)
    parser.add_argument("--engineer-retry-sleep", type=float, default=10.0)
    parser.add_argument("--engineer-retry-max-sleep", type=float, default=120.0)
    parser.add_argument(
        "--engineer-fail-http-statuses",
        type=parse_http_statuses,
        default=(),
        help=(
            "Comma-separated HTTP statuses that abort the whole evaluation instead of being "
            "counted as no-patch. Useful for authentication or exhausted-budget errors."
        ),
    )
    parser.add_argument(
        "--patch-source-root",
        type=Path,
        default=None,
        help="Reuse patch.json files from this eval root instead of calling the engineer model.",
    )
    parser.add_argument("--response-protocol", choices=["json_patch", "full_think_patch", "full_think_patch_relaxed", "prefill_think_patch", "prefill_think_patch_relaxed"], default="full_think_patch_relaxed")
    parser.add_argument("--target-base-url", default="http://127.0.0.1:8004/v1")
    parser.add_argument("--target-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--target-agent-name", default="alfworld-target")
    parser.add_argument("--alfworld-split", default="harness_r1_eval500_20260609")
    parser.add_argument("--alfworld-max-step", type=int, default=50)
    parser.add_argument("--alfworld-worker-python", type=Path, default=ALFWORLD_WORKER_PYTHON)
    parser.add_argument("--rollout-concurrency", type=int, default=10)
    parser.add_argument("--rollout-auto-retry", action="store_true")
    parser.add_argument("--rollout-max-tokens", type=int, default=4096)
    parser.add_argument("--rollout-tool-choice", choices=["auto", "required", "none"], default="required")
    parser.add_argument("--rollout-chat-template-kwargs", type=json.loads, default={})
    parser.add_argument("--rollout-disable-parallel-tool-calls", action="store_true")
    parser.add_argument("--rollout-single-tool-call-only", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--rollout-timeout", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=2400)
    parser.add_argument("--controller-port-base", type=int, default=30000)
    parser.add_argument("--worker-port-base", type=int, default=42000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-retry-patch-status",
        action="append",
        default=[],
        help=(
            "With --resume, regenerate records whose existing patch_status exactly matches "
            "this value. Repeat the option for multiple transient statuses."
        ),
    )
    parser.add_argument(
        "--resume-rerun-eval-failed",
        action="store_true",
        help="With --resume, rerun valid patches whose previous environment evaluation failed.",
    )
    parser.add_argument("--agentbench-dir", type=Path, default=AGENTBENCH_DIR)
    parser.add_argument("--agentbench-python", type=Path, default=AGENTBENCH_PYTHON)
    args = parser.parse_args()
    if args.engineer_api_key_env:
        args.engineer_api_key = os.environ.get(args.engineer_api_key_env, args.engineer_api_key)
    elif args.engineer_api_key == "EMPTY":
        args.engineer_api_key = os.environ.get("HARNESS_R1_ENGINEER_API_KEY", args.engineer_api_key)
    args.output_root = args.output_root.resolve()
    if args.patch_source_root is not None:
        args.patch_source_root = args.patch_source_root.resolve()
    args.agentbench_dir = args.agentbench_dir.resolve()
    args.agentbench_python = args.agentbench_python.resolve()
    # Keep venv python symlinks intact. Resolving them can turn
    # .venvs/alfworld-worker/bin/python into the bare interpreter and drop
    # that venv's site-packages from the worker process.
    args.alfworld_worker_python = args.alfworld_worker_python.expanduser().absolute()
    return args


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.input.resolve())
    if args.limit:
        records = records[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if run_config.get("engineer_api_key") not in ("", "EMPTY", None):
        run_config["engineer_api_key"] = "***"
    dump_json(args.output_root / "run_config.json", run_config)
    started = time.time()
    results: list[dict[str, Any]] = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.engineer_concurrency)
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    try:
        futures = [pool.submit(run_one, args, item) for item in enumerate(records)]
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            dump_json(args.output_root / "summary.partial.json", summarize(results) | {"completed": i})
            print(
                f"[eval-alfworld] {i}/{len(records)} {result.get('batch_tag')} "
                f"valid={result.get('valid_patch')} delta={result.get('delta_pass')}",
                flush=True,
            )
    except Exception:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    results = sorted(results, key=lambda r: int(r.get("start", 0)))
    summary = summarize(results)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    dump_json(args.output_root / "results.json", results)
    dump_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
