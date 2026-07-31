#!/usr/bin/env python3
"""Generate and evaluate DBBench Harness-R1 code-hook patches.

The script reads no-harness DBBench batch-debug outputs, asks an engineer
model for one patch per batch, validates the patch with harness_r1_patch.py,
and reruns the same batch with that patch. Invalid patches are counted as
no-patch.
"""

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
AGENTBENCH_DIR = REPO_ROOT / "code/life-harness/AgentBench"
AGENTBENCH_SCRIPTS = AGENTBENCH_DIR / "scripts"
DEFAULT_AGENTBENCH_PYTHON = Path("/data/cache/harness_r1_agentbench_venv_py310/bin/python")
sys.path.insert(0, str(AGENTBENCH_SCRIPTS))

from harness_r1_patch import (  # noqa: E402
    PatchValidationError,
    extract_json_object,
    extract_prefilled_think_patch_json_object,
    extract_think_patch_json_object,
    normalize_patch,
    schema_prompt,
)
from prepare_dbbench_qwen35_prefill_sft import (  # noqa: E402
    DBBENCH_PREFILL_SYSTEM_PROMPT,
    dbbench_runtime_context_section as dbbench_sft_runtime_context_section,
)


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prompt_messages_from_reference(path: Path) -> list[dict[str, str]]:
    """Load one prompt message list from JSONL or LlamaFactory JSON data."""
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"prompt prefix reference has no rows: {path}")
        row = payload[0]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"prompt prefix reference has no rows: {path}")
        row = rows[0]
    messages = row.get("prompt") or row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"prompt prefix reference is missing prompt/messages: {path}")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"invalid prompt message in reference: {path}")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"invalid prompt role/content in reference: {path}")
        normalized.append({"role": role, "content": content})
    return normalized


def align_prompt_prefix_to_reference(
    prompt: list[dict[str, str]], reference: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Replace static instructions while preserving this batch's evidence."""
    marker = "Observed no-harness rollout evidence:"

    def split_messages(messages: list[dict[str, str]]) -> tuple[int, str, str]:
        user_indices = [i for i, message in enumerate(messages) if message.get("role") == "user"]
        if len(user_indices) != 1:
            raise ValueError(f"expected one user message, found {len(user_indices)}")
        index = user_indices[0]
        content = messages[index]["content"]
        if marker not in content:
            raise ValueError(f"prompt user message is missing marker: {marker}")
        prefix, evidence = content.split(marker, 1)
        return index, prefix + marker, evidence

    prompt_user_index, _, evidence = split_messages(prompt)
    reference_user_index, reference_prefix, _ = split_messages(reference)
    reference_system = [message for message in reference if message.get("role") == "system"]
    if len(reference_system) != 1:
        raise ValueError(
            f"expected one system message in prompt reference, found {len(reference_system)}"
        )
    aligned = [dict(message) for message in prompt]
    system_indices = [i for i, message in enumerate(aligned) if message.get("role") == "system"]
    if len(system_indices) != 1:
        raise ValueError(f"expected one system message, found {len(system_indices)}")
    aligned[system_indices[0]]["content"] = reference_system[0]["content"]
    aligned[prompt_user_index]["content"] = reference_prefix + evidence
    # The reference user position is intentionally validated even though the
    # target prompt keeps its original message ordering.
    if reference_user_index < 0:
        raise AssertionError("unreachable")
    return aligned


def reward_of(row: dict[str, Any]) -> float:
    result = ((row.get("output") or {}).get("result") or {})
    try:
        return float(result.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def status_of(row: dict[str, Any]) -> str:
    return str((row.get("output") or {}).get("status") or row.get("status") or "unknown")


def collect_rewards(run_root: Path, bench: str = "dbbench") -> dict[int, float]:
    rewards: dict[int, float] = {}
    for path in sorted(run_root.glob(f"{bench}/batch_*/rollout/*/*/runs.jsonl")):
        for row in read_jsonl(path):
            idx = row.get("index")
            if isinstance(idx, int):
                rewards[idx] = reward_of(row)
    return rewards


def average_reward(rewards: dict[int, float], fallback_pass: int, batch_size: int) -> float:
    if rewards:
        return sum(float(v) for v in rewards.values()) / len(rewards)
    return fallback_pass / max(1, batch_size)


def batch_id_from_dir(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def extract_message_text(message: dict[str, Any], max_chars: int) -> str:
    content = str(message.get("content") or "")
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "\n...[truncated]"


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_len = max_chars // 2
    tail_len = max_chars - head_len
    return text[:head_len].rstrip() + "\n...[middle truncated]...\n" + text[-tail_len:].lstrip()


def render_tool_call(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    name = str(fn.get("name") or "")
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    args = re.sub(r"\s+", " ", args or "").strip()
    if len(args) > 600:
        args = args[:600].rstrip() + " ...[truncated]"
    return f"ASSISTANT_TOOL {name}: {args}"


def render_trace(row: dict[str, Any], max_messages: int, max_tool_chars: int) -> str:
    output = row.get("output") or {}
    result = output.get("result") or {}
    messages = result.get("openai_messages") or output.get("history") or []
    lines = [
        f"Trace index={row.get('index')} status={status_of(row)} reward={reward_of(row):.3f}",
    ]
    shown = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role == "user" and "Evaluation metadata:" in str(message.get("content") or ""):
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if isinstance(calls, list) and calls:
                for call in calls[:2]:
                    lines.append(render_tool_call(call))
            else:
                text = extract_message_text(message, max_chars=500)
                if text.strip():
                    lines.append("ASSISTANT_TEXT: " + re.sub(r"\s+", " ", text).strip())
        elif role == "tool":
            text = extract_message_text(message, max_chars=max_tool_chars)
            lines.append("TOOL_RESULT: " + re.sub(r"\s+", " ", text).strip())
        elif role == "user":
            text = extract_message_text(message, max_chars=900)
            if text.strip():
                lines.append("USER: " + re.sub(r"\s+", " ", text).strip())
        shown += 1
        if shown >= max_messages:
            lines.append("...[trace truncated]")
            break
    return "\n".join(lines)


def rows_from_run_root(run_root: Path, bench: str = "dbbench") -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not run_root.exists():
        return rows
    for path in sorted(run_root.glob(f"{bench}/batch_*/rollout/*/*/runs.jsonl")):
        for row in read_jsonl(path):
            idx = row.get("index")
            if isinstance(idx, int):
                rows[idx] = row
    return rows


def baseline_rows(batch_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("rollout/*/*/runs.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def build_revision_context(args: argparse.Namespace, batch_tag: str, metadata: dict[str, Any]) -> str:
    if args.revision_source_root is None:
        return ""
    prev_dir = args.revision_source_root / batch_tag
    prev_result_path = prev_dir / "result.json"
    if not prev_result_path.exists():
        if args.revision_require_previous:
            raise FileNotFoundError(f"missing previous result for revision: {prev_result_path}")
        return ""
    prev_result = load_json(prev_result_path)
    prev_patch_path = prev_dir / "patch.json"
    prev_patch_text = prev_patch_path.read_text(encoding="utf-8", errors="replace") if prev_patch_path.exists() else ""
    prev_raw_path = prev_dir / "raw_response.txt"
    prev_raw = prev_raw_path.read_text(encoding="utf-8", errors="replace") if prev_raw_path.exists() else ""
    patched_rows = rows_from_run_root(Path(str(prev_result.get("eval_root") or "")), bench="dbbench")
    baseline_rewards = {int(k): float(v) for k, v in (metadata.get("baseline_rewards") or {}).items()}
    patched_rewards = {
        int(k): float(v)
        for k, v in (prev_result.get("patched_rewards") or {}).items()
        if str(k).isdigit()
    }
    down_ids = [
        idx for idx, base in baseline_rewards.items()
        if base >= args.reward_threshold and patched_rewards.get(idx, 0.0) < args.reward_threshold
    ]
    up_ids = [
        idx for idx, base in baseline_rewards.items()
        if base < args.reward_threshold and patched_rewards.get(idx, 0.0) >= args.reward_threshold
    ]
    down_ids = sorted(down_ids)[: args.max_revision_regression_traces]
    up_ids = sorted(up_ids)[: args.max_revision_improvement_traces]
    lines = [
        "## Second-pass revision context",
        "",
        "The previous patch was valid but regressed this same batch. Generate a complete replacement patch and a new <think> section.",
        "",
        "Previous same-batch result:",
        f"- baseline_pass: {prev_result.get('baseline_pass')}/{prev_result.get('batch_size')}",
        f"- patched_pass: {prev_result.get('patched_pass')}/{prev_result.get('batch_size')}",
        f"- delta_pass: {prev_result.get('delta_pass')}",
        f"- valid_patch: {prev_result.get('valid_patch')}",
        f"- eval_status: {prev_result.get('eval_status')}",
        f"- correct_to_wrong task ids for diagnosis only: {down_ids}",
        f"- wrong_to_correct task ids for diagnosis only: {up_ids}",
        "",
        "Critical DBBench environment fact:",
        "- Successful UPDATE/INSERT/DELETE often returns [] in this environment.",
        "- Therefore do not treat [] after a data-changing SQL statement as failure.",
        "- For mutation tasks, prefer ctx['state']['mutation_attempted'] or predicates.mutation_not_attempted over custom non-empty-result heuristics.",
        "- Do not block commit after state.mutation_attempted is true unless the submitted answer is empty/give-up text.",
        "",
        "Previous raw model output:",
        truncate_middle(prev_raw, args.max_previous_response_chars).strip() or "(missing)",
        "",
        "Previous normalized patch JSON:",
        truncate_middle(prev_patch_text, args.max_previous_patch_chars).strip() or "(missing)",
        "",
        "Regression traces after applying the previous patch:",
    ]
    for idx in down_ids:
        row = patched_rows.get(idx)
        lines.append("")
        if row is None:
            lines.append(f"- missing patched trace for task {idx}")
        else:
            lines.append(render_trace(row, args.max_revision_messages_per_trace, args.max_tool_result_chars))
    if up_ids:
        lines.extend(["", "Improvement traces to preserve when narrowing the patch:"])
        for idx in up_ids:
            row = patched_rows.get(idx)
            if row is not None:
                lines.append("")
                lines.append(render_trace(row, args.max_revision_messages_per_trace, args.max_tool_result_chars))
    lines.extend(
        [
            "",
            "Revision instructions:",
            "- Diagnose the previous patch's regression in <think>.",
            "- Output a replacement patch, not a diff.",
            "- Prefer soft make_pre_hint/on_init guidance over broad on_before_action blocking.",
            "- If you use on_before_action, make it narrow and avoid mutation commit deadlocks.",
            "- Keep literal formatting guidance conservative; do not encourage changing commas, punctuation, or MySQL dialect.",
        ]
    )
    return "\n".join(lines)


def dbbench_runtime_context_section() -> str:
    return """## Optional DBBench Runtime Context

For this run, DBBench code hooks receive a read-only runtime context from the
harness substrate. Use it only to write general harness behavior, not
task-specific SQL.

- DBBench is a MySQL-backed table task environment with two tools:
  execute_sql(query) and commit_final_answer(answers).
- Hooks read ctx["action"], ctx["state"], ctx["task"], ctx["dbbench"], and
  ctx["predicates"]. There is no ctx["world"] object in the main DBBench
  runtime.
- ctx["action"] includes tool, query, answers, h2_action, h2_blocked_reason,
  commit_gate_action, and commit_gate_reason.
- ctx["state"] includes sql_count, last_sql, last_result, last_error_kind,
  last_error_text, last_result_was_error, error_streak, empty_streak,
  text_only_streak, loop_streak, mutation_attempted, candidate_answer,
  candidate_answer_shape, candidate_implausible, and remaining_rounds.
- ctx["task"] includes task_type, answer_shape, target_table, and description.
- ctx["dbbench"] includes sql_history, sql_history_raw, discovered_columns,
  db_response, and round.
- ctx["predicates"] includes no_sql_yet, commit_before_sql, commit_empty_answer,
  mutation_task, mutation_not_attempted, last_result_error, last_result_empty,
  unknown_column_error, syntax_error, repeated_sql, candidate_answer_available,
  remaining_rounds_low, and text_only_loop.
- SELECT-style tasks are judged from the submitted answer values.
- INSERT/UPDATE/DELETE tasks are judged from the final database state hash; the
  textual final answer is not the main correctness signal.
- Successful INSERT/UPDATE/DELETE frequently returns [] rather than affected-row
  metadata. Do not infer mutation failure from [] after data-changing SQL.
- Prefer ctx["state"]["mutation_attempted"] or
  ctx["predicates"]["mutation_not_attempted"] over custom scratch flags based on
  non-empty SQL results.
- ctx["state"]["last_error_kind"] == "empty" is useful for empty SELECT/filter
  debugging, but it is unsafe as a global failure signal after mutation SQL.
- ctx["state"]["candidate_answer"] is SQL-derived and noisy; do not treat it as
  an answer oracle.
- MySQL string concatenation is CONCAT(a, b), not a || b. The || operator can
  produce boolean-like values and corrupt text updates.
- Literal formatting matters: preserve commas, punctuation, spaces, percent
  signs, unicode/newline content, and exact requested text unless the task asks
  for normalization.
- Broad commit blocking is high risk. If a mutation has already been attempted,
  prefer a soft verification hint over blocking commit.
- Good harness patches should improve recurring behavior while preserving
  already-correct baseline trajectories.
""".strip()


def make_record(args: argparse.Namespace, batch_dir: Path) -> dict[str, Any] | None:
    status_path = batch_dir / "status.json"
    if not status_path.exists():
        return None
    status = load_json(status_path)
    rows = baseline_rows(batch_dir)
    if not rows:
        return None
    rewards = {int(row["index"]): reward_of(row) for row in rows if isinstance(row.get("index"), int)}
    task_ids = status.get("task_ids")
    if task_ids is not None:
        task_ids = [int(item) for item in task_ids]
    start, end = status.get("range", [None, None])
    if start is None or end is None:
        batch_id = batch_id_from_dir(batch_dir)
        start = batch_id * args.batch_size
        end = start + args.batch_size
    batch_size = len(task_ids) if task_ids is not None else int(end) - int(start)
    baseline_pass = sum(v >= args.reward_threshold for v in rewards.values())
    failures = [row for row in rows if reward_of(row) < args.reward_threshold]
    failures = sorted(failures, key=lambda r: (reward_of(r), int(r.get("index", 10**9))))[: args.max_failures_per_batch]
    if args.require_failures and not failures:
        return None
    overview_path = batch_dir / "debug/overview.md"
    overview = overview_path.read_text(encoding="utf-8", errors="replace") if overview_path.exists() else ""
    overview = overview[: args.max_overview_chars]
    evidence_lines = [
        f"# DBBench no-harness evidence for {batch_dir.name}",
        "",
        f"- task count: {batch_size}",
        f"- source slice: {start}:{end}",
        f"- baseline pass: {baseline_pass}/{batch_size}",
        f"- failures included: {len(failures)}",
        "",
        "## Existing Batch Overview",
        overview.strip() or "(no overview)",
        "",
        "## Failure Traces",
    ]
    for row in failures:
        evidence_lines.append("")
        evidence_lines.append(render_trace(row, args.max_messages_per_trace, args.max_tool_result_chars))
    if len("\n".join(evidence_lines)) > args.max_evidence_chars:
        text = "\n".join(evidence_lines)[: args.max_evidence_chars].rstrip()
        evidence = text + "\n...[evidence truncated]"
    else:
        evidence = "\n".join(evidence_lines)
    batch_id = batch_id_from_dir(batch_dir)
    metadata = {
        "benchmark": "dbbench",
        "batch_tag": f"batch_{batch_id:03d}",
        "batch_id": batch_id,
        "start": int(start),
        "end": int(end),
        "task_ids": task_ids,
        "batch_size": int(batch_size),
        "baseline_pass": int(baseline_pass),
        "baseline_rewards": {str(k): v for k, v in sorted(rewards.items())},
        "reward_threshold": args.reward_threshold,
    }
    revision_context = build_revision_context(args, metadata["batch_tag"], metadata)
    schema_response_protocol = (
        "full_think_patch"
        if args.response_protocol == "full_think_patch_relaxed"
        else args.response_protocol
    )
    use_sft_prefill_prompt = args.response_protocol == "prefill_think_patch" and not args.legacy_prefill_prompt
    user_parts = []
    if use_sft_prefill_prompt:
        user_parts.extend(
            [
                "You will edit only the reusable DBBench harness, not task answers.",
                (
                    "The chat template has already opened the assistant thinking block. "
                    "Continue concise recurring-failure reasoning, close it with </think>, "
                    "then output exactly one <patch> block."
                ),
            ]
        )
    user_parts.append(
        schema_prompt(
            bench="dbbench",
            response_protocol=schema_response_protocol,
            schema_style="dbbench_life_multihook_v1",
        )
    )
    if (
        args.add_prefill_thinking_instruction
        and args.response_protocol == "prefill_think_patch"
        and not use_sft_prefill_prompt
    ):
        user_parts.insert(
            0,
            (
                "The chat template has already opened the assistant thinking block. "
                "Continue concise recurring-failure reasoning, close it with </think>, "
                "then output exactly one <patch> block."
            ),
        )
    if use_sft_prefill_prompt:
        user_parts.append(dbbench_sft_runtime_context_section())
    elif args.include_dbbench_runtime_context or args.include_dbbench_world_model:
        user_parts.append(dbbench_runtime_context_section())
    user_parts.extend(["Observed no-harness rollout evidence:", evidence])
    if revision_context:
        user_parts.append(revision_context)
    prompt = [
        {
            "role": "system",
            "content": (
                DBBENCH_PREFILL_SYSTEM_PROMPT
                if use_sft_prefill_prompt
                else (
                    "You are a Harness-R1 engineer. You edit only reusable DBBench harness "
                    "runtime behavior. Do not solve individual database questions, do not "
                    "hard-code table cell values, exact final answers, task ids, or SQL copied "
                    "from evidence. Return the requested think/patch blocks only."
                )
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(user_parts),
        },
    ]
    if args.prompt_prefix_reference is not None:
        prompt = align_prompt_prefix_to_reference(prompt, args.prompt_prefix_reference_messages)
    return {
        "batch_dir": str(batch_dir),
        "prompt": prompt,
        "metadata": metadata
        | {
            "prompt_aligned_with_sft": use_sft_prefill_prompt,
            "legacy_prefill_prompt": bool(args.legacy_prefill_prompt),
            "prompt_prefix_reference": (
                str(args.prompt_prefix_reference)
                if args.prompt_prefix_reference is not None
                else None
            ),
        }
        | (
            {
                "revision_source_root": str(args.revision_source_root),
                "prompt_version": args.revision_prompt_version,
            }
            if args.revision_source_root is not None
            else {}
        ),
    }


def message_text(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
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
    payload: dict[str, Any] = {
        "model": args.engineer_model,
        "messages": request_messages,
        "temperature": args.engineer_temperature,
        "top_p": args.engineer_top_p,
        "max_tokens": args.engineer_max_tokens,
    }
    if args.engineer_assistant_prefill:
        payload["continue_final_message"] = True
    if args.engineer_chat_template_kwargs is not None:
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
            if attempt >= args.engineer_retries:
                raise
            delay = min(
                args.engineer_retry_sleep * (2**attempt),
                args.engineer_retry_max_sleep,
            )
            if isinstance(exc, urllib.error.HTTPError) and exc.headers:
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(float(retry_after), args.engineer_retry_max_sleep)
                    except ValueError:
                        pass
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
        else:
            raw = extract_json_object(response or "")
        return normalize_patch(raw, bench="dbbench"), ""
    except (json.JSONDecodeError, PatchValidationError, RecursionError, TypeError, ValueError) as exc:
        return None, repr(exc)


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
    baseline_pass = int(metadata["baseline_pass"])
    threshold = float(metadata.get("reward_threshold") or 1.0)
    base_rewards = {int(k): float(v) for k, v in (metadata.get("baseline_rewards") or {}).items()}
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
        "--dbbench-worker-python",
        str(args.dbbench_worker_python),
        "--bench",
        "dbbench",
        "--run-id",
        run_id,
        "--dbbench-data-file",
        str(args.dbbench_data_file),
        "--dbbench-max-round",
        str(args.dbbench_max_round),
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
        "--rollout-http-timeout",
        str(args.rollout_http_timeout),
        "--rollout-tool-choice",
        args.rollout_tool_choice,
        "--rollout-chat-template-kwargs",
        json.dumps(args.rollout_chat_template_kwargs),
        "--docker-network-name",
        args.docker_network_name,
        "--dbbench-env-driver",
        args.dbbench_env_driver,
        "--dbbench-manual-mysql-host",
        args.dbbench_manual_mysql_host,
        "--output-root",
        str(eval_root),
        "--harness-patch",
        str(patch_path),
    ]
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
    rewards = collect_rewards(eval_root, bench="dbbench")
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


def no_patch_result(record: dict[str, Any], status: str, parse_error: str = "") -> dict[str, Any]:
    metadata = record["metadata"]
    base_rewards = {int(k): float(v) for k, v in (metadata.get("baseline_rewards") or {}).items()}
    baseline_pass = int(metadata["baseline_pass"])
    batch_size = int(metadata["batch_size"])
    baseline_average_reward = average_reward(base_rewards, baseline_pass, batch_size)
    return {
        "batch_tag": metadata["batch_tag"],
        "start": metadata["start"],
        "end": metadata["end"],
        "valid_patch": False,
        "patch_status": status,
        "parse_error": parse_error,
        "baseline_pass": baseline_pass,
        "patched_pass": baseline_pass,
        "baseline_average_reward": baseline_average_reward,
        "patched_average_reward": baseline_average_reward,
        "batch_size": batch_size,
        "delta_pass": 0,
        "delta_pass_rate": 0.0,
        "delta_average_reward": 0.0,
        "score": 0.0,
    }


def run_one(args: argparse.Namespace, indexed_record: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    ordinal, record = indexed_record
    metadata = record["metadata"]
    batch_tag = str(metadata["batch_tag"])
    batch_dir = args.output_root / batch_tag
    result_path = batch_dir / "result.json"
    if args.resume and result_path.exists():
        existing = load_json(result_path)
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
            print(
                f"[eval-dbbench-gpt] rerunning existing valid patch for {batch_tag}",
                flush=True,
            )
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
            f"[eval-dbbench-gpt] retrying {batch_tag} with patch_status="
            f"{existing.get('patch_status')}",
            flush=True,
        )
    batch_dir.mkdir(parents=True, exist_ok=True)
    dump_json(batch_dir / "metadata.json", metadata)
    dump_json(batch_dir / "prompt.json", record["prompt"])
    if args.patch_source_root is not None:
        source_patch = args.patch_source_root / batch_tag / "patch.json"
        if not source_patch.exists():
            result = no_patch_result(record, "missing_source_patch_treated_as_no_patch")
            dump_json(result_path, result)
            return result
        patch_path = batch_dir / "patch.json"
        patch = load_json(source_patch)
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
    except Exception as exc:
        result = no_patch_result(record, "engineer_request_failed_treated_as_no_patch", repr(exc))
        dump_json(result_path, result)
        return result
    dump_json(batch_dir / "response.json", response_data)
    message = response_data["choices"][0].get("message") or {}
    response = message_text(message)
    for retry_idx in range(args.empty_response_retries):
        if response.strip():
            break
        dump_json(batch_dir / f"empty_response_{retry_idx}.json", response_data)
        try:
            response_data = call_chat(args, record["prompt"])
        except Exception as exc:
            result = no_patch_result(record, "engineer_request_failed_after_empty_response", repr(exc))
            dump_json(result_path, result)
            return result
        dump_json(batch_dir / "response.json", response_data)
        message = response_data["choices"][0].get("message") or {}
        response = message_text(message)
    dump_text(batch_dir / "raw_response.txt", response)
    if message.get("reasoning_content") is not None:
        dump_text(batch_dir / "reasoning_content.txt", message.get("reasoning_content") or "")
    patch, parse_error = parse_patch(response, args.response_protocol)
    if patch is None:
        result = no_patch_result(record, "invalid_patch_treated_as_no_patch", parse_error)
        dump_json(result_path, result)
        return result
    patch_path = batch_dir / "patch.json"
    dump_json(patch_path, patch)
    if args.generate_only:
        result = no_patch_result(record, "generated_only")
        result["valid_patch"] = True
        result["patch_path"] = str(patch_path)
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
    baseline_reward_sum = sum(
        float(r.get("baseline_average_reward", 0.0) or 0.0) * int(r.get("batch_size", 0))
        for r in results
    )
    patched_reward_sum = sum(
        float(r.get("patched_average_reward", 0.0) or 0.0) * int(r.get("batch_size", 0))
        for r in results
    )
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
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument(
        "--prepared-input",
        type=Path,
        default=None,
        help=(
            "Optional preconstructed prompt JSONL. When set, records are read from this file "
            "instead of being rebuilt from --source-run-root."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prompt-prefix-reference",
        type=Path,
        default=None,
        help=(
            "Optional JSONL or LlamaFactory JSON row whose system message and static user "
            "prefix replace the generated prompt prefix. Per-batch evidence is preserved."
        ),
    )
    parser.add_argument("--patch-source-root", type=Path, default=None)
    parser.add_argument("--revision-source-root", type=Path, default=None)
    parser.add_argument("--revision-require-previous", action="store_true")
    parser.add_argument("--revision-prompt-version", default="dbbench_life_multihook_v1_revision_negative_20260625")
    parser.add_argument("--include-dbbench-runtime-context", action="store_true")
    parser.add_argument(
        "--include-dbbench-world-model",
        action="store_true",
        help="Deprecated alias for --include-dbbench-runtime-context.",
    )
    parser.add_argument("--max-revision-regression-traces", type=int, default=6)
    parser.add_argument("--max-revision-improvement-traces", type=int, default=2)
    parser.add_argument("--max-revision-messages-per-trace", type=int, default=34)
    parser.add_argument("--max-previous-response-chars", type=int, default=10000)
    parser.add_argument("--max-previous-patch-chars", type=int, default=10000)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-ids", default="")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--require-failures", action="store_true")
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--max-failures-per-batch", type=int, default=4)
    parser.add_argument("--max-messages-per-trace", type=int, default=22)
    parser.add_argument("--max-tool-result-chars", type=int, default=1200)
    parser.add_argument("--max-overview-chars", type=int, default=3000)
    parser.add_argument("--max-evidence-chars", type=int, default=26000)
    parser.add_argument(
        "--response-protocol",
        choices=["json_patch", "full_think_patch", "full_think_patch_relaxed", "prefill_think_patch"],
        default="full_think_patch",
    )
    parser.add_argument(
        "--add-prefill-thinking-instruction",
        action="store_true",
        help=(
            "Deprecated ablation switch for the legacy DBBench prefill prompt. "
            "The default prefill prompt is already SFT-aligned."
        ),
    )
    parser.add_argument(
        "--legacy-prefill-prompt",
        action="store_true",
        help="Use the old DBBench prefill prompt instead of the SFT-aligned training prompt.",
    )
    parser.add_argument("--engineer-base-url", default="https://relay.shuai-ederson-clow.xyz/v1")
    parser.add_argument("--engineer-model", default="gpt-5.5")
    parser.add_argument("--engineer-api-key", default="")
    parser.add_argument("--engineer-api-key-env", default="HARNESS_R1_GPT_API_KEY")
    parser.add_argument("--engineer-timeout", type=int, default=1200)
    parser.add_argument("--engineer-max-tokens", type=int, default=12000)
    parser.add_argument("--engineer-concurrency", type=int, default=1)
    parser.add_argument("--engineer-temperature", type=float, default=0.0)
    parser.add_argument("--engineer-top-p", type=float, default=1.0)
    parser.add_argument("--engineer-chat-template-kwargs", type=json.loads, default=None)
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
    parser.add_argument("--engineer-reasoning-effort", default="high")
    parser.add_argument("--engineer-retries", type=int, default=1)
    parser.add_argument("--engineer-retry-sleep", type=float, default=10.0)
    parser.add_argument("--engineer-retry-max-sleep", type=float, default=120.0)
    parser.add_argument("--empty-response-retries", type=int, default=1)
    parser.add_argument("--target-base-url", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--target-model", default="Qwen3.5-9B")
    parser.add_argument("--target-agent-name", default="qwen35-9b-local8110-nothink")
    parser.add_argument("--rollout-concurrency", type=int, default=10)
    parser.add_argument("--rollout-max-tokens", type=int, default=4096)
    parser.add_argument("--rollout-http-timeout", type=int, default=900)
    parser.add_argument("--rollout-tool-choice", default="auto")
    parser.add_argument("--rollout-chat-template-kwargs", type=json.loads, default={"enable_thinking": False})
    parser.add_argument("--rollout-disable-parallel-tool-calls", action="store_true")
    parser.add_argument("--rollout-single-tool-call-only", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--rollout-timeout", type=int, default=3600)
    parser.add_argument("--eval-timeout", type=int, default=4200)
    parser.add_argument("--controller-port-base", type=int, default=6300)
    parser.add_argument("--worker-port-base", type=int, default=8200)
    parser.add_argument("--docker-network-name", default="harness_r1_agentbench")
    parser.add_argument("--dbbench-env-driver", choices=["docker", "manual"], default="docker")
    parser.add_argument("--dbbench-manual-mysql-host", default="127.0.0.1")
    parser.add_argument("--dbbench-data-file", type=Path, default=AGENTBENCH_DIR / "data/dbbench/db_out_new.jsonl")
    parser.add_argument("--dbbench-max-round", type=int, default=20)
    parser.add_argument("--agentbench-dir", type=Path, default=AGENTBENCH_DIR)
    parser.add_argument("--agentbench-python", type=Path, default=DEFAULT_AGENTBENCH_PYTHON)
    parser.add_argument("--dbbench-worker-python", type=Path, default=DEFAULT_AGENTBENCH_PYTHON)
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
        help=(
            "When used with --resume, reuse the existing patch.json and rerun only "
            "valid patches whose previous eval_status was eval_failed_treated_as_no_patch."
        ),
    )
    args = parser.parse_args()
    if args.engineer_api_key_env:
        args.engineer_api_key = os.environ.get(args.engineer_api_key_env, args.engineer_api_key)
    if args.source_run_root is not None:
        args.source_run_root = args.source_run_root.expanduser().resolve()
    if args.prepared_input is not None:
        args.prepared_input = args.prepared_input.expanduser().resolve()
    if args.source_run_root is None and args.prepared_input is None:
        parser.error("one of --source-run-root or --prepared-input is required")
    args.output_root = args.output_root.expanduser().resolve()
    if args.prompt_prefix_reference is not None:
        args.prompt_prefix_reference = args.prompt_prefix_reference.expanduser().resolve()
        args.prompt_prefix_reference_messages = prompt_messages_from_reference(
            args.prompt_prefix_reference
        )
    else:
        args.prompt_prefix_reference_messages = []
    if args.patch_source_root is not None:
        args.patch_source_root = args.patch_source_root.expanduser().resolve()
    if args.revision_source_root is not None:
        args.revision_source_root = args.revision_source_root.expanduser().resolve()
    args.agentbench_dir = args.agentbench_dir.expanduser().resolve()
    # Preserve virtualenv symlink paths. Resolving /data/cache/.../bin/python to
    # /usr/bin/python3.10 loses the venv layout used to discover site-packages.
    args.agentbench_python = args.agentbench_python.expanduser().absolute()
    args.dbbench_worker_python = args.dbbench_worker_python.expanduser().absolute()
    args.dbbench_data_file = args.dbbench_data_file.expanduser().resolve()
    return args


def select_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prepared_input is not None:
        records = [
            json.loads(line)
            for line in args.prepared_input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit:
            records = records[: args.limit]
        return records
    root = args.source_run_root / "dbbench"
    if not root.exists():
        raise FileNotFoundError(f"DBBench source root not found: {root}")
    selected: set[int] | None = None
    if args.batch_ids.strip():
        selected = {int(x) for x in re.split(r"[,\\s]+", args.batch_ids.strip()) if x}
    records: list[dict[str, Any]] = []
    for batch_dir in sorted(root.glob("batch_*"), key=batch_id_from_dir):
        batch_id = batch_id_from_dir(batch_dir)
        if selected is not None and batch_id not in selected:
            continue
        record = make_record(args, batch_dir)
        if record is not None:
            records.append(record)
        if args.limit and len(records) >= args.limit:
            break
    return records


def main() -> int:
    args = parse_args()
    records = select_records(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if run_config.get("engineer_api_key"):
        run_config["engineer_api_key"] = "***"
    dump_json(args.output_root / "run_config.json", run_config)
    dump_json(args.output_root / "selected_metadata.json", [r["metadata"] for r in records])
    started = time.time()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.engineer_concurrency) as pool:
        futures = [pool.submit(run_one, args, item) for item in enumerate(records)]
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            dump_json(args.output_root / "summary.partial.json", summarize(results) | {"completed": i})
            print(
                f"[eval-dbbench-gpt] {i}/{len(records)} {result.get('batch_tag')} "
                f"valid={result.get('valid_patch')} delta={result.get('delta_pass')}",
                flush=True,
            )
    results = sorted(results, key=lambda r: int(r.get("start", 0)))
    summary = summarize(results)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    summary["input_records"] = len(records)
    dump_json(args.output_root / "results.json", results)
    dump_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
