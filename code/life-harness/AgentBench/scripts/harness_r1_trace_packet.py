#!/usr/bin/env python3
"""Extract compact failure traces for direct Harness-R1 patch proposal.

This intentionally does not ask a debugger model to assign failure modes.  It
only selects non-full-reward rollouts and writes a bounded, sanitized trace
packet that a harness engineer model can analyze directly.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRODUCT_ID_RE = re.compile(r"\b[bB]0[0-9A-Za-z]{8}\b")
PATH_RE = re.compile(r"/mnt/[^\s`]+")
PRIVATE_PATH_RE = re.compile(r"\b/(?:root|home|tmp)/[^\s`'\"),;]+")


@dataclass
class TraceCase:
    index: int
    batch_id: int
    reward: float
    status: str
    messages: list[dict[str, Any]]


def sanitize(text: str) -> str:
    text = PRODUCT_ID_RE.sub("[PRODUCT_ID]", text)
    text = PATH_RE.sub("[PATH]", text)
    text = PRIVATE_PATH_RE.sub("[PATH]", text)
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def reward_of(row: dict[str, Any]) -> float:
    result = ((row.get("output") or {}).get("result") or {})
    try:
        return float(result.get("reward", 0.0) or 0.0)
    except Exception:
        return 0.0


def status_of(row: dict[str, Any]) -> str:
    return str((row.get("output") or {}).get("status") or row.get("status") or "unknown")


def messages_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = ((row.get("output") or {}).get("result") or {})
    messages = result.get("openai_messages") if isinstance(result, dict) else None
    if not isinstance(messages, list) or not messages:
        messages = (row.get("output") or {}).get("history") or []
    return [item for item in messages if isinstance(item, dict)]


def load_cases(run_root: Path, bench: str, batch_ids: set[int] | None, threshold: float) -> list[TraceCase]:
    cases: list[TraceCase] = []
    for batch_dir in sorted((run_root / bench).glob("batch_*")):
        try:
            batch_id = int(batch_dir.name.split("_", 1)[1])
        except Exception:
            continue
        if batch_ids is not None and batch_id not in batch_ids:
            continue
        for runs_path in sorted(batch_dir.glob("rollout/*/*/runs.jsonl")):
            for row in read_jsonl(runs_path):
                reward = reward_of(row)
                if reward >= threshold:
                    continue
                idx = row.get("index")
                if not isinstance(idx, int):
                    continue
                messages = messages_of(row)
                if not messages:
                    continue
                cases.append(
                    TraceCase(
                        index=idx,
                        batch_id=batch_id,
                        reward=reward,
                        status=status_of(row),
                        messages=messages,
                    )
                )
    return cases


def tool_call_lines(message: dict[str, Any]) -> list[str]:
    calls = message.get("tool_calls") or []
    lines = []
    if not isinstance(calls, list):
        return lines
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args = fn.get("arguments")
        lines.append(f"ASSISTANT_TOOL: {name} {sanitize(str(args))}")
    return lines


def _tool_calls(message: dict[str, Any]) -> list[tuple[str, str]]:
    calls = message.get("tool_calls") or []
    out = []
    if not isinstance(calls, list):
        return out
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        value = ""
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict) and parsed:
                    value = str(next(iter(parsed.values())))
                else:
                    value = raw_args
            except Exception:
                value = raw_args
        elif isinstance(raw_args, dict) and raw_args:
            value = str(next(iter(raw_args.values())))
        out.append((name, sanitize(value).strip()))
    return out


def runtime_signal_lines(case: TraceCase, bench: str = "webshop") -> list[str]:
    # Action-derived signals counted from tool_calls (never regex-scraped from text:
    # the old price/budget signals scanned user text only, missed tool-observation
    # prices, and are gone). Signals branch by bench. The webshop branch is kept
    # byte-identical to the original. The alfworld branch counts take_action plus
    # observation-derived no-ops/repeats, mirroring the harness_r1_patch predicates
    # nothing_happens_repeated / same_observation_repeated.
    actions: list[tuple[str, str]] = []
    assistant_text_turns = 0
    for message in case.messages:
        role = message.get("role")
        if role == "assistant":
            calls = _tool_calls(message)
            if calls:
                actions.extend(calls)
            elif str(message.get("content") or "").strip():
                assistant_text_turns += 1

    action_values = [f"{name}:{value.lower()}" for name, value in actions]
    repeated_actions = {k: v for k, v in Counter(action_values).items() if v > 1}

    if bench == "alfworld":
        return _alfworld_signal_lines(case, actions, assistant_text_turns, repeated_actions)

    searches = [value for name, value in actions if name == "search_action"]
    clicks = [value for name, value in actions if name == "click_action"]
    repeated_searches = {k: v for k, v in Counter(searches).items() if v > 1}
    repeated_clicks = {k: v for k, v in Counter(clicks).items() if v > 1}
    buy_now_count = sum(1 for value in clicks if value.strip().lower() == "buy now")

    lines = [
        f"- tool_actions: `{len(actions)}` total; searches `{len(searches)}`; clicks `{len(clicks)}`; buy_now `{buy_now_count}`",
    ]
    if assistant_text_turns:
        lines.append(f"- assistant_text_without_tool_turns: `{assistant_text_turns}`")
    if repeated_searches:
        rendered = "; ".join(f"{sanitize(k) or '<empty>'} x{v}" for k, v in sorted(repeated_searches.items())[:3])
        lines.append(f"- repeated_search_terms: {rendered}")
    if repeated_clicks:
        rendered = "; ".join(f"{sanitize(k) or '<empty>'} x{v}" for k, v in sorted(repeated_clicks.items())[:3])
        lines.append(f"- repeated_click_values: {rendered}")
    elif repeated_actions:
        rendered = "; ".join(f"{sanitize(k) or '<empty>'} x{v}" for k, v in sorted(repeated_actions.items())[:3])
        lines.append(f"- repeated_action_values: {rendered}")
    if "task limit" in case.status.lower():
        lines.append("- terminal_signal: `task_limit_reached`")
    return lines


def _alfworld_signal_lines(
    case: TraceCase,
    actions: list[tuple[str, str]],
    assistant_text_turns: int,
    repeated_actions: dict[str, int],
) -> list[str]:
    # ALFWorld surfaces failures in the OBSERVATION text, not the action name:
    # an inadmissible / no-op action returns "Nothing happens". Repeated identical
    # observations indicate the agent is stuck. These are the diagnostic core for
    # alfworld harness edits (see harness_r1_patch CONDITION_PREDICATES).
    observations = [
        str(message.get("content") or "")
        for message in case.messages
        if message.get("role") == "tool"
    ]
    no_op_count = sum(1 for obs in observations if "nothing happens" in obs.lower())
    norm_obs = [re.sub(r"\s+", " ", sanitize(obs).strip().lower())[:200] for obs in observations]
    repeated_obs = {k: v for k, v in Counter(norm_obs).items() if v > 1 and k}
    takes = [value for name, value in actions if name == "take_action"]

    lines = [
        f"- tool_actions: `{len(actions)}` total; take_action `{len(takes)}`; no_op_observations `{no_op_count}`",
    ]
    if assistant_text_turns:
        lines.append(f"- assistant_text_without_tool_turns: `{assistant_text_turns}`")
    if repeated_actions:
        rendered = "; ".join(f"{sanitize(k) or '<empty>'} x{v}" for k, v in sorted(repeated_actions.items())[:3])
        lines.append(f"- repeated_action_values: {rendered}")
    if repeated_obs:
        lines.append(f"- repeated_observations: `{sum(repeated_obs.values())}` total; distinct `{len(repeated_obs)}`")
    if "task limit" in case.status.lower():
        lines.append("- terminal_signal: `task_limit_reached`")
    return lines


def trim_observation(text: str, max_chars: int) -> str:
    text = sanitize(str(text or "")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].rstrip()
    tail = text[-max_chars // 2 :].lstrip()
    return head + "\n...[middle truncated]...\n" + tail


def format_trace(case: TraceCase, max_obs_chars: int, max_assistant_chars: int, bench: str = "webshop") -> str:
    lines = [
        f"## Trace case",
        "",
        f"- final_reward: `{case.reward:.4f}`",
        f"- status: `{case.status}`",
        f"- turns: `{len(case.messages)}`",
        "",
        "Runtime signals:",
        *runtime_signal_lines(case, bench),
        "",
    ]
    for pos, message in enumerate(case.messages):
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role == "user":
            text = sanitize(str(content or "")).strip()
            keep = pos < 3 or "Evaluation metadata" in text
            if keep:
                lines.append("USER:")
                lines.append(trim_observation(text, max_obs_chars))
                lines.append("")
        elif role == "assistant":
            call_lines = tool_call_lines(message)
            if call_lines:
                lines.extend(call_lines)
                lines.append("")
            else:
                text = sanitize(str(content or "")).strip()
                if text:
                    lines.append("ASSISTANT_TEXT:")
                    lines.append(trim_observation(text, max_assistant_chars))
                    lines.append("")
        elif role == "tool":
            lines.append("OBSERVATION:")
            lines.append(trim_observation(str(content or ""), max_obs_chars))
            lines.append("")
    return "\n".join(lines).rstrip()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def select_cases(cases: list[TraceCase], max_traces: int, strategy: str) -> list[TraceCase]:
    if strategy == "lowest_reward":
        return sorted(cases, key=lambda c: (c.reward, c.batch_id, c.index))[:max_traces]
    if strategy == "round_robin":
        by_batch: dict[int, list[TraceCase]] = {}
        for case in sorted(cases, key=lambda c: (c.batch_id, c.reward, c.index)):
            by_batch.setdefault(case.batch_id, []).append(case)
        out: list[TraceCase] = []
        while len(out) < max_traces and any(by_batch.values()):
            for batch_id in sorted(by_batch):
                bucket = by_batch[batch_id]
                if bucket:
                    out.append(bucket.pop(0))
                    if len(out) >= max_traces:
                        break
        return out
    raise ValueError(f"unknown strategy: {strategy}")


def build_packet(args: argparse.Namespace) -> str:
    cases = load_cases(args.run_root.resolve(), args.bench, args.batch_ids, args.reward_threshold)
    selected = select_cases(cases, args.max_traces, args.strategy)

    if args.bench == "alfworld":
        no_leak_line = (
            "- Do not encode product ids, exact object/receptacle instance ids "
            "(e.g. trailing numbers like `sinkbasin 1`), task indices, or benchmark answers."
        )
    else:
        no_leak_line = "- Do not encode product ids, exact product names, task indices, or benchmark answers."

    lines = [
        "# Harness-R1 Direct Failure Trace Packet",
        "",
        "These are deterministic extracts from no-harness rollout failures.",
        "No debugger model has labeled the failure modes. Infer recurring harness-edit opportunities from the traces.",
        "",
        "## Scope",
        "",
        f"- Benchmark: `{args.bench}`",
        f"- Source run: `{sanitize(str(args.run_root))}`",
        f"- Candidate failure traces found: `{len(cases)}`",
        f"- Traces included: `{len(selected)}`",
        f"- Selection strategy: `{args.strategy}`",
        "",
        "## Instructions for Harness Engineer",
        "",
        "- Infer recurring, general failure modes from the traces before writing a patch.",
        no_leak_line,
        "- Prefer typed Harness-R1 actions that are auditable and general.",
        "- Use hard blocking only when the runtime condition is clearly observable; otherwise prefer short hints or reusable skills.",
        "",
        "## Failure Traces",
        "",
    ]

    used_tokens = approx_tokens("\n".join(lines))
    included = 0
    for case in selected:
        block = format_trace(case, args.max_obs_chars, args.max_assistant_chars, args.bench)
        block_tokens = approx_tokens(block)
        if included and used_tokens + block_tokens > args.max_approx_tokens:
            break
        lines.append(block)
        lines.append("")
        used_tokens += block_tokens
        included += 1
    lines.append(f"Included traces after budget: `{included}`")
    lines.append(f"Approx input tokens by char/4: `{used_tokens}`")
    return "\n".join(lines).rstrip() + "\n"


def parse_batch_ids(raw: list[str]) -> set[int] | None:
    if not raw:
        return None
    out = set()
    for part in raw:
        for item in part.split(","):
            item = item.strip()
            if item:
                out.add(int(item))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract direct failure traces for Harness-R1 patching.")
    parser.add_argument("--bench", choices=["webshop", "alfworld"], default="webshop")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-ids", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--max-traces", type=int, default=20)
    parser.add_argument("--strategy", choices=["round_robin", "lowest_reward"], default="round_robin")
    parser.add_argument("--max-obs-chars", type=int, default=1200)
    parser.add_argument("--max-assistant-chars", type=int, default=600)
    parser.add_argument("--max-approx-tokens", type=int, default=18000)
    args = parser.parse_args()
    args.batch_ids = parse_batch_ids(args.batch_ids)
    text = build_packet(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[harness-r1-trace-packet] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
