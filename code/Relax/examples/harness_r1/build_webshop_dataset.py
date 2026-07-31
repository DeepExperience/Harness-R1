# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build single-step Harness-R1 WebShop RL prompts from no-harness rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTBENCH_SCRIPTS = REPO_ROOT / "code/life-harness/AgentBench/scripts"
sys.path.insert(0, str(AGENTBENCH_SCRIPTS))

from harness_r1_patch import schema_prompt  # noqa: E402
from harness_r1_trace_packet import build_packet  # noqa: E402
from harness_r1_webshop_identity import (  # noqa: E402
    WebShopIdentityError,
    identity_metadata_from_rows,
)


SYSTEM_PROMPT = (
    "You are a Harness-R1 harness engineer. Given a batch of failed WebShop "
    "rollout traces, first reason briefly, then produce one general typed "
    "harness patch. Output exactly two blocks: <think>...</think> followed by "
    "<patch>...</patch>. The patch block must contain exactly one JSON object."
)
QWEN35_PREFILL_SYSTEM_PROMPT = (
    "You are a Harness-R1 harness engineer. Given a batch of failed WebShop "
    "rollout traces, analyze the recurring failures, then produce one general "
    "typed harness patch as a single JSON object inside a <patch>...</patch> "
    "block. The patch JSON must have a top-level actions array."
)
QWEN35_PREFILL_USER_SENTENCE = (
    "The chat template has already opened the assistant thinking block. "
    "Continue concise recurring-failure reasoning, close it with </think>, "
    "then output exactly one <patch> block."
)
DEFAULT_PROMPT_VERSION = "summary_think_v2_20260604"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    except (TypeError, ValueError):
        return 0.0


def iter_run_roots(source_root: Path) -> list[Path]:
    if (source_root / "run_config.json").exists():
        return [source_root]
    roots = [
        path
        for path in sorted(source_root.iterdir())
        if path.is_dir() and (path / "run_config.json").exists()
    ]
    if not roots:
        raise FileNotFoundError(f"no batch run_config.json found under {source_root}")
    return roots


@dataclass(frozen=True)
class BatchSource:
    run_root: Path
    spec: dict[str, Any]
    output_dir: Path
    batch_id: int
    batch_tag: str


def resolve_output_dir(run_root: Path, bench: str, spec: dict[str, Any]) -> Path:
    output_dir = Path(spec["output_dir"])
    if output_dir.exists():
        return output_dir
    batch_id = int(spec.get("batch_id", 0))
    fallback = run_root / bench / f"batch_{batch_id:03d}"
    if fallback.exists():
        return fallback
    if not output_dir.is_absolute():
        return run_root / bench / "batch_000"
    return output_dir


def iter_batch_sources(source_root: Path, bench: str) -> list[BatchSource]:
    sources: list[BatchSource] = []
    for run_root in iter_run_roots(source_root):
        config = read_json(run_root / "run_config.json")
        specs = [x for x in config.get("specs", []) if x.get("bench") == bench]
        if not specs:
            continue
        multi_spec = len(specs) > 1
        for spec in specs:
            batch_id = int(spec.get("batch_id", 0))
            batch_tag = run_root.name
            if multi_spec:
                batch_tag = f"{run_root.name}_b{batch_id:03d}"
            sources.append(
                BatchSource(
                    run_root=run_root,
                    spec=spec,
                    output_dir=resolve_output_dir(run_root, bench, spec),
                    batch_id=batch_id,
                    batch_tag=batch_tag,
                )
            )
    if not sources:
        raise ValueError(f"no {bench} specs found under {source_root}")
    return sources


def load_one_spec(run_root: Path, bench: str) -> tuple[dict[str, Any], Path]:
    config = read_json(run_root / "run_config.json")
    specs = [x for x in config.get("specs", []) if x.get("bench") == bench]
    if len(specs) != 1:
        raise ValueError(f"{run_root} must contain exactly one {bench} spec, got {len(specs)}")
    spec = specs[0]
    output_dir = resolve_output_dir(run_root, bench, spec)
    return spec, output_dir


def load_rollout_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("rollout/*/*/runs.jsonl")):
        rows.extend(read_jsonl(path))
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = row.get("index")
        if isinstance(idx, int):
            by_index[idx] = row
    return [by_index[k] for k in sorted(by_index)]


def make_trace_packet(args: argparse.Namespace, run_root: Path, batch_id: int) -> str:
    packet_args = argparse.Namespace(
        run_root=run_root,
        bench=args.bench,
        batch_ids={batch_id},
        reward_threshold=args.reward_threshold,
        max_traces=args.max_traces,
        strategy=args.strategy,
        max_obs_chars=args.max_obs_chars,
        max_assistant_chars=args.max_assistant_chars,
        max_approx_tokens=args.max_approx_tokens,
    )
    return build_packet(packet_args)


def build_prompt(
    packet: str,
    bench: str,
    response_protocol: str = "full_think_patch",
    schema_style: str = "example",
) -> list[dict[str, str]]:
    task_scope = "You will edit only the reusable WebShop harness, not the task answers."
    if response_protocol == "prefill_think_patch":
        task_scope = "You will edit only the reusable WebShop harness, not task answers."
    user_parts = [task_scope]
    if response_protocol == "prefill_think_patch":
        user_parts.append(QWEN35_PREFILL_USER_SENTENCE)
    user_parts.extend(
        [
            schema_prompt(bench, response_protocol=response_protocol, schema_style=schema_style),
            "Observed no-harness rollout evidence:",
            packet,
        ]
    )
    user = "\n\n".join(user_parts)
    system_prompt = SYSTEM_PROMPT
    if response_protocol == "prefill_think_patch":
        system_prompt = QWEN35_PREFILL_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def build_record(args: argparse.Namespace, source: BatchSource) -> dict[str, Any] | None:
    spec = source.spec
    output_dir = source.output_dir
    status_path = output_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("status") != "ok" and not args.include_incomplete:
        return None

    rows = load_rollout_rows(output_dir)
    batch_size = int(spec["end"]) - int(spec["start"])
    if len(rows) < batch_size and not args.include_incomplete:
        return None

    rewards = {int(row["index"]): reward_of(row) for row in rows if isinstance(row.get("index"), int)}
    baseline_pass = sum(value >= args.reward_threshold for value in rewards.values())
    failures = len(rewards) - baseline_pass
    if failures < args.min_failures:
        return None

    packet = make_trace_packet(args, source.run_root, source.batch_id)
    prompt_schema = schema_prompt(
        args.bench,
        response_protocol=args.response_protocol,
        schema_style=args.schema_style,
    )
    if args.max_prompt_chars and len(packet) + len(prompt_schema) > args.max_prompt_chars:
        return None

    start = int(spec["start"])
    end = int(spec["end"])
    run_config = read_json(source.run_root / "run_config.json")
    identity_metadata: dict[str, Any] = {}
    try:
        if run_config.get("webshop_goal_seed") is None:
            raise WebShopIdentityError(
                f"{source.run_root}/run_config.json has no webshop_goal_seed"
            )
        identity_metadata = identity_metadata_from_rows(
            rows,
            start=start,
            end=end,
            goal_seed=int(run_config["webshop_goal_seed"]),
        )
    except WebShopIdentityError:
        if not args.allow_missing_task_manifest:
            raise

    metadata = {
        "benchmark": args.bench,
        "batch_tag": source.batch_tag,
        "source_run_root": str(source.run_root),
        "source_rollout_dir": str(output_dir),
        "start": start,
        "end": end,
        "batch_size": batch_size,
        "baseline_pass": int(baseline_pass),
        "baseline_pass_rate": baseline_pass / max(1, batch_size),
        "baseline_rewards": rewards,
        "reward_threshold": args.reward_threshold,
        "target_agent_name": args.target_agent_name,
        "target_model": args.target_model,
        "prompt_version": args.prompt_version,
        "response_protocol": args.response_protocol,
        "schema_style": args.schema_style,
        **identity_metadata,
    }
    if not identity_metadata and run_config.get("webshop_goal_seed") is not None:
        metadata["webshop_goal_seed"] = int(run_config["webshop_goal_seed"])
    return {
        "prompt": build_prompt(
            packet,
            args.bench,
            response_protocol=args.response_protocol,
            schema_style=args.schema_style,
        ),
        "label": "",
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench", choices=["webshop"], default="webshop")
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--min-failures", type=int, default=1)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--allow-missing-task-manifest",
        action="store_true",
        help=(
            "Allow legacy rollout inputs without strict per-task manifests. "
            "Do not use this flag for new SFT/RL data."
        ),
    )
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=20)
    parser.add_argument("--strategy", choices=["round_robin", "lowest_reward"], default="round_robin")
    parser.add_argument("--max-obs-chars", type=int, default=1200)
    parser.add_argument("--max-assistant-chars", type=int, default=600)
    parser.add_argument("--max-approx-tokens", type=int, default=18000)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--target-agent-name", default="qwen25-7b-rollout")
    parser.add_argument("--target-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument(
        "--response-protocol",
        choices=["full_think_patch", "prefill_think_patch"],
        default="full_think_patch",
    )
    parser.add_argument(
        "--schema-style",
        choices=[
            "example",
            "schema_only",
            "qwen25_sft",
            "webshop_code_only",
            "webshop_action_balanced_v1",
            "webshop_life_multihook_v1",
        ],
        default="example",
        help=(
            "Prompt schema style. Use qwen25_sft to reproduce the prompt used by "
            "the Qwen2.5 long-think clean SFT data; use schema_only for qwen3.5 "
            "no-real-strategy-example prompts; use webshop_code_only for code-hook "
            "patch experiments; use webshop_action_balanced_v1 for lifecycle-aware "
            "hook selection without a fixed strategy example; use "
            "webshop_life_multihook_v1 for the ALFWorld-like WebShop multihook "
            "protocol."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    records = []
    skipped = 0
    for source in iter_batch_sources(source_root, args.bench):
        record = build_record(args, source)
        if record is None:
            skipped += 1
            continue
        records.append(record)
        if args.limit_batches and len(records) >= args.limit_batches:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"wrote {len(records)} records to {args.output}; skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
