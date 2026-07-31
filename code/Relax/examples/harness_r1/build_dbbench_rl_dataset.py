#!/usr/bin/env python3
"""Build DBBench Harness-R1 RL prompt JSONL from synthetic rebatch outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eval_dbbench_gpt_patches import batch_id_from_dir, make_record


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "outputs/dbbench_harness_r1/rebatch_seed20260625/rl_train_bs10"
DEFAULT_VALID_ROOT = REPO_ROOT / "outputs/dbbench_harness_r1/rebatch_seed20260625/validation_100_bs10"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_chars(row: dict[str, Any]) -> int:
    return len(json.dumps(row.get("prompt", ""), ensure_ascii=False))


def metadata_from_record(record: dict[str, Any], args: argparse.Namespace, source_root: Path) -> dict[str, Any]:
    metadata = dict(record["metadata"])
    batch_dir = Path(record["batch_dir"])
    rewards = metadata.get("baseline_rewards") if isinstance(metadata.get("baseline_rewards"), dict) else {}
    reward_values = []
    for value in rewards.values():
        try:
            reward_values.append(float(value))
        except (TypeError, ValueError):
            pass
    batch_size = int(metadata.get("batch_size") or 0)
    baseline_pass = int(metadata.get("baseline_pass") or 0)
    metadata.update(
        {
            "benchmark": "dbbench",
            "source_run_root": str(source_root),
            "source_rollout_dir": str(batch_dir),
            "target_agent_name": args.target_agent_name,
            "target_model": args.target_model,
            "target_base_url": args.target_base_url,
            "target_disable_thinking": True,
            "response_protocol": args.response_protocol,
            "schema_style": "dbbench_life_multihook_v1",
            "prompt_version": args.prompt_version,
            "reward_threshold": args.reward_threshold,
            "baseline_pass_rate": baseline_pass / max(1, batch_size),
            "baseline_average_reward": (
                sum(reward_values) / len(reward_values)
                if reward_values
                else baseline_pass / max(1, batch_size)
            ),
            "dbbench_data_file": str(args.dbbench_data_file),
            "dbbench_max_round": args.dbbench_max_round,
        }
    )
    return metadata


def build_rows(args: argparse.Namespace, source_root: Path) -> list[dict[str, Any]]:
    helper_args = SimpleNamespace(
        batch_size=args.batch_size,
        reward_threshold=args.reward_threshold,
        max_failures_per_batch=args.max_failures_per_batch,
        require_failures=args.require_failures,
        max_overview_chars=args.max_overview_chars,
        max_messages_per_trace=args.max_messages_per_trace,
        max_tool_result_chars=args.max_tool_result_chars,
        max_evidence_chars=args.max_evidence_chars,
        response_protocol=args.response_protocol,
        legacy_prefill_prompt=False,
        add_prefill_thinking_instruction=args.add_prefill_thinking_instruction,
        include_dbbench_runtime_context=args.include_dbbench_runtime_context,
        include_dbbench_world_model=False,
        revision_source_root=None,
        revision_require_previous=False,
        revision_prompt_version="",
        max_revision_regression_traces=0,
        max_revision_improvement_traces=0,
        max_revision_messages_per_trace=0,
        max_previous_response_chars=0,
        max_previous_patch_chars=0,
    )
    root = source_root / "dbbench"
    if not root.exists():
        raise FileNotFoundError(f"DBBench source root not found: {root}")
    selected = None
    if args.batch_ids.strip():
        selected = {int(item) for item in args.batch_ids.replace(",", " ").split() if item.strip()}

    rows: list[dict[str, Any]] = []
    for batch_dir in sorted(root.glob("batch_*"), key=batch_id_from_dir):
        batch_id = batch_id_from_dir(batch_dir)
        if selected is not None and batch_id not in selected:
            continue
        record = make_record(helper_args, batch_dir)
        if record is None:
            continue
        metadata = metadata_from_record(record, args, source_root)
        metadata["rl_dataset_index"] = len(rows)
        metadata["rl_dataset"] = args.output.name
        row = {
            "prompt": record["prompt"],
            "label": "",
            "metadata": metadata,
        }
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break
    return rows


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, source_root: Path) -> dict[str, Any]:
    prompt_lengths = [prompt_chars(row) for row in rows]
    batch_sizes = [int((row.get("metadata") or {}).get("batch_size") or 0) for row in rows]
    baseline_pass = sum(int((row.get("metadata") or {}).get("baseline_pass") or 0) for row in rows)
    total_tasks = sum(batch_sizes)
    stats = {
        "output": str(args.output),
        "source_run_root": str(source_root),
        "prompt_version": args.prompt_version,
        "response_protocol": args.response_protocol,
        "schema_style": "dbbench_life_multihook_v1",
        "num_rows": len(rows),
        "num_tasks": total_tasks,
        "recommended_num_rollout": len(rows),
        "batch_size_min": min(batch_sizes) if batch_sizes else 0,
        "batch_size_max": max(batch_sizes) if batch_sizes else 0,
        "baseline_pass": baseline_pass,
        "baseline_pass_rate": baseline_pass / max(1, total_tasks),
        "prompt_chars_min": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_chars_avg": round(statistics.mean(prompt_lengths), 2) if prompt_lengths else 0,
        "prompt_chars_p50": statistics.median(prompt_lengths) if prompt_lengths else 0,
        "prompt_chars_max": max(prompt_lengths) if prompt_lengths else 0,
        "require_failures": args.require_failures,
        "max_failures_per_batch": args.max_failures_per_batch,
        "max_evidence_chars": args.max_evidence_chars,
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-ids", default="")
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--max-failures-per-batch", type=int, default=4)
    parser.add_argument("--require-failures", action="store_true")
    parser.add_argument("--max-messages-per-trace", type=int, default=22)
    parser.add_argument("--max-tool-result-chars", type=int, default=1200)
    parser.add_argument("--max-overview-chars", type=int, default=3000)
    parser.add_argument("--max-evidence-chars", type=int, default=26000)
    parser.add_argument("--response-protocol", choices=["json_patch", "full_think_patch", "prefill_think_patch"], default="prefill_think_patch")
    parser.add_argument("--add-prefill-thinking-instruction", action="store_true")
    parser.add_argument("--include-dbbench-runtime-context", action="store_true", default=True)
    parser.add_argument("--target-base-url", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--target-model", default="Qwen3.5-9B")
    parser.add_argument("--target-agent-name", default="qwen35-9b-dbbench-nothink")
    parser.add_argument("--dbbench-data-file", type=Path, default=REPO_ROOT / "external/harness_evolution/life-harness/AgentBench/data/dbbench/db_out_new.jsonl")
    parser.add_argument("--dbbench-max-round", type=int, default=20)
    parser.add_argument("--prompt-version", default="dbbench_rltrain_qwen35_9b_nothink_life_multihook_prefill_20260626")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_run_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.dbbench_data_file = args.dbbench_data_file.expanduser().resolve()
    rows = build_rows(args, source_root)
    if not rows:
        raise SystemExit("no DBBench RL rows were built")
    write_jsonl(args.output, rows)
    stats = summarize(rows, args, source_root)
    write_json(args.output.with_suffix(args.output.suffix + ".stats.json"), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
