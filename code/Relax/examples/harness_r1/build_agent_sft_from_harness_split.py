#!/usr/bin/env python3
"""Build an agent-SFT baseline dataset from the same Harness-R1 SFT split.

The input is a harness-engineer SFT JSON file.  Each row's metadata points back
to the no-harness rollout batch used as evidence.  This script resolves those
source batches, extracts target-agent trajectories from the full source split,
and also samples a patch-count-matched successful subset for ablations.

Outputs are intentionally written to a new directory and never overwrite the
original harness SFT/RL data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HARNESS_SFT = (
    REPO_ROOT
    / "data/llamafactory/mixed_alfworld_webshop_dbbench_codepatch_webshop381_aw248_db248_gptthink_qwen35_prefill_seed20260628/"
    / "mixed_alfworld_webshop_dbbench_codepatch_webshop381_aw248_db248_gptthink_qwen35_prefill_seed20260628.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data/agent_sft_baselines/same_harness_sft_split_qwen35_9b_success_openai_seed20260629"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        return json.loads(metadata)
    if isinstance(metadata, dict):
        return metadata
    raise ValueError(f"unsupported metadata type: {type(metadata)!r}")


def as_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def reward_of(run_row: dict[str, Any]) -> float:
    result = ((run_row.get("output") or {}).get("result") or {})
    try:
        return float(result.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def openai_messages_of(run_row: dict[str, Any]) -> list[dict[str, Any]]:
    result = ((run_row.get("output") or {}).get("result") or {})
    messages = result.get("openai_messages")
    if not isinstance(messages, list) or not messages:
        return []
    return messages


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep OpenAI chat fields needed for tool-call SFT and drop noisy nulls."""
    keep = {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "reasoning_content",
    }
    out = {k: v for k, v in message.items() if k in keep and v is not None}
    if out.get("role") == "assistant" and "content" not in out:
        out["content"] = ""
    return out


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def canonical_task_id(benchmark: str, run_row: dict[str, Any], harness_meta: dict[str, Any]) -> Any:
    raw_index = run_row.get("index")
    if benchmark == "alfworld":
        start = harness_meta.get("start")
        if isinstance(start, int) and isinstance(raw_index, int):
            return start + raw_index
    return raw_index


def is_patched_or_harnessed(messages: list[dict[str, Any]]) -> bool:
    """Best-effort guard against accidentally using patched rerun trajectories."""
    if not messages:
        return True
    system_text = "\n".join(
        str(msg.get("content") or "")
        for msg in messages
        if msg.get("role") == "system"
    ).lower()
    forbidden = [
        "additional reusable tool-use hint from the harness",
        "harness overlay",
        "harness hint",
        "harness intervention",
    ]
    return any(token in system_text for token in forbidden)


def find_single_runs_path(batch_dir: Path) -> Path | None:
    candidates = sorted(batch_dir.glob("rollout/*/*/runs.jsonl"))
    if not candidates:
        return None
    if len(candidates) > 1:
        # Prefer explicit no-harness task directories when both source and
        # patched reruns coexist.
        no_harness = [p for p in candidates if "noharness" in str(p).lower()]
        if no_harness:
            return no_harness[0]
    return candidates[0]


def resolve_source_batch(meta: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    benchmark = str(meta.get("benchmark") or meta.get("mixed_source_benchmark") or "").lower()
    details: dict[str, Any] = {"benchmark": benchmark}

    if benchmark in {"webshop", "alfworld"}:
        source_rollout_dir = meta.get("source_rollout_dir")
        if not source_rollout_dir and benchmark == "webshop":
            patch_path = meta.get("patch_path")
            if patch_path:
                patch_parent = as_repo_path(patch_path).parent
                patch_meta_path = patch_parent / "metadata.json"
                if patch_meta_path.exists():
                    patch_meta = read_json(patch_meta_path)
                    source_rollout_dir = patch_meta.get("source_rollout_dir")
                    details["patch_metadata_path"] = str(patch_meta_path)
        if not source_rollout_dir:
            details["error"] = "missing source_rollout_dir"
            return None, details
        batch_dir = as_repo_path(source_rollout_dir)
        details["source_batch_dir"] = str(batch_dir)
        return batch_dir, details

    if benchmark == "dbbench":
        artifact_root = meta.get("artifact_root")
        batch_tag = meta.get("batch_tag")
        if not artifact_root or not batch_tag:
            details["error"] = "missing artifact_root or batch_tag"
            return None, details
        run_config_path = as_repo_path(artifact_root) / "run_config.json"
        if not run_config_path.exists():
            details["error"] = f"missing run_config: {run_config_path}"
            return None, details
        run_config = read_json(run_config_path)
        source_run_root = run_config.get("source_run_root")
        if not source_run_root:
            details["error"] = f"missing source_run_root in {run_config_path}"
            return None, details
        batch_dir = as_repo_path(source_run_root) / "dbbench" / str(batch_tag)
        details["source_run_root"] = str(as_repo_path(source_run_root))
        details["source_batch_dir"] = str(batch_dir)
        details["run_config_path"] = str(run_config_path)
        return batch_dir, details

    details["error"] = f"unsupported benchmark: {benchmark}"
    return None, details


def make_demo(
    *,
    run_row: dict[str, Any],
    harness_meta: dict[str, Any],
    harness_index: int,
    runs_path: Path,
    threshold: float,
) -> dict[str, Any] | None:
    reward = reward_of(run_row)
    messages = [normalize_message(msg) for msg in openai_messages_of(run_row)]
    if not messages or is_patched_or_harnessed(messages):
        return None
    benchmark = str(harness_meta.get("benchmark") or harness_meta.get("mixed_source_benchmark") or "").lower()
    task_index = run_row.get("index")
    task_id = canonical_task_id(benchmark, run_row, harness_meta)
    metadata = {
        "benchmark": benchmark,
        "task_index": task_index,
        "canonical_task_id": task_id,
        "reward": reward,
        "success": reward >= threshold,
        "success_threshold": threshold,
        "status": (run_row.get("output") or {}).get("status"),
        "source_runs_path": str(runs_path),
        "source_batch_tag": harness_meta.get("batch_tag") or harness_meta.get("source_batch_tag") or harness_meta.get("batch_dir"),
        "source_start": harness_meta.get("start"),
        "source_end": harness_meta.get("end"),
        "harness_sft_index": harness_index,
        "harness_sft_mixed_index": harness_meta.get("mixed_index"),
        "harness_sft_baseline_pass": harness_meta.get("baseline_pass"),
        "harness_sft_patched_pass": first_present(harness_meta, "patched_pass", "teacher_patched_pass"),
        "harness_sft_delta_pass": first_present(harness_meta, "delta_pass", "teacher_delta_pass"),
    }
    return {"messages": messages, "metadata": metadata}


def task_dedup_key(demo: dict[str, Any]) -> tuple[str, Any]:
    metadata = demo["metadata"]
    return str(metadata.get("benchmark")), metadata.get("canonical_task_id")


def dedupe_by_task(demos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep one trajectory per benchmark/task.

    For agent SFT, the most useful single trajectory is usually the highest
    reward one.  If multiple candidates tie, prefer the shorter successful
    transcript to avoid overweighting long failed loops.
    """
    selected: dict[tuple[str, Any], dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()

    def score(demo: dict[str, Any]) -> tuple[float, int, int]:
        metadata = demo["metadata"]
        reward = float(metadata.get("reward") or 0.0)
        success_bonus = 1 if metadata.get("success") else 0
        # Higher is better.  Use negative length so shorter tied trajectories win.
        return reward, success_bonus, -len(demo.get("messages") or [])

    for demo in demos:
        key = task_dedup_key(demo)
        if key[1] is None:
            key = (key[0], (demo["metadata"].get("source_runs_path"), demo["metadata"].get("task_index")))
        previous = selected.get(key)
        if previous is None:
            selected[key] = demo
            continue
        duplicate_counts[key[0]] += 1
        if score(demo) > score(previous):
            selected[key] = demo

    rows = list(selected.values())
    rows.sort(key=lambda item: (str(item["metadata"].get("benchmark")), str(item["metadata"].get("canonical_task_id"))))
    stats = {
        "input_total": len(demos),
        "output_total": len(rows),
        "extra_duplicates_removed_total": len(demos) - len(rows),
        "extra_duplicates_removed_by_benchmark": dict(duplicate_counts),
        "output_counts": dict(Counter(d["metadata"]["benchmark"] for d in rows)),
    }
    return rows, stats


def build_candidates(
    harness_rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_unique: list[dict[str, Any]] = []
    all_with_repeats: list[dict[str, Any]] = []
    success_unique: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()

    for i, row in enumerate(harness_rows):
        meta = parse_metadata(row)
        benchmark = str(meta.get("benchmark") or meta.get("mixed_source_benchmark") or "").lower()
        batch_dir, details = resolve_source_batch(meta)
        entry = {
            "harness_sft_index": i,
            "benchmark": benchmark,
            "batch_tag": meta.get("batch_tag") or meta.get("source_batch_tag") or meta.get("batch_dir"),
            **details,
        }
        if batch_dir is None or not batch_dir.exists():
            entry["status"] = "missing_source_batch"
            manifest.append(entry)
            continue
        runs_path = find_single_runs_path(batch_dir)
        if runs_path is None or not runs_path.exists():
            entry["status"] = "missing_runs_jsonl"
            manifest.append(entry)
            continue

        rows = read_jsonl(runs_path)
        usable = 0
        extracted = 0
        skipped_duplicate = 0
        for run_row in rows:
            demo = make_demo(
                run_row=run_row,
                harness_meta=meta,
                harness_index=i,
                runs_path=runs_path,
                threshold=threshold,
            )
            if demo is None:
                continue
            usable += 1
            all_with_repeats.append(demo)
            key = (
                str(demo["metadata"]["benchmark"]),
                demo["metadata"]["task_index"],
                str(runs_path),
            )
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            all_unique.append(demo)
            if demo["metadata"]["success"]:
                success_unique.append(demo)
                extracted += 1

        entry.update(
            {
                "status": "ok",
                "runs_path": str(runs_path),
                "runs_rows": len(rows),
                "usable_trajectories_extracted": usable,
                "unique_trajectories_duplicate_skipped": skipped_duplicate,
                "success_demos_extracted": extracted,
            }
        )
        manifest.append(entry)

    return all_unique, all_with_repeats, success_unique, manifest


def sample_same_counts(
    candidates: list[dict[str, Any]],
    target_counts: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for demo in candidates:
        by_benchmark[str(demo["metadata"]["benchmark"])].append(demo)

    selected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for benchmark, target_n in sorted(target_counts.items()):
        pool = by_benchmark.get(benchmark, [])
        pool = pool[:]
        rng.shuffle(pool)
        take_n = min(target_n, len(pool))
        if take_n < target_n:
            warnings.append(f"{benchmark}: requested {target_n}, available {len(pool)}")
        selected.extend(pool[:take_n])
    rng.shuffle(selected)
    stats = {
        "target_counts": target_counts,
        "available_success_counts": {k: len(v) for k, v in sorted(by_benchmark.items())},
        "selected_counts": dict(Counter(d["metadata"]["benchmark"] for d in selected)),
        "selected_total": len(selected),
        "warnings": warnings,
    }
    return selected, stats


def infer_target_counts(harness_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in harness_rows:
        meta = parse_metadata(row)
        benchmark = str(meta.get("benchmark") or meta.get("mixed_source_benchmark") or "").lower()
        counts[benchmark] += 1
    return dict(counts)


def write_dataset_info(output_dir: Path, entries: dict[str, str]) -> None:
    # LLaMA-Factory installations differ in how much native OpenAI tool-call
    # supervision they support.  The raw JSON keeps tool_calls/tool roles; if a
    # trainer rejects them, render this JSON with the model chat template first.
    payload = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
                "tool_tag": "tool",
            },
        }
        for dataset_name, file_name in entries.items()
    }
    dump_json(output_dir / "dataset_info.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-sft-json", type=Path, default=DEFAULT_HARNESS_SFT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument(
        "--same-counts",
        default="auto",
        help="Comma-separated counts like webshop=381,alfworld=248,dbbench=248, or auto.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harness_path = args.harness_sft_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    harness_rows = read_json(harness_path)
    if not isinstance(harness_rows, list):
        raise SystemExit(f"harness SFT JSON must be a list: {harness_path}")

    if args.same_counts == "auto":
        target_counts = infer_target_counts(harness_rows)
    else:
        target_counts = {}
        for item in args.same_counts.split(","):
            name, value = item.split("=", 1)
            target_counts[name.strip().lower()] = int(value)

    all_unique, all_with_repeats, success_candidates, manifest = build_candidates(
        harness_rows,
        args.success_threshold,
    )
    selected, sample_stats = sample_same_counts(success_candidates, target_counts, args.seed)

    dataset_name = output_dir.name
    selected_json_name = f"{dataset_name}.json"
    all_unique_jsonl_name = f"{dataset_name}.all_source_trajectories_openai.jsonl"
    all_with_repeats_jsonl_name = f"{dataset_name}.all_source_trajectories_with_repeats_openai.jsonl"
    success_jsonl_name = f"{dataset_name}.success_source_trajectories_openai.jsonl"
    all_taskdedup_jsonl_name = f"{dataset_name}.all_source_trajectories_taskdedup_openai.jsonl"
    success_taskdedup_jsonl_name = f"{dataset_name}.success_source_trajectories_taskdedup_openai.jsonl"
    manifest_name = f"{dataset_name}.source_manifest.jsonl"

    all_taskdedup, all_taskdedup_stats = dedupe_by_task(all_unique)
    success_taskdedup, success_taskdedup_stats = dedupe_by_task(success_candidates)

    dump_json(output_dir / selected_json_name, selected)
    dump_jsonl(output_dir / all_unique_jsonl_name, all_unique)
    dump_jsonl(output_dir / all_with_repeats_jsonl_name, all_with_repeats)
    dump_jsonl(output_dir / success_jsonl_name, success_candidates)
    dump_jsonl(output_dir / all_taskdedup_jsonl_name, all_taskdedup)
    dump_jsonl(output_dir / success_taskdedup_jsonl_name, success_taskdedup)
    dump_jsonl(output_dir / manifest_name, manifest)
    dataset_info_entries = {
        f"{dataset_name}_full_split_all": all_unique_jsonl_name,
        f"{dataset_name}_full_split_all_with_repeats": all_with_repeats_jsonl_name,
        f"{dataset_name}_full_split_success": success_jsonl_name,
        f"{dataset_name}_full_split_all_taskdedup": all_taskdedup_jsonl_name,
        f"{dataset_name}_full_split_success_taskdedup": success_taskdedup_jsonl_name,
        f"{dataset_name}_patch_count_success": selected_json_name,
    }
    write_dataset_info(output_dir, dataset_info_entries)

    manifest_status = Counter(item.get("status") for item in manifest)
    all_counts = Counter(d["metadata"]["benchmark"] for d in all_unique)
    all_repeat_counts = Counter(d["metadata"]["benchmark"] for d in all_with_repeats)
    success_counts = Counter(d["metadata"]["benchmark"] for d in success_candidates)
    stats = {
        "harness_sft_json": str(harness_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "success_threshold": args.success_threshold,
        "harness_rows": len(harness_rows),
        "harness_counts": infer_target_counts(harness_rows),
        "source_manifest_status": dict(manifest_status),
        "all_source_trajectories_unique_total": len(all_unique),
        "all_source_trajectories_unique_counts": dict(all_counts),
        "all_source_trajectories_with_repeats_total": len(all_with_repeats),
        "all_source_trajectories_with_repeats_counts": dict(all_repeat_counts),
        "success_source_trajectories_total": len(success_candidates),
        "success_source_trajectories_counts": dict(success_counts),
        "all_source_trajectories_taskdedup": all_taskdedup_stats,
        "success_source_trajectories_taskdedup": success_taskdedup_stats,
        **sample_stats,
        "files": {
            "patch_count_matched_success_dataset": selected_json_name,
            "all_source_trajectories_jsonl": all_unique_jsonl_name,
            "all_source_trajectories_with_repeats_jsonl": all_with_repeats_jsonl_name,
            "success_source_trajectories_jsonl": success_jsonl_name,
            "all_source_trajectories_taskdedup_jsonl": all_taskdedup_jsonl_name,
            "success_source_trajectories_taskdedup_jsonl": success_taskdedup_jsonl_name,
            "source_manifest_jsonl": manifest_name,
            "dataset_info": "dataset_info.json",
        },
        "dataset_info_entries": dataset_info_entries,
        "notes": [
            "Trajectories are extracted only from no-harness source runs resolved from harness SFT metadata.",
            "Rows with system text suggesting harness hints/interventions are skipped.",
            "The all_source_trajectories file is the full source-split trajectory set, de-duplicated by benchmark/task/runs_path.",
            "The all_source_trajectories_with_repeats file preserves every trajectory reference from every harness SFT source batch.",
            "The taskdedup files keep one best trajectory per benchmark/canonical_task_id; ALFWorld canonical ids use source_start + local task index.",
            "The selected JSON is only a patch-count-matched successful subset for ablation.",
            "All JSON outputs preserve OpenAI tool_calls and role=tool messages for tool-use SFT.",
        ],
    }
    dump_json(output_dir / f"{dataset_name}.stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
