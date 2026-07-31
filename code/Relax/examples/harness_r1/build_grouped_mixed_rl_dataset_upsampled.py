# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build source-homogeneous mixed RL data with benchmark-level resampling."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_OFFSETS = {
    "alfworld": 101,
    "webshop": 211,
    "dbbench": 401,
}


def _infer_benchmark(path: Path, row: dict[str, Any]) -> str:
    name = path.name.lower()
    if "alfworld" in name:
        return "alfworld"
    if "webshop" in name:
        return "webshop"
    if "dbbench" in name:
        return "dbbench"
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    raw = str(metadata.get("benchmark") or "").strip().lower()
    if raw in {"alfworld", "webshop"}:
        return raw
    if raw == "dbbench":
        return "dbbench"
    text = json.dumps(row.get("prompt", ""), ensure_ascii=False).lower()
    if "alfworld" in text:
        return "alfworld"
    if "webshop" in text:
        return "webshop"
    if "dbbench" in text:
        return "dbbench"
    raise ValueError(f"cannot infer benchmark for {path}")


def _read_jsonl(path: Path, expected_benchmark: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark = _infer_benchmark(path, row)
            if benchmark != expected_benchmark:
                raise ValueError(
                    f"{path}:{line_no} inferred benchmark {benchmark!r}, "
                    f"expected {expected_benchmark!r}"
                )
            metadata = row.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_no} metadata must be an object")
            metadata["benchmark"] = benchmark
            metadata["mixed_source_file"] = str(path)
            metadata["mixed_source_line"] = line_no
            rows.append(row)
    return rows


def _copy_row(row: dict[str, Any], *, benchmark: str, upsampled: bool, source_index: int | None = None) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    metadata = copied.setdefault("metadata", {})
    metadata["benchmark"] = benchmark
    if upsampled:
        metadata["mixed_upsampled_duplicate"] = True
        metadata["mixed_upsampled_for"] = benchmark
        if source_index is not None:
            metadata["mixed_upsampled_source_index"] = source_index
    return copied


def _make_base_groups(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    group_size: int,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    shuffled = [_copy_row(row, benchmark=benchmark, upsampled=False) for row in rows]
    rng.shuffle(shuffled)
    remainder = len(shuffled) % group_size
    if remainder:
        need = group_size - remainder
        for _ in range(need):
            duplicate = _copy_row(rng.choice(rows), benchmark=benchmark, upsampled=False)
            metadata = duplicate.setdefault("metadata", {})
            metadata["mixed_padded_duplicate"] = True
            metadata["mixed_padded_duplicate_for"] = benchmark
            shuffled.append(duplicate)
    return [shuffled[start : start + group_size] for start in range(0, len(shuffled), group_size)]


def _make_resampled_groups(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    group_size: int,
    target_groups: int,
    rng: random.Random,
) -> tuple[list[list[dict[str, Any]]], int, int, int]:
    groups = _make_base_groups(rows, benchmark=benchmark, group_size=group_size, rng=rng)
    base_groups = len(groups)
    if target_groups < base_groups:
        rng.shuffle(groups)
        downsampled_groups = base_groups - target_groups
        return groups[:target_groups], base_groups, 0, downsampled_groups

    upsampled_rows = 0
    for group_idx in range(target_groups - base_groups):
        group: list[dict[str, Any]] = []
        for _ in range(group_size):
            source_index = rng.randrange(len(rows))
            group.append(
                _copy_row(
                    rows[source_index],
                    benchmark=benchmark,
                    upsampled=True,
                    source_index=source_index,
                )
            )
            upsampled_rows += 1
        for row in group:
            row.setdefault("metadata", {})["mixed_upsample_extra_group_index"] = group_idx
        groups.append(group)
    return groups, base_groups, upsampled_rows, 0


def _parse_group_counts(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        benchmark = key.strip()
        out[benchmark] = int(value)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_chars(row: dict[str, Any]) -> int:
    return len(json.dumps(row.get("prompt", ""), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld", required=True, type=Path)
    parser.add_argument("--webshop", required=True, type=Path)
    parser.add_argument("--dbbench", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=20260623, type=int)
    parser.add_argument("--group-size", default=4, type=int)
    parser.add_argument(
        "--group-counts",
        required=True,
        help="Comma-separated target groups, e.g. webshop=130,alfworld=130,dbbench=130",
    )
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    target_group_counts = _parse_group_counts(args.group_counts)

    source_paths = {
        "alfworld": args.alfworld,
        "webshop": args.webshop,
    }
    if args.dbbench is not None:
        source_paths["dbbench"] = args.dbbench
    missing_group_counts = set(source_paths) - set(target_group_counts)
    if missing_group_counts:
        raise ValueError(f"--group-counts missing: {sorted(missing_group_counts)}")
    extra_group_counts = set(target_group_counts) - set(source_paths)
    if extra_group_counts:
        raise ValueError(f"--group-counts has no source for: {sorted(extra_group_counts)}")
    source_rows = {
        benchmark: _read_jsonl(path, benchmark)
        for benchmark, path in source_paths.items()
    }

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    base_group_counts: dict[str, int] = {}
    upsampled_row_counts: dict[str, int] = {}
    downsampled_group_counts: dict[str, int] = {}
    for benchmark, rows in source_rows.items():
        source_rng = random.Random(args.seed + SOURCE_OFFSETS[benchmark])
        benchmark_groups, base_group_count, upsampled_rows, downsampled_groups = _make_resampled_groups(
            rows,
            benchmark=benchmark,
            group_size=args.group_size,
            target_groups=target_group_counts[benchmark],
            rng=source_rng,
        )
        base_group_counts[benchmark] = base_group_count
        upsampled_row_counts[benchmark] = upsampled_rows
        downsampled_group_counts[benchmark] = downsampled_groups
        groups.extend((benchmark, group) for group in benchmark_groups)

    group_rng = random.Random(args.seed)
    group_rng.shuffle(groups)

    output_rows: list[dict[str, Any]] = []
    for group_index, (benchmark, group) in enumerate(groups):
        for group_pos, row in enumerate(group):
            metadata = row.setdefault("metadata", {})
            metadata["mixed_index"] = len(output_rows)
            metadata["mixed_seed"] = args.seed
            metadata["mixed_dataset"] = args.output.name
            metadata["mixed_builder"] = "grouped_by_benchmark_upsampled"
            metadata["mixed_group_index"] = group_index
            metadata["mixed_group_pos"] = group_pos
            metadata["mixed_group_size"] = args.group_size
            metadata["mixed_group_benchmark"] = benchmark
            output_rows.append(row)

    _write_jsonl(args.output, output_rows)

    group_violations = 0
    group_labels: list[str] = []
    for start in range(0, len(output_rows), args.group_size):
        group = output_rows[start : start + args.group_size]
        labels = {
            str((row.get("metadata") or {}).get("benchmark"))
            for row in group
        }
        label = next(iter(labels)) if len(labels) == 1 else "MIXED"
        group_labels.append(label)
        if len(labels) != 1 or len(group) != args.group_size:
            group_violations += 1

    counts = Counter(str((row.get("metadata") or {}).get("benchmark")) for row in output_rows)
    group_counts = Counter(group_labels)
    prompt_lengths = [_prompt_chars(row) for row in output_rows]
    stats = {
        "output": str(args.output),
        "seed": args.seed,
        "group_size": args.group_size,
        "num_rows": len(output_rows),
        "num_groups": len(groups),
        "recommended_num_rollout": len(groups),
        "target_group_counts": dict(sorted(target_group_counts.items())),
        "base_group_counts": dict(sorted(base_group_counts.items())),
        "upsampled_row_counts": dict(sorted(upsampled_row_counts.items())),
        "downsampled_group_counts": dict(sorted(downsampled_group_counts.items())),
        "benchmark_counts": dict(sorted(counts.items())),
        "benchmark_group_counts": dict(sorted(group_counts.items())),
        "homogeneous_group_violations": group_violations,
        "prompt_chars_min": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_chars_avg": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0,
        "prompt_chars_max": max(prompt_lengths) if prompt_lengths else 0,
        "first_24_group_labels": group_labels[:24],
        "sources": {benchmark: str(path) for benchmark, path in source_paths.items()},
    }
    args.output.with_suffix(args.output.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
