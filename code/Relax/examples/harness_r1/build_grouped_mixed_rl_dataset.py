# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build a mixed RL prompt dataset with source-homogeneous rollout batches.

Relax consumes prompt rows contiguously when ``--rollout-shuffle`` is disabled.
This builder lays rows out in fixed-size groups where every row in a group has
the same benchmark. With ``rollout_batch_size == group_size`` and
``ROLLOUT_SHUFFLE=0``, each rollout step/chunk sees prompts from a single
benchmark while the group order is still shuffled across benchmarks.
"""

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


def _cap_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return rows
    rng = random.Random(seed + SOURCE_OFFSETS[benchmark] * 1009)
    selected = list(rows)
    rng.shuffle(selected)
    return selected[:cap]


def _prompt_chars(row: dict[str, Any]) -> int:
    return len(json.dumps(row.get("prompt", ""), ensure_ascii=False))


def _make_groups(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    group_size: int,
    tail_policy: str,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)

    remainder = len(shuffled) % group_size
    if remainder:
        if tail_policy == "error":
            raise ValueError(f"{benchmark} has {len(shuffled)} rows, not divisible by group_size={group_size}")
        if tail_policy == "drop":
            shuffled = shuffled[: len(shuffled) - remainder]
        elif tail_policy == "pad":
            need = group_size - remainder
            if not shuffled:
                raise ValueError(f"cannot pad empty source {benchmark}")
            for _ in range(need):
                duplicate = copy.deepcopy(rng.choice(shuffled))
                metadata = duplicate.setdefault("metadata", {})
                metadata["mixed_padded_duplicate"] = True
                metadata["mixed_padded_duplicate_for"] = benchmark
                shuffled.append(duplicate)
        else:
            raise ValueError(f"unknown tail_policy: {tail_policy}")

    groups: list[list[dict[str, Any]]] = []
    for start in range(0, len(shuffled), group_size):
        group = shuffled[start : start + group_size]
        if len(group) != group_size:
            raise AssertionError("internal grouping error")
        groups.append(group)
    return groups


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld", required=True, type=Path)
    parser.add_argument("--webshop", required=True, type=Path)
    parser.add_argument("--dbbench", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=20260621, type=int)
    parser.add_argument("--group-size", default=4, type=int)
    parser.add_argument("--tail-policy", choices=["pad", "drop", "error"], default="pad")
    parser.add_argument(
        "--balanced-cap-per-source",
        type=int,
        default=0,
        help="If positive, deterministically sample at most this many rows from each source before grouping.",
    )
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")

    source_paths = {
        "alfworld": args.alfworld,
        "webshop": args.webshop,
    }
    if args.dbbench is not None:
        source_paths["dbbench"] = args.dbbench
    source_rows = {
        benchmark: _read_jsonl(path, benchmark)
        for benchmark, path in source_paths.items()
    }
    raw_source_counts = {benchmark: len(rows) for benchmark, rows in source_rows.items()}
    if args.balanced_cap_per_source > 0:
        source_rows = {
            benchmark: _cap_rows(
                rows,
                benchmark=benchmark,
                cap=args.balanced_cap_per_source,
                seed=args.seed,
            )
            for benchmark, rows in source_rows.items()
        }

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for benchmark, rows in source_rows.items():
        source_rng = random.Random(args.seed + SOURCE_OFFSETS[benchmark])
        for group in _make_groups(
            rows,
            benchmark=benchmark,
            group_size=args.group_size,
            tail_policy=args.tail_policy,
            rng=source_rng,
        ):
            groups.append((benchmark, group))

    group_rng = random.Random(args.seed)
    group_rng.shuffle(groups)

    output_rows: list[dict[str, Any]] = []
    for group_index, (benchmark, group) in enumerate(groups):
        for group_pos, row in enumerate(group):
            metadata = row.setdefault("metadata", {})
            metadata["mixed_index"] = len(output_rows)
            metadata["mixed_seed"] = args.seed
            metadata["mixed_dataset"] = args.output.name
            metadata["mixed_builder"] = "grouped_by_benchmark"
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
        "tail_policy": args.tail_policy,
        "num_rows": len(output_rows),
        "num_groups": len(groups),
        "recommended_num_rollout": len(groups),
        "benchmark_counts": dict(sorted(counts.items())),
        "benchmark_group_counts": dict(sorted(group_counts.items())),
        "homogeneous_group_violations": group_violations,
        "raw_source_counts": raw_source_counts,
        "balanced_cap_per_source": args.balanced_cap_per_source,
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
