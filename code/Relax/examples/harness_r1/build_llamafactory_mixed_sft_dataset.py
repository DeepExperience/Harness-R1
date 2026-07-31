#!/usr/bin/env python3
"""Build a shuffled LLaMA-Factory ShareGPT SFT dataset from multiple sources."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset_info(dataset_dir: Path, dataset_name: str, file_name: str) -> None:
    write_json(
        dataset_dir / "dataset_info.json",
        {
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
                },
            }
        },
    )


def infer_benchmark(path: Path, row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("benchmark"):
        return str(metadata["benchmark"])
    name = path.name.lower()
    for key in ("alfworld", "webshop"):
        if key in name:
            return key
    text = json.dumps(row.get("messages", []), ensure_ascii=False).lower()
    if "alfworld" in text:
        return "alfworld"
    if "webshop" in text:
        return "webshop"
    return "unknown"


def load_source(path: Path, *, metadata_mode: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a list, got {type(payload).__name__}")
    rows: list[dict[str, Any]] = []
    for line_no, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_no} must be an object")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"{path}:{line_no} missing ShareGPT messages")
        metadata = dict(row.get("metadata") or {})
        benchmark = infer_benchmark(path, row)
        metadata.update(
            {
                "mixed_source_benchmark": benchmark,
                "mixed_source_file": str(path),
                "mixed_source_index": line_no - 1,
            }
        )
        out: dict[str, Any] = {"messages": messages}
        if metadata_mode == "object":
            out["metadata"] = metadata
        elif metadata_mode == "string":
            out["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        elif metadata_mode != "drop":
            raise ValueError(f"unknown metadata_mode: {metadata_mode}")
        rows.append(out)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument(
        "--balanced-cap-per-source",
        type=int,
        default=0,
        help="If positive, shuffle each source with --seed and keep at most this many rows per source before mixing.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=["string", "object", "drop"],
        default="string",
        help="Use string/drop for LLaMA-Factory compatibility across heterogeneous source metadata.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    sources: list[dict[str, Any]] = []
    for source in args.source:
        source_rows = load_source(source, metadata_mode=args.metadata_mode)
        raw_source_rows = len(source_rows)
        if args.balanced_cap_per_source > 0 and len(source_rows) > args.balanced_cap_per_source:
            source_rng = random.Random(f"{args.seed}:{source}")
            source_rng.shuffle(source_rows)
            source_rows = source_rows[: args.balanced_cap_per_source]
        rows.extend(source_rows)
        counts = Counter()
        for row in source_rows:
            if args.metadata_mode == "string":
                metadata = json.loads(str(row.get("metadata") or "{}"))
            else:
                metadata = row.get("metadata") or {}
            counts[str(metadata.get("mixed_source_benchmark"))] += 1
        source_counts.update(counts)
        sources.append(
            {
                "path": str(source),
                "counts": dict(sorted(counts.items())),
                "rows": raw_source_rows,
                "selected": len(source_rows),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for idx, row in enumerate(rows):
        if args.metadata_mode == "object":
            row.setdefault("metadata", {})["mixed_index"] = idx
            row["metadata"]["mixed_seed"] = args.seed
            row["metadata"]["mixed_dataset"] = args.dataset_name
        elif args.metadata_mode == "string":
            metadata = json.loads(str(row.get("metadata") or "{}"))
            metadata["mixed_index"] = idx
            metadata["mixed_seed"] = args.seed
            metadata["mixed_dataset"] = args.dataset_name
            row["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{args.dataset_name}.json"
    write_json(args.output_dir / file_name, rows)
    write_dataset_info(args.output_dir, args.dataset_name, file_name)
    stats = {
        "dataset_name": args.dataset_name,
        "selected": len(rows),
        "seed": args.seed,
        "metadata_mode": args.metadata_mode,
        "balanced_cap_per_source": args.balanced_cap_per_source,
        "source_counts": dict(sorted(source_counts.items())),
        "sources": sources,
    }
    write_json(args.output_dir / f"{args.dataset_name}.stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
