#!/usr/bin/env python3
"""Validate AgentBench lite preset configs.

Goal: a fast smoke check that doesn't require building Docker images or running
any tasks. It only validates that:
- lite config files exist and are valid YAML
- referenced tasks exist in configs/tasks/task_assembly.yaml
- lite assignment references tasks that exist in the task assembly

Run:
  python scripts/validate_lite_configs.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Set

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse YAML: {path}: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping at top-level in {path}, got {type(data)}")
    return data


def main() -> int:
    start_lite = ROOT / "configs" / "start_task_lite.yaml"
    assign_lite = ROOT / "configs" / "assignments" / "lite.yaml"
    task_assembly = ROOT / "configs" / "tasks" / "task_assembly.yaml"

    for p in (start_lite, assign_lite, task_assembly):
        if not p.exists():
            raise RuntimeError(f"Missing required file: {p}")

    start_cfg = load_yaml(start_lite)
    assign_cfg = load_yaml(assign_lite)
    assembly_cfg = load_yaml(task_assembly)

    # Collect task names from imported task configs (based on file names).
    imports = assembly_cfg.get("import", [])
    if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
        raise RuntimeError("configs/tasks/task_assembly.yaml must have a list field: import")

    task_names: Set[str] = set()
    for rel in imports:
        # rel like "webshop.yaml" -> file stem "webshop".
        task_names.add(pathlib.Path(rel).stem)

    # start_task_lite.yaml: ensure start keys look like known tasks.
    start = start_cfg.get("start", {})
    if not isinstance(start, dict) or not start:
        raise RuntimeError("configs/start_task_lite.yaml must have non-empty mapping field: start")

    unknown_in_start = sorted([k for k in start.keys() if str(k).split("-")[0] not in task_names])
    if unknown_in_start:
        raise RuntimeError(
            "start_task_lite.yaml references tasks not present in task_assembly imports: "
            + ", ".join(map(str, unknown_in_start))
        )

    # assignments/lite.yaml: ensure tasks exist.
    assignments = assign_cfg.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise RuntimeError("configs/assignments/lite.yaml must have a non-empty list field: assignments")

    unknown_in_assign = []
    for a in assignments:
        if not isinstance(a, dict):
            raise RuntimeError("Each assignment must be a mapping")
        tasks = a.get("task", [])
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, list):
            raise RuntimeError("assignment.task must be a string or list")
        for t in tasks:
            base = str(t).split("-")[0]
            if base not in task_names:
                unknown_in_assign.append(t)

    if unknown_in_assign:
        raise RuntimeError(
            "assignments/lite.yaml references tasks not present in task_assembly imports: "
            + ", ".join(map(str, unknown_in_assign))
        )

    print("OK: lite configs look valid")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
