"""Portable paths shared by the public evaluators and reward runners."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTBENCH_DIR = REPO_ROOT / "code/life-harness/AgentBench"


def normalize_pythonpath(env: dict[str, str]) -> None:
    """Keep caller-relative import paths valid after subprocess cwd changes."""
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(item or ".").expanduser().absolute())
            for item in env["PYTHONPATH"].split(os.pathsep)
        )


def repo_path(value: str | Path) -> Path:
    """Resolve repository-relative config paths without dereferencing venvs."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def default_python(agentbench_dir: str | Path = AGENTBENCH_DIR) -> Path:
    candidate = repo_path(agentbench_dir) / ".venv/bin/python"
    return candidate if candidate.is_file() else Path(sys.executable)


def default_worker_python(agentbench_dir: str | Path, bench: str, fallback: Path) -> Path:
    names = {
        "webshop": ("webshop-worker",),
        "alfworld": ("alfworld-worker-clean", "alfworld-worker"),
    }
    for name in names.get(bench, ()):
        candidate = repo_path(agentbench_dir) / ".venvs" / name / "bin/python"
        if candidate.is_file():
            return candidate
    return fallback
