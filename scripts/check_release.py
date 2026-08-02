#!/usr/bin/env python3
"""Run source-only release checks without benchmark assets or model endpoints."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REQUIRED = (
    "README.md",
    "code/Relax/relax/entrypoints/train.py",
    "code/Relax/examples/harness_r1/reward_mixed_codepatch.py",
    "code/life-harness/AgentBench/scripts/harness_r1_patch.py",
    "code/life-harness/AgentBench/scripts/harness_r1_batch_debug.py",
    "code/life-harness/AgentBench/src/server/harness/code_runner.py",
    "configs/rl/mixed_codepatch.yaml",
    "scripts/train_engineer_rl.sh",
    "scripts/eval_webshop.sh",
    "demo/run_demo.py",
    "demo/artifacts/patch.json",
)
FORBIDDEN = {
    "private API key": re.compile(
        r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_]{20,}|MAAS[A-Za-z0-9_-]{16,})\b"
    ),
    "private key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "credential in URL": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "research-cluster address": re.compile(r"\b10\.217(?:\.\d{1,3}){2}\b"),
    "research workspace path": re.compile(r"/mnt/" r"tidalfs[^/\s]*/"),
}


def files() -> list[Path]:
    ignored_parts = {".git", "__pycache__", ".pytest_cache"}
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in ignored_parts for part in path.parts)
    )


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def check_required() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit("missing required release files:\n" + "\n".join(missing))
    print("required files: ok")


def check_symlinks() -> None:
    errors: list[str] = []
    root_resolved = ROOT.resolve()
    for path in ROOT.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError:
            errors.append(f"{path.relative_to(ROOT)}: broken symlink")
            continue
        if root_resolved not in target.parents and target != root_resolved:
            errors.append(f"{path.relative_to(ROOT)}: points outside release")
    if errors:
        raise SystemExit("symlink check failed:\n" + "\n".join(errors))
    print("symlinks: ok")


def scan_sensitive_text() -> None:
    errors: list[str] = []
    for path in files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    if errors:
        raise SystemExit("sensitive-text scan failed:\n" + "\n".join(errors))
    print("sensitive text: ok")


def main() -> int:
    check_required()
    check_symlinks()
    scan_sensitive_text()

    shell_scripts = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "scripts").glob("*.sh")
    ]
    for script in shell_scripts:
        run(["bash", "-n", script])

    with tempfile.TemporaryDirectory(prefix="harness-r1-pycache-") as cache:
        env = dict(os.environ)
        env["PYTHONPYCACHEPREFIX"] = cache
        run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            env=env,
        )
        run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "code/Relax/examples/harness_r1",
                "code/life-harness/AgentBench/scripts",
                "code/life-harness/AgentBench/src/server/harness",
                "demo",
                "scripts",
                "tests",
            ],
            env=env,
        )
    print("release checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
