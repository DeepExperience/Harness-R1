#!/usr/bin/env python3
"""Propose and validate Harness-R1 typed harness patches from batch overviews."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from harness_r1_patch import (
    PatchValidationError,
    extract_json_object,
    normalize_patch,
    schema_prompt,
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def default_agentbench_dir(repo_root: Path) -> Path:
    external = repo_root / "external/harness_evolution/life-harness/AgentBench"
    if external.exists():
        return external
    return Path(__file__).resolve().parents[1]


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    return env


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def message_text(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    if content.strip():
        return content
    return str(message.get("reasoning_content") or "")


def collect_overviews(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.overview or []:
        p = Path(raw).resolve()
        if p.exists():
            paths.append(p)
    if args.run_root:
        run_root = args.run_root.resolve()
        batch_filter = None
        if args.batch_ids:
            batch_filter = {int(x) for part in args.batch_ids for x in part.split(",") if x.strip()}
        for p in sorted((run_root / args.bench).glob("batch_*/debug/overview.md")):
            if batch_filter is not None:
                try:
                    batch_id = int(p.parents[1].name.split("_", 1)[1])
                except Exception:
                    continue
                if batch_id not in batch_filter:
                    continue
            paths.append(p)
    seen = set()
    unique: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise SystemExit("no overview files found; pass --run-root or --overview")
    return unique


def read_overview_bundle(paths: list[Path], max_chars: int) -> str:
    parts = []
    remaining = max_chars
    for p in paths:
        if remaining <= 0:
            break
        text = p.read_text(encoding="utf-8", errors="replace")
        excerpt = text[:remaining]
        remaining -= len(excerpt)
        parts.append(f"## Source: {p}\n\n{excerpt}")
    return "\n\n---\n\n".join(parts)


def build_prompt(args: argparse.Namespace, overviews: str) -> list[dict[str, str]]:
    system = (
        "You are a Harness-R1 harness engineer. Convert debugger overviews into a "
        "small, general typed harness patch. Prefer simple, auditable changes. "
        "Do not leak task answers."
    )
    user = (
        schema_prompt(args.bench)
        + "\n\nCurrent rollout setting:\n"
        + f"- benchmark: {args.bench}\n"
        + "- baseline: no-harness failure/debug overviews\n"
        + "- objective: propose a single-turn patch for future train-batch validation\n\n"
        + f"{args.input_label}:\n\n"
        + overviews
    )
    if args.engineer_prompt_prefix:
        user = args.engineer_prompt_prefix + user
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat(
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    try:
        return message_text(data["choices"][0]["message"])
    except Exception as exc:
        raise RuntimeError(f"unexpected chat response: {data}") from exc


def validate_manual_patch(args: argparse.Namespace, out_dir: Path) -> int:
    raw = json.loads(args.manual_patch.read_text(encoding="utf-8"))
    patch = normalize_patch(raw, bench=args.bench)
    dump_json(out_dir / "patch.json", patch)
    dump_text(out_dir / "status.txt", "manual patch validated\n")
    print(f"[harness-r1-edit] validated patch -> {out_dir / 'patch.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    agentbench_dir = default_agentbench_dir(repo_root)
    parser = argparse.ArgumentParser(description="Harness-R1 typed harness patch proposer.")
    parser.add_argument("--bench", choices=["webshop", "alfworld"], required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--overview", action="append", default=[])
    parser.add_argument("--batch-ids", nargs="*", default=[])
    parser.add_argument("--manual-patch", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-overview-chars", type=int, default=40000)
    parser.add_argument("--engineer-base-url", default="http://127.0.0.1:30040/v1")
    parser.add_argument("--engineer-model", default="gpt-5.4")
    parser.add_argument("--engineer-api-key", default="any")
    parser.add_argument("--engineer-timeout", type=int, default=600)
    parser.add_argument("--engineer-max-tokens", type=int, default=4096)
    parser.add_argument("--engineer-prompt-prefix", default="")
    parser.add_argument("--input-label", default="Failure-analysis inputs")
    parser.add_argument("--repair-attempts", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agentbench-dir", type=Path, default=agentbench_dir)
    args = parser.parse_args()
    args.repo_root = repo_root
    args.agentbench_dir = args.agentbench_dir.resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.agentbench_dir / "outputs/harness_r1_edits" / f"{args.bench}_{stamp}"
    args.output_dir = args.output_dir.resolve()
    if args.max_overview_chars <= 0:
        raise SystemExit("--max-overview-chars must be positive")
    if args.repair_attempts < 0:
        raise SystemExit("--repair-attempts must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            run_config[key] = str(value)
        elif isinstance(value, list):
            run_config[key] = [str(x) if isinstance(x, Path) else x for x in value]
        else:
            run_config[key] = value
    dump_json(out_dir / "run_config.json", run_config)
    if args.manual_patch:
        return validate_manual_patch(args, out_dir)

    overview_paths = collect_overviews(args)
    overviews = read_overview_bundle(overview_paths, args.max_overview_chars)
    messages = build_prompt(args, overviews)
    dump_json(out_dir / "overview_sources.json", [str(p) for p in overview_paths])
    dump_text(out_dir / "prompt.md", messages[-1]["content"])
    if args.dry_run:
        dump_text(out_dir / "status.txt", "dry run: prompt generated, model not called\n")
        print(f"[harness-r1-edit] dry run prompt -> {out_dir / 'prompt.md'}")
        return 0

    attempt_messages = list(messages)
    last_exc: Exception | None = None
    for attempt in range(args.repair_attempts + 1):
        try:
            raw_response = call_chat(
                base_url=args.engineer_base_url,
                model=args.engineer_model,
                api_key=args.engineer_api_key,
                messages=attempt_messages,
                timeout=args.engineer_timeout,
                max_tokens=args.engineer_max_tokens,
            )
            dump_text(out_dir / f"raw_response_attempt_{attempt}.txt", raw_response)
            dump_text(out_dir / "raw_response.txt", raw_response)
            patch_raw = extract_json_object(raw_response)
            patch = normalize_patch(patch_raw, bench=args.bench)
            dump_json(out_dir / "patch.json", patch)
            dump_text(out_dir / "status.txt", f"ok attempt={attempt}\n")
            print(f"[harness-r1-edit] patch -> {out_dir / 'patch.json'}")
            return 0
        except (json.JSONDecodeError, PatchValidationError) as exc:
            last_exc = exc
            if attempt >= args.repair_attempts:
                break
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": raw_response if "raw_response" in locals() else ""},
                {
                    "role": "user",
                    "content": (
                        f"The previous JSON patch failed validation with error: {exc!r}\n"
                        "Return one corrected JSON object only. Use only the allowed schema, "
                        "condition variables, and predicates from the original instructions. "
                        "Do not include markdown or commentary."
                    ),
                },
            ]
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
        ) as exc:
            last_exc = exc
            break

    dump_text(out_dir / "status.txt", f"failed: {last_exc!r}\n")
    print(f"[harness-r1-edit] failed: {last_exc!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
