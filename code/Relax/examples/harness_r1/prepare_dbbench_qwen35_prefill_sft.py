#!/usr/bin/env python3
"""Prepare DBBench Qwen3.5 prefill-thinking SFT data from GPT patch rollouts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTBENCH_SCRIPTS = REPO_ROOT / "code/life-harness/AgentBench/scripts"
sys.path.insert(0, str(AGENTBENCH_SCRIPTS))

from harness_r1_patch import schema_prompt  # noqa: E402


DBBENCH_PREFILL_SYSTEM_PROMPT = (
    "You are a Harness-R1 harness engineer. Given a batch of failed DBBench "
    "SQL-agent rollout traces, analyze recurring failures, then produce one "
    "reusable DBBench code-hook harness patch as a single JSON object inside "
    "a <patch>...</patch> block."
)


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


def load_results(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for result_path in root.glob("batch_*/result.json"):
        rows[result_path.parent.name] = read_json(result_path)
    return rows


def extract_evidence(prompt_messages: list[dict[str, Any]]) -> str:
    user_prompt = "\n\n".join(
        str(message.get("content") or "") for message in prompt_messages if message.get("role") == "user"
    )
    marker = "Observed no-harness rollout evidence:"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[1].strip()
    trace_marker = "# DBBench"
    if trace_marker in user_prompt:
        return user_prompt[user_prompt.index(trace_marker) :].strip()
    return user_prompt.strip()


def dbbench_runtime_context_section() -> str:
    return """## DBBench Runtime Context

DBBench code hooks receive a read-only runtime context from the harness substrate.
Use it only to write general harness behavior, not task-specific SQL.

- DBBench is a MySQL-backed table task environment with two tools:
  execute_sql(query) and commit_final_answer(answers).
- Hooks read ctx["action"], ctx["state"], ctx["task"], ctx["dbbench"], and
  ctx["predicates"]. There is no ctx["world"] object in this runtime.
- ctx["action"] includes tool, query, answers, h2_action, h2_blocked_reason,
  commit_gate_action, and commit_gate_reason.
- ctx["state"] includes sql_count, last_sql, last_result, last_error_kind,
  last_error_text, last_result_was_error, error_streak, empty_streak,
  text_only_streak, loop_streak, mutation_attempted, candidate_answer,
  candidate_answer_shape, candidate_implausible, and remaining_rounds.
- ctx["task"] includes task_type, answer_shape, target_table, and description.
- ctx["dbbench"] includes sql_history, sql_history_raw, discovered_columns,
  db_response, and round.
- ctx["predicates"] includes no_sql_yet, commit_before_sql, commit_empty_answer,
  mutation_task, mutation_not_attempted, last_result_error, last_result_empty,
  unknown_column_error, syntax_error, repeated_sql, candidate_answer_available,
  remaining_rounds_low, and text_only_loop.
- SELECT-style tasks are judged from the submitted answer values.
- INSERT/UPDATE/DELETE tasks are judged from the final database state. A
  successful mutation may return an empty SQL result list; do not treat empty
  mutation results as failure by themselves.
"""


def build_prefill_prompt(evidence: str, max_evidence_chars: int) -> list[dict[str, str]]:
    if max_evidence_chars > 0 and len(evidence) > max_evidence_chars:
        evidence = evidence[:max_evidence_chars].rstrip() + "\n\n...[evidence truncated for SFT length budget]..."
    user = "\n\n".join(
        [
            "You will edit only the reusable DBBench harness, not task answers.",
            (
                "The chat template has already opened the assistant thinking block. "
                "Continue concise recurring-failure reasoning, close it with </think>, "
                "then output exactly one <patch> block."
            ),
            schema_prompt(
                "dbbench",
                response_protocol="prefill_think_patch",
                schema_style="dbbench_life_multihook_v1",
            ),
            dbbench_runtime_context_section(),
            "Observed no-harness rollout evidence:",
            evidence,
        ]
    )
    return [
        {"role": "system", "content": DBBENCH_PREFILL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def clean_reasoning(text: str, max_words: int) -> str:
    text = re.sub(r"</?(?:think|patch)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json|python|text)?[\s\S]*?```", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[\s\S]*?\}", " ", text)
    text = text.replace("`", " ")
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'").strip()
    if not text:
        text = "Analyze recurring DBBench SQL-agent failures and choose a reusable harness code-hook edit."
    words = text.split()
    if max_words > 0 and len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(" ,.;:") + "."
    return text


def load_reasoning(batch_dir: Path, patch: dict[str, Any], max_words: int) -> str:
    raw_path = batch_dir / "raw_response.txt"
    if raw_path.exists():
        raw = raw_path.read_text(encoding="utf-8", errors="replace")
        if "</think>" in raw.lower():
            raw = re.split(r"</think>", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        elif "<patch>" in raw.lower():
            raw = re.split(r"<patch>", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        return clean_reasoning(raw, max_words)
    return clean_reasoning(str(patch.get("description") or ""), max_words)


def build_target(reasoning: str, patch: dict[str, Any]) -> str:
    patch_text = json.dumps(patch, ensure_ascii=False, indent=2)
    return f"{reasoning}\n</think>\n<patch>\n{patch_text}\n</patch>"


def merged_records(
    original_root: Path,
    repair_root: Path | None,
    rerun_root: Path | None,
) -> dict[str, tuple[dict[str, Any], Path]]:
    original = load_results(original_root)
    repair = load_results(repair_root) if repair_root else {}
    rerun = load_results(rerun_root) if rerun_root else {}
    out: dict[str, tuple[dict[str, Any], Path]] = {}
    for tag, result in original.items():
        artifact_root = original_root
        chosen = result
        if result.get("eval_status") != "ok" and tag in repair:
            chosen = repair[tag]
            artifact_root = repair_root or original_root
        if tag in rerun and rerun[tag].get("eval_status") == "ok":
            chosen = rerun[tag]
            # The rerun root reuses patch files but does not have teacher raw
            # responses. Keep the original/repair artifact root for SFT target.
        out[tag] = (chosen, artifact_root)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, default=None)
    parser.add_argument("--eval-failed-rerun-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--min-delta-pass", type=int, default=0)
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--summary-max-words", type=int, default=160)
    parser.add_argument("--max-evidence-chars", type=int, default=26000)
    args = parser.parse_args()

    original_root = args.original_root.expanduser().resolve()
    repair_root = args.repair_root.expanduser().resolve() if args.repair_root else None
    rerun_root = args.eval_failed_rerun_root.expanduser().resolve() if args.eval_failed_rerun_root else None
    merged = merged_records(original_root, repair_root, rerun_root)

    stats: dict[str, Any] = {
        "total_merged_batches": len(merged),
        "valid_eval_ok": 0,
        "invalid_or_eval_failed": 0,
        "positive": 0,
        "zero": 0,
        "negative": 0,
        "selected": 0,
        "min_delta_pass": args.min_delta_pass,
        "include_zero": args.include_zero,
        "include_negative": args.include_negative,
        "summary_max_words": args.summary_max_words,
        "max_evidence_chars": args.max_evidence_chars,
        "source_roots": {
            "original": str(original_root),
            "repair": str(repair_root) if repair_root else None,
            "eval_failed_rerun": str(rerun_root) if rerun_root else None,
        },
    }
    examples: list[dict[str, Any]] = []
    for tag in sorted(merged, key=lambda item: int(item.split("_", 1)[1])):
        result, artifact_root = merged[tag]
        if not (result.get("valid_patch") and result.get("eval_status") == "ok"):
            stats["invalid_or_eval_failed"] += 1
            continue
        stats["valid_eval_ok"] += 1
        delta = int(result.get("delta_pass") or 0)
        if delta > 0:
            stats["positive"] += 1
        elif delta == 0:
            stats["zero"] += 1
        else:
            stats["negative"] += 1
        selected = delta >= args.min_delta_pass
        if delta == 0 and args.include_zero:
            selected = True
        if delta < 0 and args.include_negative:
            selected = True
        if not selected:
            continue

        batch_dir = artifact_root / tag
        prompt_path = batch_dir / "prompt.json"
        patch_path = batch_dir / "patch.json"
        if not prompt_path.exists() or not patch_path.exists():
            stats["invalid_or_eval_failed"] += 1
            continue
        patch = read_json(patch_path)
        prompt = read_json(prompt_path)
        evidence = extract_evidence(prompt)
        reasoning = load_reasoning(batch_dir, patch, args.summary_max_words)
        target = build_target(reasoning, patch)
        messages = build_prefill_prompt(evidence, args.max_evidence_chars)
        metadata = {
            "benchmark": "dbbench",
            "batch_tag": tag,
            "artifact_root": str(artifact_root),
            "patch_path": str(patch_path),
            "target_style": "prefill_think",
            "prompt_rewrite_protocol": "prefill_think_patch",
            "schema_style": "dbbench_life_multihook_v1",
            "prompt_version": "dbbench_life_multihook_v1_runtimectx_20260626",
            "start": result.get("start"),
            "end": result.get("end"),
            "baseline_pass": result.get("baseline_pass"),
            "patched_pass": result.get("patched_pass"),
            "delta_pass": delta,
            "delta_pass_rate": result.get("delta_pass_rate"),
            "eval_status": result.get("eval_status"),
            "patch_status": result.get("patch_status"),
        }
        examples.append({"messages": messages + [{"role": "assistant", "content": target}], "metadata": metadata})
        stats["selected"] += 1
        if args.max_examples and stats["selected"] >= args.max_examples:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{args.dataset_name}.json"
    write_json(args.output_dir / file_name, examples)
    write_dataset_info(args.output_dir, args.dataset_name, file_name)
    write_json(args.output_dir / f"{args.dataset_name}.stats.json", stats)
    print(json.dumps({"dataset_dir": str(args.output_dir), "dataset_name": args.dataset_name, **stats}, indent=2))
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
