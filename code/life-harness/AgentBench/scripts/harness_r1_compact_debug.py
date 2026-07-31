#!/usr/bin/env python3
"""Build a sanitized Harness-R1 engineer input from debugger outputs.

The raw debugger overviews are useful for audit, but too detailed for a harness
engineer model: they contain task ids, product ids, product titles, and
per-example reasoning.  This script compresses successful debugger analyses into
aggregate failure modes that are easier to convert into typed harness actions.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureMode:
    key: str
    title: str
    priority: str
    keywords: tuple[str, ...]
    symptoms: tuple[str, ...]
    runtime_signals: tuple[str, ...]
    candidate_actions: tuple[str, ...]
    risks: tuple[str, ...]
    dsl_coverage: str


FAILURE_MODES = [
    FailureMode(
        key="attribute_selection",
        title="Required attributes or variants are not selected or not confirmed",
        priority="high",
        keywords=(
            "attribute",
            "variant",
            "select",
            "selected",
            "option",
            "color",
            "size",
            "style",
            "finish",
            "no visual confirmation",
            "identical page state",
        ),
        symptoms=(
            "Agent buys while selectable product options are still available.",
            "Agent clicks a variant but the next observation gives no explicit selected-state feedback.",
            "Agent treats available options as already selected.",
        ),
        runtime_signals=(
            "action.tool == click_action",
            "action.value_normalized == buy now",
            "pred.required_options_unselected",
            "state.buy_now_available",
            "state.product_stall_turns",
            "task.required_color / task.required_size / task.required_material",
        ),
        candidate_actions=(
            "add_guard_rule: block buy now when required_options_unselected is true.",
            "add_or_edit_skill: always identify required option groups and selected values before buying.",
            "add_recovery_rule: if product page appears stalled after option clicks, inject a selected-state verification hint.",
            "edit_tool_hint(click_action): describe variant clicks as required selections, not optional navigation.",
        ),
        risks=(
            "Do not tell the agent which exact option is correct; only expose selection status or a generic checklist.",
            "Some products have no required options, so guards should depend on runtime predicates, not all product pages.",
        ),
        dsl_coverage="strong",
    ),
    FailureMode(
        key="pre_purchase_verification",
        title="Purchase happens before all instruction requirements are verified",
        priority="high",
        keywords=(
            "verify",
            "verification",
            "unverified",
            "description",
            "features",
            "not visible",
            "assumed",
            "premature purchase",
            "before buying",
            "before purchasing",
        ),
        symptoms=(
            "Agent buys from a title-level match without checking details for hidden requirements.",
            "Agent explicitly notes uncertainty about a requirement but proceeds to buy.",
            "Description/features are available, but the agent does not inspect them when title evidence is incomplete.",
        ),
        runtime_signals=(
            "state.page_type == product",
            "state.buy_now_available",
            "action.value_normalized == buy now",
            "state.remaining_steps",
        ),
        candidate_actions=(
            "add_or_edit_skill: before buy now, list each requirement and whether it is visible, selected, or still unknown.",
            "edit_tool_hint(click_action): buy now should only be used after required attributes and hidden requirements are checked.",
            "add_recovery_rule: when product page stalls, suggest checking details or returning to search if a requirement is unknown.",
        ),
        risks=(
            "Forcing detail-page clicks on every product can waste turns; prefer soft hints unless a required attribute is unknown.",
            "Do not encode product-category-specific facts as universal rules.",
        ),
        dsl_coverage="partial",
    ),
    FailureMode(
        key="search_loop",
        title="Search or pagination loops consume the turn budget without purchase",
        priority="high",
        keywords=(
            "search loop",
            "repeated search",
            "pagination",
            "next >",
            "task limit",
            "never clicked",
            "without completing",
            "no purchase",
            "exhausted",
        ),
        symptoms=(
            "Agent repeatedly searches or paginates while avoiding product inspection.",
            "Agent rejects all visible candidates for an exact match and reaches the task limit.",
            "Agent does not switch from exploration to best-available purchase near the step budget.",
        ),
        runtime_signals=(
            "pred.search_loop_detected",
            "pred.duplicate_search_repeated",
            "state.back_to_search_count",
            "state.duplicate_search_count",
            "state.remaining_steps",
        ),
        candidate_actions=(
            "set_config: lower duplicate-search and search-loop thresholds for WebShop.",
            "add_recovery_rule: on search_loop_detected, ask the agent to inspect a promising candidate or simplify the query.",
            "add_recovery_rule: when remaining steps are low, prompt a best-current-candidate decision instead of another search.",
        ),
        risks=(
            "Over-aggressive fallback may cause wrong purchases when exact matches are available on later pages.",
            "Best-candidate prompts should be generic and never point to a specific product.",
        ),
        dsl_coverage="strong",
    ),
    FailureMode(
        key="semantic_mismatch",
        title="Core product type or semantic attribute is confused",
        priority="medium",
        keywords=(
            "wrong product type",
            "product type",
            "category",
            "semantic",
            "misinterpreted",
            "color mismatch",
            "brand",
            "entity",
            "does not match",
            "suboptimal product",
        ),
        symptoms=(
            "Agent prioritizes a visible secondary attribute while missing the required product type.",
            "Agent treats a related color, brand, count, packaging, or qualifier as equivalent without evidence.",
            "Agent buys a plausible but semantically different item.",
        ),
        runtime_signals=(
            "task.task_type",
            "task.required_color",
            "action.value_normalized",
            "state.page_type",
        ),
        candidate_actions=(
            "add_or_edit_skill: hard constraints are product type, explicit brand/entity, required color, count, material, and price.",
            "edit_tool_hint(search_action): refine queries with missing hard constraints when results are only partial matches.",
            "add_recovery_rule: on product_page_stalled, suggest comparing the title/options against all hard constraints.",
        ),
        risks=(
            "Semantic mismatch detection is only partial in the current DSL; avoid hard blocks for nuanced language.",
            "Do not use concrete examples from individual tasks as rules.",
        ),
        dsl_coverage="partial",
    ),
    FailureMode(
        key="over_strict_matching",
        title="Agent is too strict about exact specs and misses acceptable candidates",
        priority="medium",
        keywords=(
            "overly strict",
            "exact match",
            "close match",
            "closest match",
            "tolerance",
            "minor spec",
            "settle",
            "best match",
        ),
        symptoms=(
            "Agent rejects near matches indefinitely even when other constraints are satisfied.",
            "Agent searches for exact dimensions, count, or wording until the turn limit.",
            "Agent does not use a best-available fallback when the catalog lacks an exact match.",
        ),
        runtime_signals=(
            "state.remaining_steps",
            "pred.search_loop_detected",
            "pred.product_page_stalled",
        ),
        candidate_actions=(
            "add_or_edit_skill: distinguish hard constraints from soft/tolerance constraints, especially under low step budget.",
            "add_recovery_rule: when remaining steps are low after repeated search, choose the best candidate that satisfies hard constraints.",
        ),
        risks=(
            "Tolerance rules can lower precision for tasks requiring exact values.",
            "Keep tolerance guidance conservative and step-budget dependent.",
        ),
        dsl_coverage="partial",
    ),
    FailureMode(
        key="tool_protocol_or_completion",
        title="Tool-call or post-purchase protocol causes loops",
        priority="low",
        keywords=(
            "text-only",
            "tool call",
            "empty tool_calls",
            "must call a tool",
            "post-purchase",
            "completion",
            "after clicking buy now",
        ),
        symptoms=(
            "Agent emits text instead of a tool call after purchase-related actions.",
            "Agent clicks buy now but then loops because completion state is unclear.",
        ),
        runtime_signals=(
            "action.value_normalized == buy now",
            "state.remaining_steps",
        ),
        candidate_actions=(
            "edit_tool_hint(click_action): after selecting all attributes, buy now is the terminal purchase action.",
            "Future runtime extension: detect post-buy completion instead of forcing continued actions.",
        ),
        risks=(
            "Some of this is benchmark protocol handling rather than harness editing.",
            "Current typed DSL only partially covers post-purchase termination.",
        ),
        dsl_coverage="weak",
    ),
]


FORBIDDEN_PATTERNS = [
    (re.compile(r"\b[bB]0[0-9A-Za-z]{8}\b"), "[PRODUCT_ID]"),
    (re.compile(r"\bwebshop-b\d{3}-[A-Za-z0-9_-]+\b"), "[TRACE_ID]"),
    (re.compile(r"\bglobal=\d+\b"), "global=[ID]"),
    (re.compile(r"\bindex=\d+\b"), "index=[ID]"),
    (re.compile(r"/mnt/[^\s`]+"), "[PATH]"),
]


def sanitize(text: str) -> str:
    out = text
    for pattern, repl in FORBIDDEN_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def batch_dirs(run_root: Path, bench: str, batch_ids: set[int] | None) -> list[Path]:
    out = []
    for path in sorted((run_root / bench).glob("batch_*")):
        try:
            batch_id = int(path.name.split("_", 1)[1])
        except Exception:
            continue
        if batch_ids is not None and batch_id not in batch_ids:
            continue
        out.append(path)
    return out


def classify(response: str) -> list[str]:
    low = response.lower()
    matched = []
    for mode in FAILURE_MODES:
        if any(keyword in low for keyword in mode.keywords):
            matched.append(mode.key)
    return matched


def summarize(args: argparse.Namespace) -> str:
    batches = batch_dirs(args.run_root.resolve(), args.bench, args.batch_ids)
    if not batches:
        raise SystemExit("no batch dirs found")

    selected_rewards: list[float] = []
    selected_statuses = Counter()
    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    debug_rows: list[dict[str, Any]] = []
    failed_errors = Counter()
    rollout_rows = 0
    debug_records = 0
    batch_names = []

    for batch in batches:
        batch_names.append(batch.name)
        manifest_path = batch / "debug/manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rollout_rows += int(manifest.get("runs_count") or 0)
            debug_records += int(manifest.get("records_count") or 0)
            for item in manifest.get("selected") or []:
                try:
                    selected_rewards.append(float(item.get("reward") or 0.0))
                except Exception:
                    pass
                selected_statuses[str(item.get("status") or "unknown")] += 1
        for row in read_jsonl(batch / "debug/results.jsonl"):
            debug_rows.append(row)
            if row.get("status") != "success":
                failed_errors[str(row.get("error") or "unknown")] += 1
                continue
            modes = classify(row.get("response") or "")
            for key in modes:
                rows_by_mode[key].append(row)

    success_count = sum(1 for row in debug_rows if row.get("status") == "success")
    fail_count = len(debug_rows) - success_count
    avg_reward = sum(selected_rewards) / len(selected_rewards) if selected_rewards else 0.0
    median_reward = statistics.median(selected_rewards) if selected_rewards else 0.0

    lines = [
        "# Harness-R1 Sanitized Engineer Input",
        "",
        "This file is the only failure-analysis context intended for the harness engineer model.",
        "It aggregates debugger findings and removes task/product-specific details.",
        "",
        "## Scope",
        "",
        f"- Benchmark: `{args.bench}`",
        f"- Source run: `{args.run_root}`",
        f"- Batches: `{', '.join(batch_names)}`",
        f"- Rollout rows covered by manifests: `{rollout_rows}`",
        f"- Selected non-full/error traces: `{len(selected_rewards)}`",
        f"- Debug records requested: `{debug_records}`",
        f"- Debug rows available: `{len(debug_rows)}` (`{success_count}` success, `{fail_count}` failed)",
        f"- Selected reward mean/median: `{avg_reward:.4f}` / `{median_reward:.4f}`",
        f"- Selected statuses: `{dict(selected_statuses)}`",
        "",
        "## Harness Engineer Instructions",
        "",
        "- Propose only general changes that could help future tasks.",
        "- Do not mention product ids, exact product names, task indices, or benchmark answers.",
        "- Prefer typed actions already supported by Harness-R1.",
        "- Prefer soft, auditable interventions for semantic uncertainty; reserve hard guards for runtime predicates with clear meaning.",
        "",
        "## Aggregated Failure Modes",
        "",
    ]

    total_success = max(1, success_count)
    for mode in FAILURE_MODES:
        rows = rows_by_mode.get(mode.key, [])
        if not rows:
            continue
        lines.extend(
            [
                f"### {mode.title}",
                "",
                f"- Priority: `{mode.priority}`",
                f"- Frequency among successful debugger analyses: `{len(rows)}/{total_success}`",
                "- Observed symptoms:",
            ]
        )
        if not args.hide_dsl_coverage:
            lines.insert(-1, f"- Current DSL coverage: `{mode.dsl_coverage}`")
        lines.extend(f"  - {item}" for item in mode.symptoms)
        if not args.hide_runtime_signals:
            lines.append("- Runtime signals the harness can use:")
            lines.extend(f"  - `{item}`" for item in mode.runtime_signals)
        if args.include_candidate_actions:
            lines.append("- Candidate typed harness changes:")
            lines.extend(f"  - {item}" for item in mode.candidate_actions)
        lines.append("- Generalization risks:")
        lines.extend(f"  - {item}" for item in mode.risks)
        lines.append("")

    if failed_errors:
        lines.extend(
            [
                "## Debugger Reliability Notes",
                "",
                "- Some debugger calls failed to produce parseable answers. Treat successful aggregate patterns as signal, not ground truth.",
                "- Failure types:",
            ]
        )
        for err, count in failed_errors.most_common():
            lines.append(f"  - `{sanitize(err)}`: `{count}`")
        lines.append("")

    if args.include_recommended_focus:
        lines.extend(
            [
                "## Recommended Patch Focus",
                "",
                "A strong first patch should combine:",
                "",
                "1. A guarded `buy now` intervention for clearly unselected required options.",
                "2. A short reusable skill for pre-purchase requirement verification.",
                "3. A recovery rule for repeated search/product-page stall that asks for a best-candidate or refined-query decision.",
                "",
                "Avoid product-specific semantic examples in the patch text.",
            ]
        )
    return sanitize("\n".join(lines).rstrip() + "\n")


def parse_batch_ids(raw: list[str]) -> set[int] | None:
    if not raw:
        return None
    out = set()
    for part in raw:
        for item in part.split(","):
            item = item.strip()
            if item:
                out.add(int(item))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact debugger outputs for harness engineering.")
    parser.add_argument("--bench", choices=["webshop", "alfworld"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-ids", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-candidate-actions",
        action="store_true",
        help="Include hand-written candidate typed harness changes for bootstrap runs.",
    )
    parser.add_argument(
        "--include-recommended-focus",
        action="store_true",
        help="Include the hand-written recommended patch focus section.",
    )
    parser.add_argument(
        "--hide-runtime-signals",
        action="store_true",
        help="Omit allowed runtime signal hints from failure-mode summaries.",
    )
    parser.add_argument(
        "--hide-dsl-coverage",
        action="store_true",
        help="Omit DSL coverage hints from failure-mode summaries.",
    )
    args = parser.parse_args()
    args.batch_ids = parse_batch_ids(args.batch_ids)
    text = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[harness-r1-compact] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
