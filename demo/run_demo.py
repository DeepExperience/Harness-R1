#!/usr/bin/env python3
"""Offline Harness-R1 demo: validate and execute a stored engineer patch.

No GPU, no model endpoint, and no benchmark assets are required. The patch, the
engineer's reasoning, the baseline metadata, and the batch outcome are stored
artifacts from a real evaluation; the validation and the hook decisions printed
below are computed live by this repository's own validator and sandbox.

    python demo/run_demo.py            # colored output when the terminal supports it
    python demo/run_demo.py --no-color
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

sys.path.insert(0, str(ROOT / "code/life-harness/AgentBench/scripts"))
sys.path.insert(0, str(ROOT / "code/life-harness/AgentBench"))

from harness_r1_patch import (  # noqa: E402
    PatchValidationError,
    normalize_patch,
    require_code_hook_only_patch,
)
from src.server.harness.code_runner import (  # noqa: E402
    HookCompileError,
    compile_hook,
    run_hook,
)

COLOR = True


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def dim(t: str) -> str:
    return c(t, "2")


def bold(t: str) -> str:
    return c(t, "1")


def green(t: str) -> str:
    return c(t, "32")


def red(t: str) -> str:
    return c(t, "31")


def yellow(t: str) -> str:
    return c(t, "33")


def cyan(t: str) -> str:
    return c(t, "36")


def step(n: int, title: str, subtitle: str) -> None:
    print()
    print(f"{bold(cyan(f'[{n}/4]'))} {bold(title)}  {dim(subtitle)}")
    print(dim("─" * 74))


def wrap(text: str, width: int = 70, indent: str = "      ") -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(indent + line for line in lines)


def clip(text: str, width: int = 64) -> str:
    """Truncate at a word boundary so the demo output never cuts mid-word."""
    if len(text) <= width:
        return text
    head = text[: width - 1]
    if " " in head:
        head = head[: head.rfind(" ")]
    return head + "…"


def load(name: str) -> Any:
    return json.loads((ARTIFACTS / name).read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color")
    parser.add_argument(
        "--force-color",
        action="store_true",
        help="emit ANSI color even when stdout is not a terminal",
    )
    args = parser.parse_args()

    global COLOR
    COLOR = not args.no_color and (args.force_color or sys.stdout.isatty())

    meta = load("metadata.json")
    patch_raw = load("patch.json")
    result = load("result.json")
    replay = load("replay_states.json")
    think = (ARTIFACTS / "engineer_think.txt").read_text().strip()

    print()
    print(bold("  Harness-R1 — failure evidence → executable patch → runtime guard"))
    print(dim("  Stored artifacts; validation and hook execution run live."))

    # ------------------------------------------------------------------ 1
    step(1, "Failure evidence", f"{meta['benchmark']} tasks {meta['start']}–{meta['end'] - 1}")
    rewards = meta["baseline_rewards"]
    print(f"      frozen target : {meta['target_model']}  ({meta['target_agent_name']})")
    print(
        f"      baseline      : {red(str(meta['baseline_pass']) + '/' + str(meta['batch_size']))}"
        f" fully successful   avg reward "
        f"{red(f'{sum(rewards.values()) / len(rewards):.3f}')}"
    )
    print()
    print(dim("      per-task baseline reward"))
    line = "      "
    for task_id in sorted(rewards, key=int):
        value = rewards[task_id]
        mark = green(f"{value:.2f}") if value >= 1.0 else red(f"{value:.2f}")
        line += f"{dim(task_id)}:{mark}  "
    print(line)

    # ------------------------------------------------------------------ 2
    step(2, "Engineer patch", f"{patch_raw.get('schema_version', 'n/a')}")
    print(dim("      <think>"))
    excerpt = think.split("\n\n")[0]
    print(wrap(excerpt))
    print(dim("      </think>"))
    print()
    print(f"      description : {patch_raw.get('description', '')}")
    hooks = [a["hook"] for a in patch_raw["actions"]]
    print(f"      hooks       : {', '.join(bold(h) for h in hooks)}")

    # ------------------------------------------------------------------ 3
    step(3, "Validation and sandbox", "this repository's real checks")
    try:
        patch = normalize_patch(patch_raw, bench=patch_raw["benchmark"])
        require_code_hook_only_patch(patch)
    except PatchValidationError as exc:
        print(f"      {red('REJECTED')}  {exc}")
        return 1
    print(f"      {green('ok')}  schema, benchmark, and hook set        {dim('normalize_patch')}")
    print(f"      {green('ok')}  code-hook-only protocol                {dim('require_code_hook_only_patch')}")

    compiled: dict[str, Any] = {}
    for action in patch["actions"]:
        try:
            compiled[action["hook"]] = compile_hook(
                action["code"], benchmark=patch["benchmark"]
            )
        except HookCompileError as exc:
            print(f"      {red('REJECTED')}  {action['hook']}: {exc}")
            return 1
    print(
        f"      {green('ok')}  AST safety, leakage, and compilation    "
        f"{dim('compile_hook × ' + str(len(compiled)))}"
    )

    guard = compiled.get("on_before_action")
    if guard is None:
        print(f"      {red('no on_before_action hook to replay')}")
        return 1

    # ------------------------------------------------------------------ 4
    step(4, "Replay", "recorded WebShop states through the compiled guard")
    failures = 0
    for scenario in replay["scenarios"]:
        ctx = scenario["ctx"]
        notebook: dict[str, Any] = {}
        effect = run_hook(guard, ctx, notebook, hook_name="on_before_action")
        kind = (effect or {}).get("kind", "no intervention")
        ok = kind == scenario["expect"]
        failures += 0 if ok else 1

        print()
        print(f"      {bold(scenario['title'])}  {dim('(' + scenario['id'] + ')')}")
        print(f"      instruction  {dim(clip(scenario['instruction']))}")
        print(f"      page         {dim(clip(scenario['product']))}")
        print(
            f"      price        ${ctx['state']['current_price']:.2f} "
            f"{dim('/ budget $' + format(ctx['state']['price_max'], '.2f'))}"
        )
        print(f"      target wants {yellow(ctx['action']['final_action'])}")

        if effect is None:
            print(f"      guard        {green('passes the action through')}")
        else:
            label = red(kind) if kind == "block_and_prompt" else yellow(kind)
            print(f"      guard        {label}", end="")
            if effect.get("action"):
                print(f" → {green('click[' + effect['action'] + ']')}")
            else:
                print()
            if effect.get("message"):
                print(wrap(effect["message"], indent="                   " + dim("│ ")))
        print(f"      {green('✓') if ok else red('✗')} expected {scenario['expect']}")

    # ------------------------------------------------------------------ end
    print()
    print(dim("─" * 74))
    size = result["batch_size"]
    baseline_pass = "{}/{}".format(result["baseline_pass"], size)
    patched_pass = "{}/{}".format(result["patched_pass"], size)
    baseline_avg = "{:.3f}".format(result["baseline_average_reward"])
    patched_avg = "{:.3f}".format(result["patched_average_reward"])
    delta = "{:+.3f}".format(result["delta_average_reward"])

    print("  batch outcome after installing this patch and rerunning the same tasks:")
    print(
        "    success  {} → {}     avg reward  {} → {}     engineer reward  {}".format(
            red(baseline_pass), green(patched_pass), red(baseline_avg), green(patched_avg), bold(delta)
        )
    )
    print()

    if failures:
        print(red(f"  {failures} scenario(s) did not match the expected effect"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
