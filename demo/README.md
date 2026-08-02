# Offline Demo

```bash
python demo/run_demo.py
```

No GPU, no model endpoint, no benchmark assets. Runs in about a second and exits
non-zero if anything fails to validate or a guard decision changes.

The demo walks one real editing instance end to end:

1. **Failure evidence** — the frozen Qwen3.5-9B target's baseline on WebShop
   tasks 80–89: 1/10 fully successful, mean reward 0.587.
2. **Engineer patch** — the harness engineer's actual reasoning and the patch it
   produced: a single `on_before_action` guard on `Buy Now`.
3. **Validation and sandbox** — this repository's real
   `normalize_patch`, `require_code_hook_only_patch`, and `compile_hook` accept
   and compile the patch under the AST safety and leakage rules.
4. **Replay** — three recorded WebShop product-page states are pushed through the
   compiled guard, and its actual returned effect is printed.

The guard fires where it should and stays out of the way where it should not:

| Scenario | Pending action | Guard effect |
|---|---|---|
| Required flavor has no option on the page | `click[buy now]` | `block_and_prompt` — go back to search |
| Required color visible but unselected | `click[buy now]` | `rewrite_action` → `click[gray]` |
| Options already selected | `click[buy now]` | no intervention |

Installing that patch and rerunning the same ten tasks took the frozen target
from **1/10 to 5/10** (mean reward 0.587 → 0.768, engineer reward **+0.182**).

## What is stored and what is computed

Everything under `artifacts/` is copied from one real stored evaluation. Nothing
about the patch or the numbers is written for the demo:

| File | Contents |
|---|---|
| `metadata.json` | Baseline metadata: task range, per-task baseline rewards, target identity, prompt and response protocol |
| `engineer_think.txt` | The engineer's verbatim `<think>` block |
| `patch.json` | The verbatim generated patch |
| `result.json` | The rerun outcome: baseline vs patched pass counts and rewards |
| `replay_states.json` | Runtime contexts for the replay (see provenance note below) |

The **validation results and the guard decisions are not stored** — they are
computed when you run the script, by importing
[`harness_r1_patch.py`](../code/life-harness/AgentBench/scripts/harness_r1_patch.py)
and
[`code_runner.py`](../code/life-harness/AgentBench/src/server/harness/code_runner.py)
directly. Editing `artifacts/patch.json` changes what the demo prints; breaking
the code makes the sandbox reject it.

## Provenance of the replay states

`replay_states.json` follows the documented WebShop `ctx` contract (see the
`webshop` branch of `schema_prompt()` in `harness_r1_patch.py`). Instructions,
product titles, prices, and clickable lists are copied verbatim from the stored
baseline trajectories of batch 008. Each scenario carries a `provenance` field:

- `task-85-missing-flavor` and `task-83-options-selected` are **recorded** — both
  the state and the pending action are what the target actually did.
- `task-83-unselected-options` is **counterfactual** — the page state is verbatim,
  but the pending action is set to `click[buy now]` to ask what the guard would
  rule at that point. In the stored run the target selected the options itself,
  so the guard was never consulted there.

## Scope

This is one batch from one stored evaluation, chosen because a single-hook patch
makes the mechanism legible. It is not the paper's headline result and not a
benchmark run. The aggregate numbers are in [../docs/RESULTS.md](../docs/RESULTS.md);
the reward definition and the same-batch rerun protocol are in
[../docs/METHOD.md](../docs/METHOD.md). To run real evaluations against a served
target, see [Evaluation](../README.md#evaluation).
