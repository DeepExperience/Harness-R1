# Case Studies

What a learned harness patch actually does at runtime.

These are stored evaluations from the validation-selected Harness-R1 engineer
used for the main frozen-target results. In every case the target agent is the
same frozen Qwen3.5-9B model before and after patch installation, and the patched
run uses the same ten tasks as its baseline evidence.

We inspect the **runtime trace** in addition to the generated code, so that an
intended edit is not mistaken for an intervention that actually executed. These
cases illustrate distinct mechanisms and limitations; the aggregate claims live
in [RESULTS.md](RESULTS.md), not in these selected examples.

| Environment | Success before | Success after | Δ | Primary behavior illustrated |
|---|---|---|---|---|
| [WebShop](#webshop-correcting-a-premature-purchase) | 2/10 | 5/10 | +3 | A narrow guard delays purchase until required options are selected |
| [ALFWorld](#alfworld-coordinating-multiple-lifecycle-positions) | 1/10 | 6/10 | +5 | State tracking, stage guidance, and an action guard form a closed loop |
| [DBBench](#dbbench-preserving-schema-and-stored-value-conventions) | 4/10 | 6/10 | +2 | Schema recovery and format-preserving mutation outperform a valid GLM-5.2 edit |

A [fourth case](#a-failure-case-of-direct-harness-editing) shows a frontier
editor whose plausible diagnosis compiles into behavior that *lowers* success —
the failure mode outcome-grounded training is designed to remove.

## WebShop: correcting a premature purchase

**Observed failure.** In WebShop batch 008, several trajectories reached a
relevant, in-budget product but issued `Buy Now` before choosing an option the
instruction required.

**Generated edit.** A single pre-action intervention. It activates only for a
normalized `Buy Now` action, and blocks it when either the current price exceeds
the budget or a required product option remains unselected. The message asks the
target to choose a visible matching option, or return to search if no such option
exists.

**Task-level behavior.** One task requests a synthetic hairpiece in black-brown
under $40. The baseline target finds a suitable product but buys it without
selecting the color, scoring a partial 0.667. With the patch installed the target
still proposes the same premature purchase — the guard blocks it, the target then
selects `black brown`, and the purchase scores 1.0.

| Run | Relevant action sequence | Reward |
|---|---|---|
| No intervention | search → open product → `Buy Now` with color unselected | 0.667 |
| Harness-R1 | search → open product → attempted `Buy Now` → guard message → select `black brown` → `Buy Now` | **1.000** |

Across the ten-task batch the same guard raises full successes from 2 to 5 and
mean WebShop reward from 0.682 to 0.768, while preserving both tasks that were
already fully successful.

> An effective harness edit need not replace the target's policy with a large
> controller. A low-bandwidth intervention at the point of an unsafe action
> preserves the target's search behavior while changing the final outcome.

## ALFWorld: coordinating multiple lifecycle positions

**Observed failure.** ALFWorld batch 045 contains recurrent failures where the
target finds an object but omits a required transformation, moves toward the
wrong receptacle, or enters a transform–place loop.

**Generated edit.** The patch coordinates all four lifecycle positions:

| Lifecycle position | Installed behavior |
|---|---|
| Episode initialization | Initialize the current stage and provide the reusable find–take–transform–place ordering |
| After environment feedback | Update whether the target is held, the required transformation is complete, and the destination has been reached |
| Before model decision | Inject a stage-specific hint for the next unresolved subgoal |
| Before environment execution | Block a premature placement, or a placement at the wrong destination |

**Runtime behavior.** The intervention trace records **56** stage hints or guard
messages across the batch.

| Example | Effective intervention sequence | Outcome |
|---|---|---|
| Hot mug in cabinet | take-target hint → heat-at-microwave hint → go-to-destination hint → put-target hint | failure → success |
| Cooled egg on countertop | stage hint → attempted wrong placement → destination guard → corrected placement | failure → success |

At batch level the patch rescues six baseline failures but regresses one baseline
success, a net 1/10 → 6/10. It does not resolve everything: one two-object
trajectory continues to alternate between destination and placement guidance.

> This is a genuine closed-loop harness policy — persistent state, conditioned
> guidance, and a guard that acts on it — while also showing that stage tracking
> can remain imperfect.

## DBBench: preserving schema and stored-value conventions

**Observed failure.** DBBench batch 022 contains recurring failures around
multi-word identifiers, schema recovery, and exact mutation values.

**Generated edit.** A stage-aware patch that recommends schema inspection after
identifier errors, asks the target to inspect the affected row before mutation,
and verifies the stored value before commit. It raises the frozen target from
4/10 to 6/10. On the same baseline evidence and tasks, a valid GLM-5.2 patch
reaches only 5/10.

**Task-level contrast.** One task asks the agent to change the length of the
`Moosehead Grand Prix` entry in the multi-word table `Race Schedule`. The
no-intervention trajectory recovers the quoted table name and observes the
existing value `3 Hours`, but writes `4 hours`; the exact-format evaluator marks
it incorrect. GLM-5.2 supplies general backtick and mutation-verification
guidance, yet its guided trajectory makes the same lower-case write. Harness-R1
first triggers schema recovery, inspects the existing row, writes `4 Hours` to
match the stored convention, and verifies the row before committing.

| Runtime condition | Batch success | Example outcome |
|---|---|---|
| No intervention | 4/10 | `4 hours` (failure) |
| GLM-5.2 patch | 5/10 | `4 hours` (failure) |
| **Harness-R1 patch** | **6/10** | `4 Hours` (success) |

> This paired example does not rely on an invalid competitor output — both
> engineers produce executable patches. The difference is that the
> Harness-R1-guided run converts schema and row evidence into the exact stored
> representation the task requires.

## A failure case of direct harness editing

An off-the-shelf model can access the complete lifecycle interface and still
produce harmful interventions. On ALFWorld, Gemini-3.5-Flash receives full
failure evidence and may edit all four lifecycle positions. Its patches reduce
success from 208/500 (41.6%) to 177/500 (35.4%), a drop of 6.2 pp. Among 39 valid
patches, **21 reduce** batch success, 12 improve it, and 6 leave it unchanged.

The largest regression is a batch falling from 7/10 to 0/10. That patch installs
broad `on_before_action` rules that force actions from a locally plausible stage
estimate. On two-object tasks it prematurely places the first object instead of
collecting both before placement — overriding decisions the frozen target had
previously executed correctly.

> Execution traces plus a powerful base model do not yield a reliable harness
> editor. A plausible diagnosis can still compile into overly aggressive runtime
> behavior. Harness-R1 post-trains the editing policy on realized task outcomes,
> which directly penalizes patches that degrade rerun performance.

## Cross-case interpretation

- **WebShop.** The recurring failure is premature purchase. Harness-R1 installs a
  narrow action guard conditioned on runtime predicates — though the guard cannot
  repair an earlier choice of the wrong product.
- **ALFWorld.** The recurring failures are omitted transformations and incorrect
  placement. Harness-R1 combines persistent stage state, targeted hints, and a
  placement guard, while routing and two-object state can still cause regressions
  or loops.
- **DBBench.** The recurring failure is format and schema mismatch. Harness-R1
  routes the target through evidence collection before mutation, which a
  generically worded competitor patch does not achieve.

## See also

- [PATCH_FORMAT.md](PATCH_FORMAT.md) — the JSON contract these patches conform to
- [examples/webshop_patch.json](../examples/webshop_patch.json) — a complete patch
- [RESULTS.md](RESULTS.md) — the aggregate numbers these cases sit inside
