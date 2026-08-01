# Results

Full result tables from the Harness-R1 paper. All numbers are percentages;
deltas are in percentage points (pp).

Throughout, the **target agent** runs the benchmark tasks and stays frozen, and
the **harness engineer** is the trained model that writes the runtime patch.
Every reported gain comes from rerunning the frozen target on the same tasks
with the generated patch installed — never from a learned judge or a
self-reported score.

- [Main results](#main-results)
- [Target-agent generalization](#target-agent-generalization)
- [Held-out task generalization](#held-out-task-generalization)
- [Lifecycle-position ablation](#lifecycle-position-ablation)
- [Data splits](#data-splits)

## Main results

Target: frozen Qwen3.5-9B. `Score` is the mean shaped WebShop reward, `Succ.` is
task success rate, and `Avg.` is the equal-weight average of WebShop Succ.,
ALFWorld All, and DBBench Succ.

**Bold** marks the highest and _italic_ the second-highest distinct value in
each column, computed over all non-Reflection rows (ties share a mark).

| Method | ALF Pick | Look | Clean | Heat | Cool | Pick2 | **ALF All** | WS Score | **WS Succ.** | **DB Succ.** | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B (default harness) | 75.8 | 66.7 | 16.9 | 28.8 | 9.2 | 47.5 | 40.6 | 66.0 | 31.2 | 61.0 | 44.3 |
| **_Prompt-based agentic methods_** | | | | | | | | | | | |
| ReAct | 79.8 | 54.8 | 27.3 | 27.4 | 10.3 | 53.3 | 43.4 | 66.8 | 37.4 | 61.7 | 47.5 |
| Self-Refine | 70.7 | 45.2 | 11.7 | 28.8 | 6.9 | 57.4 | 39.0 | 61.7 | 29.0 | 57.3 | 41.8 |
| Reflection <sup>‡</sup> | 91.9 | 90.5 | 24.7 | 56.2 | 19.5 | 73.8 | 59.2 | 61.2 | 43.6 | 64.7 | 55.8 |
| **_Frontier models as fixed harness editors_** | | | | | | | | | | | |
| Qwen3.5-397B | 76.8 | 66.7 | 14.3 | 31.5 | 11.5 | 47.5 | 41.2 | 66.4 | 32.8 | 63.3 | 45.8 |
| GLM-5.2 | 77.8 | 78.6 | 18.2 | 27.4 | 16.1 | 54.9 | 45.0 | 68.4 | 36.0 | _65.3_ | 48.8 |
| Kimi-K2.6 | 73.7 | 71.4 | 14.3 | 30.1 | 12.6 | 49.2 | 41.4 | 63.7 | 31.8 | 62.7 | 45.3 |
| DeepSeek-V4-Pro | 72.7 | 69.0 | 13.0 | 30.1 | 16.1 | 47.5 | 41.0 | 65.1 | 32.4 | 64.3 | 45.9 |
| Gemini-3.5-Flash | 67.7 | 69.0 | 14.3 | 24.7 | 10.3 | 35.2 | 35.4 | 63.2 | 33.6 | 64.0 | 44.3 |
| GPT-5.5 | 80.8 | 78.6 | 31.2 | 28.8 | 17.2 | 35.2 | 43.2 | 61.4 | 36.6 | 64.0 | 47.9 |
| **_Ours_** | | | | | | | | | | | |
| Supervised-only engineer | 80.8 | 66.7 | 22.1 | 24.7 | 11.5 | 36.1 | 39.4 | 67.8 | 38.6 | 61.3 | 46.4 |
| **Harness-R1** | 77.8 | _81.0_ | _58.4_ | 43.8 | 34.5 | 39.3 | 53.2 | _69.9_ | 42.2 | _65.3_ | 53.6 |
| Agent SFT | _91.9_ | **100.0** | 46.8 | _56.2_ | _48.3_ | _85.2_ | _71.2_ | **71.5** | _42.6_ | 63.7 | _59.2_ |
| **Agent SFT + Harness-R1** | **93.9** | **100.0** | **81.8** | **74.0** | **72.4** | **86.1** | **84.0** | 68.7 | **43.0** | **65.7** | **64.2** |

<sup>‡</sup> Reflection is reported under a separate two-episode `success@2`
protocol: its success columns are cumulative over two episodes and its Score is
measured after retrying first-episode failures. All other rows are `success@1`,
so Reflection is not ranked against them.

Headline reads:

- **Outcome-trained editing improves the frozen target on every benchmark.**
  Average success rises 44.3 → 53.6 (**+9.3 pp**), with the largest absolute gain
  on ALFWorld (40.6 → 53.2). Harness-R1 is 7.1 pp above the supervised-only
  engineer, which isolates the contribution of online RL over cold-start SFT.
- **A trained 9B engineer beats much larger fixed editors.** The strongest
  frontier editor is GLM-5.2 at 48.8, versus 53.6 for Harness-R1. Frontier models
  optimize for a plausible-looking edit; they never rerun the target, so they
  cannot tell whether an edit actually raises success.
- **Fixed harness patterns are not reliably helpful.** ReAct adds 3.2 pp, while
  Self-Refine *removes* 2.5 pp — one hand-crafted rule applied uniformly ignores
  both the target's specific failure modes and whether the intervention works.
- **The engineer keeps helping after the actor is fine-tuned.** Direct agent SFT
  lifts the unmodified target to 59.2; a target-specific engineer retrained for
  that stronger actor reaches 64.2 (**+5.0 pp**). Harness editing does not
  saturate once the agent improves, which is what makes engineer/target
  co-evolution plausible.

## Target-agent generalization

The single learned editing policy is applied to target agents never seen during
training. Each target supplies its **own** failure traces and receives newly
generated, target-specific patches — this measures transfer of the editing
*policy*, not replay of one fixed patch.

WebShop uses the fixed-seed 500-task rerun (goal seed 233); ALFWorld and DBBench
use their test sets. `Δ Avg.` is the equal-weight average of the three benchmark
deltas. <sup>†</sup> marks the primary Qwen3.5-9B target also used in the main
results.

| Target agent | WS before | WS after | ALF before | ALF after | DB before | DB after | **Δ Avg.** |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 21.2 | 31.0 | 2.0 | 10.0 | 16.7 | 30.3 | +10.5 |
| Llama-3.1-70B | 39.2 | 38.8 | 21.0 | 35.2 | 31.7 | 33.7 | +5.3 |
| Llama-3.2-1B | 0.0 | 0.0 | 0.0 | 0.2 | 8.0 | 14.3 | +2.2 |
| Llama-3.2-3B | 8.4 | 13.6 | 1.2 | 7.2 | 7.0 | 16.7 | +7.0 |
| Llama-3.3-70B | 35.4 | 41.8 | 27.4 | 46.2 | 60.3 | 63.0 | +9.3 |
| Gemma-3-1B | 0.0 | 0.0 | 0.0 | 1.2 | 0.3 | 2.7 | +1.2 |
| Gemma-3-4B | 13.8 | 28.4 | 4.0 | 10.4 | 14.3 | 29.3 | +12.0 |
| Gemma-3-12B | 22.8 | 32.2 | 12.8 | 18.8 | 41.0 | 56.7 | +10.4 |
| Gemma-3-27B | 27.8 | 37.2 | 18.6 | 29.4 | 52.3 | 60.0 | +9.3 |
| Gemma-4-12B-it | 39.4 | 39.4 | 35.8 | 55.2 | 61.3 | 67.7 | +8.6 |
| Gemma-4-26B-A4B-it | 39.2 | 39.6 | 49.2 | 65.4 | 60.0 | 69.0 | +8.5 |
| Gemma-4-31B-it | 42.0 | 41.8 | 54.4 | 73.2 | 65.7 | 69.0 | +7.3 |
| Qwen2.5-72B | 37.8 | 39.6 | 70.8 | 68.8 | 51.3 | 54.7 | +1.0 |
| Qwen3-4B | 25.0 | 38.6 | 22.8 | 29.1 | 38.7 | 55.7 | +12.3 |
| Qwen3-8B | 30.6 | 34.6 | 23.0 | 27.7 | 49.3 | 57.3 | +5.6 |
| Qwen3-14B | 33.8 | 37.2 | 22.3 | 39.0 | 50.0 | 60.0 | +10.0 |
| Qwen3.5-4B | 33.2 | 36.8 | 20.7 | 38.6 | 60.7 | 65.0 | +8.6 |
| **Qwen3.5-9B** <sup>†</sup> | 31.2 | 42.2 | 40.6 | 53.2 | 61.0 | 65.3 | +9.3 |
| Qwen3.5-27B | 42.0 | 42.0 | 72.4 | 81.3 | 70.3 | 72.3 | +3.6 |
| Qwen3.5-35B-A3B | 30.6 | 31.8 | 62.2 | 69.2 | 62.7 | 68.7 | +4.7 |
| Qwen3.6-27B | 43.4 | 44.2 | 70.6 | 78.6 | 69.7 | 72.7 | +3.9 |
| **Mean, 20 unseen targets** | 28.3 | 32.4 | 29.4 | 39.1 | 43.6 | 50.9 | **+7.1** |
| **Mean, all 21 targets** | 28.4 | 32.9 | 30.0 | 39.8 | 44.4 | 51.6 | **+7.2** |

Every target-level average is positive. Across the full 21 × 3 matrix, **56 of
63** target-benchmark combinations improve, four are unchanged (all on WebShop),
and the three regressions are all small: WebShop on Llama-3.1-70B (−0.4),
ALFWorld on Qwen2.5-72B (−2.0), and WebShop on Gemma-4-31B-it (−0.2).

Aggregating matched tasks within each benchmark, gains stay positive at **+4.15**
on WebShop, **+9.63** on ALFWorld, and **+7.37** on DBBench. The benchmark-averaged
gain across the twenty unseen targets is **+7.06 pp**, with no per-target retuning.

## Held-out task generalization

Can a handful of failures produce a patch that helps tasks the engineer never
saw? For each benchmark and seed, every engineer observes the **same 10 failures**
from the frozen Qwen3.5-9B target, generates one benchmark-specific patch, and
applies it to all remaining tasks. Repeated over three matched evidence seeds.
Invalid patches count as no intervention (zero delta).

`Valid` counts how many of the nine seed-benchmark patches installed a real
intervention. Error terms are the sample standard deviation across the three
seeds.

| Engineer | Valid | Held-out (1,270 tasks) | Full split (1,300 tasks) |
|---|---|---|---|
| **Harness-R1** | 9/9 | **+8.9 ± 1.5** | **+9.2 ± 1.5** |
| Qwen3.5-397B | 8/9 | −4.3 ± 2.5 | −3.9 ± 2.5 |
| DeepSeek-V4-Pro | 6/9 | −0.4 ± 3.6 | −0.2 ± 3.5 |

The gap is not only in the mean. Harness-R1 is positive on **every** seed at a
tight ±1.5, whereas both frontier engineers average negative and straddle zero
across seeds (±2.5 and ±3.6), swinging between marginal gains and sizable
regressions. Converting sparse failure evidence into a broadly useful edit is a
capability that scale alone does not confer, and one that outcome-grounded
training makes both stronger and more consistent.

## Lifecycle-position ablation

Holding the frozen vanilla target and the generated patches fixed, one lifecycle
position is disabled at a time. Each configuration reruns the target three times
per benchmark; `Avg.` uses the same equal-benchmark weighting as the main table,
and parenthesized values are the drop relative to the full patch.

| Configuration | WebShop | ALFWorld | DBBench | **Avg.** |
|---|---|---|---|---|
| **Full patch** | 41.6 | 52.1 | 65.4 | **53.1** |
| w/o episode init | 41.5 | 51.3 | 63.8 | 52.2 (−0.9) |
| w/o pre-decision | 41.5 | 50.4 | 65.6 | 52.5 (−0.6) |
| w/o pre-action | 31.5 | 50.7 | 65.4 | 49.2 (−3.9) |
| w/o post-feedback | 41.7 | 41.9 | 65.8 | 49.8 (−3.3) |
| No intervention | 31.8 | 40.7 | 60.1 | 44.2 |

The full patch reaches 53.1% average success, 8.9 pp above no intervention.
Removing **pre-action mediation** or **post-feedback recovery** causes the largest
drops (3.9 and 3.3 pp), while removing episode initialization or pre-decision
costs only 0.9 and 0.6 pp.

The dominant position is environment-dependent: pre-action mediation matters most
on WebShop (41.6 → 31.5), whereas post-feedback recovery matters most on ALFWorld
(52.1 → 41.9). Because one patch can coordinate several positions — and the
evaluated WebShop patches contain only pre-action edits — these effects are
conditional and should **not** be summed into a universal importance ranking. The
practical implication is the opposite: which position matters is itself something
the editor must decide per environment, which a fixed strategy cannot do.

## Data splits

Task-level splits, fixed **before** any trajectory collection or patch
generation. SFT train and RL train are disjoint task partitions; validation is
used only for checkpoint selection and test only for final evaluation. These
count distinct benchmark tasks, not trajectories, failure packets, generated
patches, or optimizer samples.

| Benchmark | SFT train | RL train | Train total | Validation | Test |
|---|---|---|---|---|---|
| WebShop | 5,290 | 5,190 | 10,480 | 100 | 500 |
| ALFWorld | 1,380 | 1,280 | 2,660 | 99 | 500 |
| DBBench | 2,401 | 2,302 | 4,703 | 100 | 300 |
| **Total** | **9,071** | **8,772** | **17,843** | **299** | **1,300** |

- **WebShop.** Task indices 0–499 are the fixed test set. The remaining pool is
  split at the task-batch level into SFT and RL partitions with seed 20260603;
  100 RL-side tasks are reserved for validation with seed 20260623. Goal seed 233
  is used throughout baseline and patched execution.
- **ALFWorld.** Stratified by the six task families. The 500-task test set is all
  109 tasks from the official `new_std` split plus a stratified 391-task sample
  from `train_valid`. With seed 20260614 the rest splits into 1,380 SFT tasks and
  an RL-side partition, of which 99 are reserved for validation.
- **DBBench.** The 4,803 available training tasks are shuffled with seed 20260625
  and split 2,401 / 2,402; 100 RL-side tasks are reserved for validation. The
  separate 300-task standard test set is used only for evaluation.

Teacher filtering, failure-packet construction, benchmark balancing, and
multi-candidate sampling all operate *within* these partitions, so their record
counts are training-accounting quantities rather than additional task splits.

### Training record counts

Derived from the splits above. These are the quantities the reported runs
actually consumed; the hyperparameters that go with them live in
[`configs/sft/qwen35_9b_engineer_coldstart.yaml`](../configs/sft/qwen35_9b_engineer_coldstart.yaml),
[`configs/sft/qwen35_9b_agent_sft.example.yaml`](../configs/sft/qwen35_9b_agent_sft.example.yaml),
and [`scripts/train_engineer_rl.sh`](../scripts/train_engineer_rl.sh).

| Stage | Records | Composition |
|---|---|---|
| Engineer cold-start SFT | 877 editing examples | 381 WebShop / 248 ALFWorld / 248 DBBench |
| Engineer online GRPO | ~1,500 failure packets | disjoint task split from SFT |
| Direct target-agent SFT | 2,515 trajectories | 901 WebShop / 774 ALFWorld / 840 DBBench |

Cold-start candidates come from a **GPT-5.5 teacher** and are retained only if
they are executable, complete the same-batch rerun, and achieve a non-negative
task-reward change. Agent-SFT trajectories are successful no-intervention
episodes, deduplicated by benchmark and canonical task identity.

All three stages run on a single node with **8× NVIDIA H800** GPUs.

## See also

- [CASE_STUDIES.md](CASE_STUDIES.md) — what the generated patches actually do
- [METHOD.md](METHOD.md) — the failure → edit → rerun loop and reward definition
- [PATCH_FORMAT.md](PATCH_FORMAT.md) — the executable patch contract
