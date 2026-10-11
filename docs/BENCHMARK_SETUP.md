# Benchmark Setup

Large environment assets are installed separately. Task manifests and the
DBBench main-table test records are included. The source tree preserves the
expected AgentBench layout:

```text
code/life-harness/AgentBench/data/
  alfworld/
  webshop/
  dbbench/
```

Follow the upstream AgentBench, WebShop, and ALFWorld licenses and installation
instructions.

## WebShop

For the paper's E0 main-table result, follow
[the dedicated reproduction guide](WEBSHOP_MAIN_TABLE.md), including the
archived inputs, original patches, and task-identity verification.

WebShop uses `pyserini`/`jnius` and requires a JDK with `javac`. Set:

```bash
export HARNESS_R1_JAVA_HOME=/path/to/jdk
export JAVA_HOME="$HARNESS_R1_JAVA_HOME"
```

Use a dedicated worker environment if its search dependencies conflict with
the AgentBench controller:

```bash
export WEBSHOP_WORKER_PYTHON=/path/to/webshop/bin/python
```

All new baseline and patched runs must use `webshop_goal_seed=233` and strict
task manifests.

The WebShop dependency must also seed product and goal generation **before**
the final goal shuffle. The updated
`code/life-harness/AgentBench/src/server/tasks/webshop/webshop.patch` includes
this fix for upstream revision `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`.
It seeds Python's RNG with 233 before loading products/generating goals, seeds
again before shuffling goals, and restores the caller's RNG state. Earlier
copies of this patch omitted the first seed; rebuild existing worker images
with the updated dependency patch. Passing `goal_seed=233` to an unpatched
upstream environment does not fix product-price randomness.

The WebShop main-table E0 run used `webshop_code_only` with SFT-aligned
materialized prompts, `prefill_think_patch`, and assistant prefill `<think>\n`.
`webshop_life_multihook_v1` is a different model-facing prompt and should not be
substituted when reproducing that row. The paper task manifests have the full
product-price-map hash
`9f44958915599c0b494c1b6b94c8e57d2011058f779e49deb20a025716817438`.
Also compare individual instruction and goal hashes; matching this one hash
does not establish that the search index or serving stack is identical.

## ALFWorld

For the main-table E0 input/output artifacts and runnable commands, see
[`reproduction/alfworld_e0/`](../reproduction/alfworld_e0/).

The 500-task Harness-R1 test split is available in
[`data/alfworld/test_set.json`](../data/alfworld/test_set.json). It preserves the
original task order and contains 109 tasks from `new_std` plus 391 stratified
samples from `train_valid`, selected with seed `20260609`. Each entry is a game
path relative to the ALFWorld data directory; the game assets are installed
separately.

Install ALFWorld and its game assets in a dedicated environment, then copy the
split into AgentBench's ALFWorld data directory:

```bash
mkdir -p code/life-harness/AgentBench/data/alfworld
cp data/alfworld/test_set.json code/life-harness/AgentBench/data/alfworld/test_set.json
export ALFWORLD_WORKER_PYTHON=/path/to/alfworld/bin/python
export ALFWORLD_SPLIT=test_set
```

## DBBench

The [DBBench E0 reference](../reproduction/dbbench_e0/) includes the original
inputs/outputs, exact [300-task test file](../data/dbbench/standard.jsonl), and
main-table replay/generation commands. It uses `standard.jsonl` in consecutive
batches of 10 and the manual MySQL driver.

DBBench supports:

- `DBBENCH_ENV_DRIVER=docker`, using the AgentBench Docker controller;
- `DBBENCH_ENV_DRIVER=manual`, using a pre-provisioned MySQL 8 instance.

Manual mode avoids per-task container setup and is recommended on nested
container platforms:

```bash
export DBBENCH_ENV_DRIVER=manual
export DBBENCH_MYSQL_HOST=127.0.0.1
export DBBENCH_DATA_FILE=/path/to/db_out_new.jsonl
```

Use isolated schemas or a reset-capable controller when running concurrent
workers. Do not point evaluation at a shared production database.
