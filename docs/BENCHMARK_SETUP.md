# Benchmark Setup

Benchmark assets are intentionally not vendored. The source tree preserves the
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

## ALFWorld

Install ALFWorld and its game assets in a dedicated environment:

```bash
export ALFWORLD_WORKER_PYTHON=/path/to/alfworld/bin/python
export ALFWORLD_SPLIT=harness_r1_eval
```

Create the deterministic split file under AgentBench's ALFWorld data directory.
The public source includes split builders but not generated split contents.

## DBBench

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

