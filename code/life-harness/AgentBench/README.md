# AgentBench Runtime

This subproject provides the AgentBench task environments and the Harness-R1
code-hook runtime. It is kept separate because it uses Docker-based task workers
and a Python 3.9 dependency stack.

Supported tasks: ALFWorld, DBBench, WebShop.

## Installation

```bash
cd AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
```

Docker is required for the task workers.

## Configure the Agent

Configure the OpenAI-compatible agent endpoint in
`configs/agents/api_agents.yaml`. The defaults target self-hosted servers and
use `Bearer EMPTY`; replace the endpoint and authorization value locally and do
not commit private keys or service URLs.

Check that a profile resolves:

```bash
python -m src.client.agent_test --config configs/agents/api_agents.yaml --agent qwen35-local
```

## Build Images and Start Task Services

```bash
docker pull mysql:8
```

Start Redis, the controller, and the worker for the benchmark you are running:

```bash
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller alfworld-std
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller dbbench-std
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller webshop-std
```

Restart the corresponding services after editing task configuration. WebShop can
take several minutes to become ready after Docker reports the container started.

## Run a Plain Evaluation

```bash
python -m src.assigner --config configs/assignments/alfworld.yaml
python -m src.assigner --config configs/assignments/dbbench.yaml
python -m src.assigner --config configs/assignments/webshop.yaml
```

Configuration lives in `configs/agents/api_agents.yaml` (endpoint, model,
sampling), `configs/tasks/*.yaml` (split, max steps/rounds, harness switches),
and `configs/assignments/*.yaml` (profile, concurrency, trials, output).

Harness-R1 does not use these assignment files directly. It drives batched
rollouts through `scripts/harness_r1_batch_debug.py`, which generates task,
agent, and assignment configs per batch.

## Harness Switches

`configs/tasks/*.yaml` carries the legacy Life-Harness switches `h2`, `h3`,
`h4`, and `h5`, gated by a master `enabled` flag (top-level or nested under
`harness`). Harness-R1 experiments keep all of them disabled: the engineer's
generated code hooks are the only active intervention.

## Default Evaluation Settings

| Benchmark | Agent sampling | Agent max tokens | Max step / rounds |
| --- | --- | ---: | ---: |
| ALFWorld | temperature = 0.0 | 4096 | 50 |
| DBBench | temperature = 0.0 | 4096 | 15 |
| WebShop | temperature = 0.0 | 4096 | 20 |
