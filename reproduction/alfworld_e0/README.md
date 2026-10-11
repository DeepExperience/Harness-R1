# ALFWorld E0 Main-Table Reference

Original **identity-clean E0 iter143** evaluation: **203/500 -> 266/500**
(40.6% -> 53.2%). All **50 engineer inputs and raw outputs** are included.
There are **43 valid patches**, all evaluated successfully; the other 7 outputs
hit the 12288-token limit and were counted as no-patch, using baseline rewards.
The unsuccessful outputs are preserved too.

| Files | Contents |
|---|---|
| [data/engineer_inputs.jsonl](data/engineer_inputs.jsonl) | Original materialized system/user messages; no prompt reconstruction needed |
| [data/engineer_prompt_prefix.json](data/engineer_prompt_prefix.json) | Exact shared model-facing prefix |
| `outputs/<batch_tag>/raw_response.txt` | All 50 original engineer outputs |
| `outputs/<batch_tag>/patch.json` | The 43 valid patches; absent for the 7 invalid outputs |
| `outputs/<batch_tag>/result.json` | Original validity, outcome, finish reason, and token usage |
| [data/tasks.jsonl](data/tasks.jsonl) | Task index, game path, and paired effective rewards for all 500 tasks |
| [data/evaluation_settings.json](data/evaluation_settings.json) | Key settings, including trace-packet limits |
| [data/provenance.json](data/provenance.json) | Original-file hashes and export changes |

Model-facing messages and raw outputs are unchanged. Metadata omits private
source paths and renames the historical split to `test_set`. The published
[`data/alfworld/test_set.json`](../../data/alfworld/test_set.json) is byte-identical
to the original split, including its 500-task order. Per-batch reward maps use
local indices 0-9; `tasks.jsonl` also supplies the global indices 0-499.

## Key Settings

| Setting | Engineer | Target |
|---|---|---|
| Model | Identity-clean E0 iter143 | Vanilla Qwen3.5-9B |
| Temperature | 0 | 0 |
| Top-p | 1 | Request default |
| Max tokens | 12288 | 4096 |
| Thinking | Enabled | Disabled |
| Prefill | `<think>\n` with an actual newline | None |
| Concurrency | 8 batches | 10 tasks per batch |

Schema: `alfworld_life_multihook_v1`; protocol: `prefill_think_patch`.
50 steps per task; 50 consecutive batches of 10. The engineer used SGLang with
the base Qwen3.5-9B tokenizer/default chat template, no explicit stop string,
and up to 3 retries for request failures. Invalid outputs were not regenerated
with larger budgets or selected by rerun reward.

Packet settings: `max_traces=4`, `strategy=lowest_reward`, `max_obs_chars=350`,
`max_assistant_chars=200`, `max_approx_tokens=12000`. Use the provided messages
to preserve the historical packet and prompt exactly.

## Run

With ALFWorld 0.2.2 assets and the worker environment installed:

```bash
mkdir -p code/life-harness/AgentBench/data/alfworld
cp data/alfworld/test_set.json code/life-harness/AgentBench/data/alfworld/test_set.json
export EVAL_PYTHON=/path/to/evaluator/bin/python
export AGENTBENCH_PYTHON=/path/to/controller/bin/python
export ALFWORLD_WORKER_PYTHON=/path/to/alfworld-worker/bin/python
export TARGET_BASE_URL=http://127.0.0.1:8000/v1
export TARGET_MODEL=Qwen3.5-9B
export CONTROLLER_PORT_BASE=33000
export WORKER_PORT_BASE=34000
export OUTPUT_ROOT=/new/path/alfworld_patch_replay
bash scripts/reproduce_alfworld_main_table.sh replay
```

Replay uses the original 43 patches and preserves the 7 no-patch fallbacks.
For new engineer generation, use a new output directory:

```bash
export ENGINEER_BASE_URL=http://127.0.0.1:8001/v1
export ENGINEER_MODEL=harness-r1
export ENGINEER_BACKEND=vllm
export OUTPUT_ROOT=/new/path/alfworld_patch_regeneration
bash scripts/reproduce_alfworld_main_table.sh generate
```

The vLLM mode sends `add_generation_prompt=false` with the assistant prefill.
Use unused controller/worker ports; the script checks the installed split and
refuses to overwrite an existing output directory. Add `--limit 1` for a smoke
check, or `--generate-only` to generate without target reruns.

These runs compare with archived baseline rewards. For a new causal estimate
on another target-serving stack, also collect its matching no-harness baseline.
Verify the reference files with `sha256sum -c SHA256SUMS` from this directory.
