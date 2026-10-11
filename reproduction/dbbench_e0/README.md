# DBBench E0 Main-Table Reference

Original **identity-clean E0 iter143** result: **183/300 -> 196/300**
(61.0% -> 65.33%). All **30 engineer inputs, raw outputs, and valid patches**
are included. All 30 patched batches completed successfully.

| Files | Contents |
|---|---|
| [data/engineer_inputs.jsonl](data/engineer_inputs.jsonl) | Materialized system/user messages with the SFT-aligned prefix; one local path redacted per input |
| [data/engineer_prompt_prefix.json](data/engineer_prompt_prefix.json) | Exact shared model-facing prefix |
| `outputs/batch_NNN/raw_response.txt` | All 30 original engineer outputs |
| `outputs/batch_NNN/patch.json` | Original evaluated patches |
| `outputs/batch_NNN/result.json` | Archived batch outcome, finish reason, and token usage |
| [data/tasks.jsonl](data/tasks.jsonl) | All 300 task-record hashes and paired rewards |
| [data/evaluation_settings.json](data/evaluation_settings.json) | Key inference and evidence settings |
| [data/provenance.json](data/provenance.json) | Original-file hashes and export details |

The exact main-table test file is
[`data/dbbench/standard.jsonl`](../../data/dbbench/standard.jsonl): **300 tasks in
file order, 30 consecutive batches of 10, no shuffle**. Its SHA256 is:

```text
cf8dec6fb8051f46e6171136824aaa8b8707c3affbb3185c0bad2b39bc6b71e2
```

Each prompt retains the original schema and failure traces, with its internal
`Rollout result dir` path replaced by `[PATH]`. Both original and exported
prompt hashes are recorded. Raw outputs, patches, and test-file bytes are
unchanged. Metadata omits private paths and adds schema/protocol labels.
Reward maps use global task indices 0-299. The test records contain evaluator
labels and database definitions; they are loaded by the task runtime, not
appended to engineer messages.

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

Schema: `dbbench_life_multihook_v1`; protocol: `prefill_think_patch`.
The original E0 reruns used **manual MySQL**, 20 rounds, and target
`tool_choice=auto`. The engineer used SGLang with no explicit stop string and
no `reasoning_effort` override. Use the provided inputs to preserve the actual
prompt rather than reconstructing it from schema names.

The dataset builder's missing prompt-reference fields and `eval_dbbench.sh`'s
extra closing brace in exported JSON kwargs are fixed. The generic wrapper
also accepts `ENGINEER_EXTRA_BODY`, for example
`'{"add_generation_prompt":false}'` for vLLM assistant continuation.

## Run

With a prepared MySQL 8 instance and the DBBench worker environment installed:

```bash
export EVAL_PYTHON=/path/to/evaluator/bin/python
export AGENTBENCH_PYTHON=/path/to/controller/bin/python
export DBBENCH_WORKER_PYTHON=/path/to/dbbench-worker/bin/python
export DBBENCH_MYSQL_HOST=127.0.0.1:3306
export TARGET_BASE_URL=http://127.0.0.1:8000/v1
export TARGET_MODEL=Qwen3.5-9B
export CONTROLLER_PORT_BASE=36000
export WORKER_PORT_BASE=37000
export OUTPUT_ROOT=/new/path/dbbench_patch_replay
bash scripts/reproduce_dbbench_main_table.sh replay
```

To regenerate patches from the original inputs, use a new output directory:

```bash
export ENGINEER_BASE_URL=http://127.0.0.1:8001/v1
export ENGINEER_MODEL=harness-r1
export ENGINEER_BACKEND=vllm
export OUTPUT_ROOT=/new/path/dbbench_patch_regeneration
bash scripts/reproduce_dbbench_main_table.sh generate
```

The vLLM mode sends `add_generation_prompt=false`. Add `--limit 1` for a smoke
check or `--generate-only` for generation without reruns. The script validates
the test-file checksum and refuses to overwrite existing output.

This entry point follows the paper's manual-MySQL, separate-controller-port
rerun setup. Use unused ports; it does not configure a rootless Docker runtime
or a persistent shared controller. These runs compare against archived baseline
rewards; a causal estimate on another serving stack requires its matching
no-harness baseline. Verify reference files with `sha256sum -c SHA256SUMS` from
this directory.
