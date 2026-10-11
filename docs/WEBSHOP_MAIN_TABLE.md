# Reproducing the WebShop Main Table

The original identity-clean E0 **iter143** run is archived under
[`reproduction/webshop_e0/`](../reproduction/webshop_e0/). It used 500 tasks,
indices 0 through 499, in 50 consecutive batches of 10 without an additional
evaluation-time shuffle. These indices refer to the seeded WebShop goal list.

| Metric | No-harness baseline | E0 patched rerun |
|---|---:|---:|
| Success | 156/500 (31.2%) | 211/500 (42.2%) |
| Mean reward | 0.6595866666666667 | 0.69923 |
| Valid patches / completed patched batches | - | 50 / 50 |

All 500 archived baseline/patched task manifests match. These are archived
model results, not newly collected model rollouts. The environment identity
checks described below were performed independently on 2026-10-11.

## Initialization and Assets

Use WebShop revision `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` with the updated
[AgentBench dependency patch](../code/life-harness/AgentBench/src/server/tasks/webshop/webshop.patch).
The existing [Dockerfile](../code/life-harness/AgentBench/src/server/tasks/webshop/Dockerfile)
applies that patch during image construction; rebuild old worker images to
include the fix. For an existing dependency built with the original release's
patch, apply the incremental fix inside its WebShop source directory:

```bash
cd /path/to/installed/WebShop
git apply --check /path/to/Harness-R1/reproduction/webshop_e0/fixes/webshop_goal_seed.patch
git apply /path/to/Harness-R1/reproduction/webshop_e0/fixes/webshop_goal_seed.patch
```

Do not apply the incremental fix to an already corrected source. The original
release omitted the paper environment's additional `SimServer` seeding:

1. Save the caller's Python RNG state, then seed **233 before** loading products,
   generating product prices, and generating goal price limits.
2. Independently seed **233 again before shuffling goals**.
3. Restore the saved RNG state after the shared simulator has been constructed.

The upstream code seeds only the shuffle. Passing `goal_seed=233` through its
`WebAgentTextEnv(**kwargs)` does not itself seed product-price generation.

With our retained assets and real search index, three independent processes
produced the following checks:

| Initialization | Goals | Archived task manifests matched |
|---|---:|---:|
| Retained local dependency with the paper fix | 12087 | 500/500 |
| Original release dependency plus outer `random.seed(233)` | 12087 | 500/500 |
| Original release dependency plus the incremental fix | 12087 | 500/500 |

Thus an outer `random.seed(233)` immediately before environment construction
reproduces the archived identities with these same assets and seed. The full
dependency fix also restores the caller's random state and handles nondefault
seeds. It does not change the task reward or hook implementation.

The canonical `product_prices_sha256` for 1,181,430 unique products is:

```text
9f44958915599c0b494c1b6b94c8e57d2011058f779e49deb20a025716817438
```

Use the actual worker Python, installed WebShop module, and Java 11:

```bash
export JAVA_HOME=/path/to/jdk11
export PYTHONPATH=/path/to/installed/WebShop
/path/to/webshop-worker/bin/python scripts/verify_webshop_environment.py \
  --output /new/path/environment_check_1.json
/path/to/webshop-worker/bin/python scripts/verify_webshop_environment.py \
  --output /new/path/environment_check_2.json
```

Both should report `matching_tasks: 500`. `--outer-seed` tests the workaround
against an old dependency. A matching price hash alone is insufficient: the
script also checks instruction, goal, and combined task hashes.

The retained worker environment measured for these checks is Python 3.9.25,
Temurin Java 11.0.31, pyserini 0.17.1, faiss-cpu 1.7.4, NumPy 1.26.4,
spaCy 3.7.5, en_core_web_lg 3.7.1, and torch 1.11.0. This is a measurement
of the retained environment, not an immutable July package lockfile.
The [verification records](../reproduction/webshop_e0/verification/) contain
the reports and hashes of the product data and Lucene index files.

To fingerprint another installation without importing its worker dependencies:

```bash
python scripts/hash_webshop_assets.py --webshop /path/to/installed/WebShop \
  --output /new/path/asset_hashes.json
```

Rebuilt Lucene indexes may differ at the byte level even when semantically
equivalent. Matching task manifests verifies goals and prices, but does not
establish matching search results, product descriptions, or model behavior.
If the host exports torch library paths from another Python environment,
remove those entries from `LD_LIBRARY_PATH` for the worker to avoid ABI errors.

## Exact Engineer Input and Serving

The main-table schema was **`webshop_code_only`**, with the original materialized
SFT-aligned prefix. `webshop_life_multihook_v1` is a different model-facing
prompt. Use the [original input JSONL](../reproduction/webshop_e0/data/engineer_inputs.jsonl);
changing only another file's metadata label does not align its messages.

The original prefix allows all four hook positions. All 50 archived patches
generated only `on_before_action`; no later hook masking was applied.

| Setting | Engineer | Target |
|---|---|---|
| Model | Identity-clean E0 iter143 | Vanilla Qwen3.5-9B |
| Temperature | 0 | 0 |
| Top-p | 1 | Request default |
| Max output tokens | 12288 | 4096 |
| Thinking | Enabled | Disabled |
| Assistant prefill | `<think>\n`, with an actual newline | None |
| Tool choice | No tool schema sent | `auto` |
| Concurrency | 8 batches | 10 tasks within each batch |
| Horizon | - | 20 rounds |

The engineer's archived startup log records SGLang, the base Qwen3.5-9B
tokenizer and its default chat template, TP1/DP1, BF16, context length 32768,
Triton attention, PyTorch sampling, and no separate reasoning parser. These
settings do not identify an immutable package-version lockfile or target-model
Hub revision.

The evaluator appends the assistant prefill and sends
`continue_final_message=true`. It did not send an explicit stop string;
normal server/tokenizer termination applied. All 50 archived responses have
`finish_reason=stop`. There were up to 3 retries for request failures,
not best-of-N selection by rerun reward. Invalid patches normally count as
no-patch; this run had 50 valid patches.

For vLLM versions requiring it, `add_generation_prompt=false` is consistent
with continuing the assistant prefill. The dedicated script sends this field
when `ENGINEER_BACKEND=vllm`. Different serving stacks can still generate
different text at temperature zero.

## Replay and Regenerate

First verify task identity, then replay the original patches, then regenerate
patches from the original engineer inputs. This separates target/runtime
differences from engineer-generation differences.

```bash
export EVAL_PYTHON=/path/to/evaluator/bin/python
export AGENTBENCH_PYTHON=/path/to/controller/bin/python
export WEBSHOP_WORKER_PYTHON=/path/to/webshop-worker/bin/python
export TARGET_BASE_URL=http://127.0.0.1:8000/v1
export TARGET_MODEL=Qwen3.5-9B
export CONTROLLER_PORT_BASE=30000
export WORKER_PORT_BASE=31000
export OUTPUT_ROOT=/new/path/e0_patch_replay
bash scripts/reproduce_webshop_main_table.sh replay
```

Use `replay --limit 1` with a separate output directory for a smoke check.
The replay mode needs no engineer endpoint. To regenerate patches:

```bash
export ENGINEER_BASE_URL=http://127.0.0.1:8001/v1
export ENGINEER_MODEL=harness-r1
export ENGINEER_BACKEND=vllm
export OUTPUT_ROOT=/new/path/e0_patch_regeneration
bash scripts/reproduce_webshop_main_table.sh generate
```

These diagnostic runs use archived baseline rewards. A causal estimate on a
different serving stack additionally requires a new no-harness rollout on
that same stack and the identical tasks.

The original baseline used one controller for its 50 worker batches. Patched
evaluation assigned a separate controller port to each batch and launched a
fresh worker process. The shared simulator holds separate per-episode sessions.
This was a local worker-Python setup. The dedicated script expects unused
controller and worker port ranges. A persistent shared controller must isolate
task registrations across repeated runs; duplicate names can retain stale task
configurations.

The [full artifact bundle](../reproduction/webshop_e0/full-artifacts.tar.gz?raw=true)
also provides both archived rollout sets and a verified goals/prices snapshot.
The snapshot is for simulator verification only and must not be added to model
inputs. The full product catalog and search index remain separate benchmark
assets.
