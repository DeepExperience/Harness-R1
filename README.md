# Harness-R1

Harness-R1 trains a *harness engineer*: a model that reads a batch of failed
agent trajectories and writes a reusable runtime patch. The patch is validated,
compiled into sandboxed hooks, and scored by rerunning a frozen target agent on
exactly the same tasks.

```text
frozen target rollout
  -> batch failure packet
  -> harness engineer
  -> <think>...</think><patch>...</patch>
  -> parse, validate, and sandbox hooks
  -> rerun the same target on the same task identities
  -> reward = patched metric - baseline metric
```

The only patch action is `add_code_hook`. A patch may define:

| Hook | Role |
|---|---|
| `on_init` | Initialize notebook state and inject reusable skills/tool hints. |
| `on_post_step` | Update notebook state after an environment step. |
| `make_pre_hint` | Emit deduplicated soft guidance before the next action. |
| `on_before_action` | Narrowly allow, block, rewrite, or force an action when supported. |

See [docs/METHOD.md](docs/METHOD.md) and
[docs/PATCH_FORMAT.md](docs/PATCH_FORMAT.md).

## Layout

```text
code/Relax/                      trimmed RL framework snapshot (upstream: redai-infra/Relax)
code/Relax/examples/harness_r1/  rewards, evaluators, dataset builders
code/life-harness/AgentBench/    task runtimes and code-hook execution
configs/                         SFT, RL, and endpoint configuration
scripts/                         launch and check wrappers
tests/                           protocol and sandbox unit tests
```

Neither benchmark assets, model weights, nor training data are bundled. See
[docs/BENCHMARK_SETUP.md](docs/BENCHMARK_SETUP.md) for how to install the
WebShop, ALFWorld, and DBBench environments.

## Setup

```bash
python -m venv code/life-harness/AgentBench/.venv
code/life-harness/AgentBench/.venv/bin/pip install -r \
  code/life-harness/AgentBench/requirements.txt

cp configs/eval/endpoints.env.example .env
# edit .env: engineer endpoint, frozen target endpoint, benchmark interpreters
set -a; source .env; set +a
```

Source-only checks (no endpoints or benchmark assets needed):

```bash
python scripts/check_release.py
```

## Evaluation

Evaluation is fixed-batch run-patch-rerun: read a prompt plus immutable
baseline metadata, generate one patch, validate it, rerun the frozen target on
the same tasks, and report patched minus baseline.

The target server must return structured `message.tool_calls`; XML-looking tool
text inside `message.content` is not equivalent and silently zeroes rewards.
Probe before a long run:

```bash
python scripts/probe_openai_tool_calls.py \
  --base-url "$TARGET_BASE_URL" --model "$TARGET_MODEL"
```

Then run one of the wrappers with an input JSONL and an output directory:

```bash
bash scripts/eval_webshop.sh  /path/to/webshop_test.jsonl  outputs/webshop
bash scripts/eval_alfworld.sh /path/to/alfworld_test.jsonl outputs/alfworld
bash scripts/eval_dbbench.sh  /path/to/dbbench_test.jsonl  outputs/dbbench
```

Benchmark-specific environment variables (ALFWorld split and interpreter,
DBBench MySQL, and so on) are documented in
[docs/BENCHMARK_SETUP.md](docs/BENCHMARK_SETUP.md).

Patch generation and target rerun can be split, which is the correct protocol
for cross-target generalization:

```bash
GENERATE_ONLY=1 bash scripts/eval_webshop.sh INPUT OUT_GENERATED
PATCH_SOURCE_ROOT=OUT_GENERATED bash scripts/eval_webshop.sh INPUT OUT_RERUN
```

Valid patches whose rerun hit an infrastructure failure can be retried with
`RESUME=1 RESUME_RERUN_EVAL_FAILED=1`. An invalid patch is no-patch and scores
zero; an environment failure is a missing evaluation and must be reported
separately, never retried until it turns positive.

WebShop results are reportable only when `webshop_goal_seed=233` and the strict
`webshop_batch_identity_v1` task manifests agree between baseline and patched
rerun. Matching integer task indices are not proof of a paired comparison.

## Training

Three model roles: a **target agent** that runs benchmark tasks and stays
frozen within a stage, the **harness engineer** being trained, and the frozen
**reference policy** used by GRPO. The reward is not a learned judge — every
valid patch is compiled and scored by an actual same-batch rerun.

### 1. Cold-start SFT

Each SFT row has ordered `system`, `user`, `assistant` messages; the chat
template supplies the opening assistant `<think>\n` prefill and the target
continues that block and ends with a `<patch>` JSON object. Point
`configs/sft/qwen35_9b_engineer_coldstart.yaml` at your base model, dataset,
and output path, then run:

```bash
bash scripts/train_engineer_sft.sh
```

The reference setup is full SFT, 2 epochs, learning rate `1e-5`, cutoff length
32768, bfloat16, effective global batch size 24.

### 2. Build online-RL prompts

For each target stage: freeze the target checkpoint and serving options, run
no-harness trajectories on the training split, build batch failure packets, and
group records without mixing benchmarks inside a rollout group. Validation and
test task identities stay out of SFT and RL.

```bash
python code/Relax/examples/harness_r1/build_webshop_dataset.py --help
python code/Relax/examples/harness_r1/build_alfworld_dataset.py --help
python code/Relax/examples/harness_r1/build_dbbench_rl_dataset.py --help
python code/Relax/examples/harness_r1/build_grouped_mixed_rl_dataset.py --help
```

An RL row carries the prompt plus the immutable baseline metadata used by the
reward:

```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "label": "",
  "metadata": {
    "benchmark": "webshop",
    "batch_tag": "b0001",
    "start": 10, "end": 20, "batch_size": 10,
    "baseline_pass": 2,
    "baseline_rewards": {"10": 0.0},
    "target_model": "Qwen3.5-9B",
    "target_agent_name": "qwen35-9b-nothink"
  }
}
```

Never reuse baseline rewards produced by a different target model or serving
protocol. SFT, RL, and evaluation must also agree on the exact system prompt,
static user prefix and schema, `schema_style`, and the `prefill_think_patch`
response protocol; changing the static protocol causes train/eval drift and
sharply reduces format validity.

### 3. Online GRPO

Edit `configs/rl/mixed_codepatch.yaml` to point at the AgentBench and
benchmark interpreters, benchmark assets, a writable reward cache, and the
frozen target endpoint that produced the baseline JSONL. Then:

```bash
export QWEN35_HF=/path/to/engineer-sft-checkpoint
export HARNESS_R1_DATA=/path/to/grouped_rl.jsonl
export NUM_ROLLOUT=174
export HARNESS_R1_CONFIG="$PWD/configs/rl/mixed_codepatch.yaml"
export SAVE_PATH=/path/to/checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_engineer_rl.sh
```

Reference defaults: rollout batch size 4, 8 samples per prompt, global batch
size 32, 4 iterations per train update, rollout temperature/top-p 0.7/0.95, max
prompt/response 28672/12288, learning rate `1e-6`, no validity bonus, reward =
delta average reward. Use `ROLLOUT_SHUFFLE=0` for pre-grouped mixed data.

`reward_mixed_codepatch.py` dispatches on `metadata["benchmark"]` to the
WebShop, ALFWorld, or DBBench reward.

## License

Harness-R1 code is released under Apache-2.0. Relax, AgentBench, and the
benchmark environments retain their original notices and licenses; see
[NOTICE](NOTICE).
