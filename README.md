<div align="center">

# Harness-R1

**Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/License-Apache%202.0-D22128.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-coming%20soon-B31B1B.svg)](#citation)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-Harness--R1-FFD21E.svg)](https://huggingface.co/ShaoShuai0605/Harness-R1)
[![Engineer](https://img.shields.io/badge/Engineer-Qwen3.5--9B-6E56CF.svg)](#training)
[![Training](https://img.shields.io/badge/Training-SFT%20%2B%20online%20GRPO-0B7285.svg)](#3-online-grpo)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-WebShop%20%7C%20ALFWorld%20%7C%20DBBench-2F6F4E.svg)](docs/BENCHMARK_SETUP.md)

[Demo](#-demo) ·
[Overview](#overview) ·
[Highlights](#highlights) ·
[Results](#results) ·
[Models](#model-checkpoints) ·
[Layout](#repository-layout) ·
[Installation](#installation) ·
[Evaluation](#evaluation) ·
[Training](#training) ·
[Citation](#citation)

Shuai Shao<sup>1,2‡\*</sup>, Kangning Zhang<sup>1,2‡\*</sup>, Qingyao Li<sup>1,2\*</sup>, Shijian Wang<sup>3</sup>, Hao Wang<sup>2</sup>,<br>
Wenxiang Jiao<sup>2✉</sup>, Yuan Lu<sup>2✉</sup>, Yi Guo<sup>2</sup>, Weiwen Liu<sup>1✉</sup>, Weinan Zhang<sup>1✉</sup>

<sup>1</sup>Shanghai Jiao Tong University · <sup>2</sup>Xiaohongshu Inc. · <sup>3</sup>Southeast University<br>
<sub><sup>‡</sup>Equal contribution · <sup>\*</sup>Work done during internship at Xiaohongshu Inc. · <sup>✉</sup>Corresponding authors</sub>

</div>

## 🎬 Demo

<div align="center">
  <img src="assets/demo.gif" alt="Harness-R1 demo" width="88%">
</div>

```bash
python demo/run_demo.py
```

## Overview

Harness-R1 trains a *harness engineer*: a model that reads a batch of failed
agent trajectories and writes a reusable runtime patch. The patch is validated,
compiled into sandboxed hooks, and scored by rerunning a frozen target agent on
exactly the same tasks.

<div align="center">
  <img src="assets/framework.png" alt="Harness-R1 framework" width="100%">
</div>

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

| Hook | Lifecycle position | Role |
|---|---|---|
| `on_init` | Episode initialization | Initialize notebook state and inject reusable skills/tool hints. |
| `make_pre_hint` | Pre-decision | Emit deduplicated soft guidance before the next action. |
| `on_before_action` | Pre-action | Narrowly allow, block, rewrite, or force an action when supported. |
| `on_post_step` | Post-feedback | Update notebook state after an environment step. |

A hook body is ordinary Python; what the runtime honors is its return value. The
patch contract lives in
[`harness_r1_patch.py`](code/life-harness/AgentBench/scripts/harness_r1_patch.py)
(`schema_prompt`, `normalize_patch`) and the sandbox rules in
[`code_runner.py`](code/life-harness/AgentBench/src/server/harness/code_runner.py);
[`examples/webshop_patch.json`](examples/webshop_patch.json) is a complete patch.

## Highlights

- **Rewards come from real reruns, not a judge** — every valid patch is compiled,
  installed, and scored by rerunning the frozen target on the same task identities.
- **Lifecycle-wide editing** — one patch can coordinate four intervention points
  around the frozen policy.
- **A trained 9B engineer beats larger fixed editors** — 53.6% vs 48.8% for the
  strongest frontier editor (GLM-5.2).
- **Gains survive target fine-tuning** — 44.3 → 53.6 vanilla, 59.2 → 64.2 after
  agent SFT.
- **Transfers without retuning** — improves all 20 unseen targets (+7.06 pp) and
  1,270 held-out tasks.
- **Patches run sandboxed** — AST validation rejects imports, I/O, dynamic
  evaluation, and leakage; hook failures degrade to no effect.

## Results

Target: frozen Qwen3.5-9B. `Score` is the mean shaped WebShop reward, `Succ.` is
task success rate, and `Avg.` is the equal-weight average of WebShop Succ.,
ALFWorld All, and DBBench Succ.

| Method | ALFWorld All | WebShop Score | WebShop Succ. | DBBench Succ. | **Avg.** |
|---|---|---|---|---|---|
| Qwen3.5-9B (default harness) | 40.6 | 66.0 | 31.2 | 61.0 | 44.3 |
| ReAct | 43.4 | 66.8 | 37.4 | 61.7 | 47.5 |
| Self-Refine | 39.0 | 61.7 | 29.0 | 57.3 | 41.8 |
| Reflection <sup>‡</sup> | 59.2 | 61.2 | 43.6 | 64.7 | 55.8 |
| Qwen3.5-397B | 41.2 | 66.4 | 32.8 | 63.3 | 45.8 |
| GLM-5.2 | 45.0 | 68.4 | 36.0 | 65.3 | 48.8 |
| Kimi-K2.6 | 41.4 | 63.7 | 31.8 | 62.7 | 45.3 |
| DeepSeek-V4-Pro | 41.0 | 65.1 | 32.4 | 64.3 | 45.9 |
| Gemini-3.5-Flash | 35.4 | 63.2 | 33.6 | 64.0 | 44.3 |
| GPT-5.5 | 43.2 | 61.4 | 36.6 | 64.0 | 47.9 |
| Supervised-only engineer | 39.4 | 67.8 | 38.6 | 61.3 | 46.4 |
| **Harness-R1** | 53.2 | 69.9 | 42.2 | 65.3 | **53.6** |
| Agent SFT | 71.2 | **71.5** | 42.6 | 63.7 | 59.2 |
| **Agent SFT + Harness-R1** | **84.0** | 68.7 | **43.0** | **65.7** | **64.2** |

<sup>‡</sup> Reflection uses a separate two-episode `success@2` protocol and is
not ranked against the single-episode rows.

Harness-R1 raises the frozen target **44.3 → 53.6** (+9.3 pp, and 7.1 pp above
the supervised-only engineer). After direct agent SFT, a target-specific engineer
lifts the stronger actor **59.2 → 64.2** (+5.0 pp).

<div align="center">
  <img src="assets/motivation.png" alt="Matched-baseline reward change by editor" width="82%">
</div>

### Generalization across target agents

Each unseen target supplies its own failure traces and receives newly generated
patches, so this measures transfer of the editing policy, not replay of a fixed
patch. Across **20 unseen targets** the gain is **+7.06 pp** with every
target-level average positive; 56 of 63 target-benchmark combinations improve and
the three regressions are all ≤ 2.0 pp.

<div align="center">
  <img src="assets/generalization.png" alt="Cross-target generalization heatmap" width="52%">
</div>

### Held-out tasks and lifecycle positions

From the **same 10 failures**, each engineer writes one patch applied to all
remaining tasks (1,270 held-out, three seeds). Harness-R1 gains **+8.9 ± 1.5 pp**
and is positive on every seed; both frontier engineers average negative.
Disabling one lifecycle position at a time, pre-action (−3.9) and post-feedback
(−3.3) dominate, and which one dominates is environment-dependent.

<div align="center">
  <img src="assets/heldout.png" alt="Held-out task generalization" width="49%">
  <img src="assets/lifecycle.png" alt="Lifecycle position ablation" width="49%">
</div>

## Model Checkpoints

Both engineers live in **[🤗 ShaoShuai0605/Harness-R1](https://huggingface.co/ShaoShuai0605/Harness-R1)**
(Qwen3.5-9B base, Apache-2.0).

| Subfolder | Paper row | Trained against |
|---|---|---|
| [`harness-r1`](https://huggingface.co/ShaoShuai0605/Harness-R1/tree/main/harness-r1) | Harness-R1 | Frozen vanilla Qwen3.5-9B target |
| [`agent-sft-harness-r1`](https://huggingface.co/ShaoShuai0605/Harness-R1/tree/main/agent-sft-harness-r1) | Agent SFT + Harness-R1 | Frozen agent-SFT Qwen3.5-9B target |

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO, SUBFOLDER = "ShaoShuai0605/Harness-R1", "harness-r1"
tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder=SUBFOLDER, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(REPO, subfolder=SUBFOLDER, trust_remote_code=True)
```

Serve the subfolder behind an OpenAI-compatible endpoint and point
`ENGINEER_BASE_URL` / `ENGINEER_MODEL` at it in
[`configs/eval/endpoints.env.example`](configs/eval/endpoints.env.example). Keep
`ENGINEER_CHAT_TEMPLATE_KWARGS` at `{"enable_thinking":true}` for the
`prefill_think_patch` protocol.

> [!IMPORTANT]
> An engineer is only meaningful against the target it was trained for.
> `agent-sft-harness-r1` reproduces its row only when the frozen target is the
> agent-SFT model. Target agents are not part of this release; build one with
> [`configs/sft/qwen35_9b_agent_sft.example.yaml`](configs/sft/qwen35_9b_agent_sft.example.yaml)
> or serve any target you want to edit.

## Repository Layout

```text
Harness-R1/
├── assets/                            figures from the paper
├── code/
│   ├── Relax/                         trimmed RL framework snapshot (upstream: redai-infra/Relax)
│   │   └── examples/harness_r1/       rewards, evaluators, dataset builders
│   └── life-harness/AgentBench/       task runtimes and code-hook execution
│       ├── scripts/                   patch protocol, trace packets, workers
│       └── src/server/harness/        sandboxed code_runner and per-benchmark hooks
├── configs/
│   ├── eval/endpoints.env.example     engineer/target endpoints and interpreters
│   ├── rl/mixed_codepatch.yaml        online RL reward and runtime configuration
│   └── sft/                           cold-start engineer SFT and agent SFT
├── demo/                              offline demo (no endpoints or benchmark assets)
├── docs/BENCHMARK_SETUP.md            installing the benchmark environments
├── examples/webshop_patch.json        a complete validated patch
├── scripts/                           launch and check wrappers
└── tests/                             protocol and sandbox unit tests
```

Neither benchmark assets, model weights, nor training data are bundled. See
[docs/BENCHMARK_SETUP.md](docs/BENCHMARK_SETUP.md) for how to install the
WebShop, ALFWorld, and DBBench environments.

## Installation

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

> [!IMPORTANT]
> The target server must return structured `message.tool_calls`. XML-looking tool
> text inside `message.content` is not equivalent and silently zeroes rewards.
> Probe before a long run:
>
> ```bash
> python scripts/probe_openai_tool_calls.py \
>   --base-url "$TARGET_BASE_URL" --model "$TARGET_MODEL"
> ```

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

> [!IMPORTANT]
> WebShop results are reportable only when `webshop_goal_seed=233` and the strict
> `webshop_batch_identity_v1` task manifests agree between baseline and patched
> rerun. Matching integer task indices are **not** proof of a paired comparison.

## Training

Three model roles: a **target agent** that runs benchmark tasks and stays
frozen within a stage, the **harness engineer** being trained, and the frozen
**reference policy** used by GRPO. The reward is not a learned judge — every
valid patch is compiled and scored by an actual same-batch rerun.

All three stages run on a single node with 8× NVIDIA H800 GPUs. The reference
runs use 877 cold-start editing examples, roughly 1,500 RL failure packets, and
2,515 agent-SFT trajectories.

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

> [!WARNING]
> Never reuse baseline rewards produced by a different target model or serving
> protocol. SFT, RL, and evaluation must also agree on the exact system prompt,
> static user prefix and schema, `schema_style`, and the `prefill_think_patch`
> response protocol; changing the static protocol causes train/eval drift and
> sharply reduces format validity.

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

## Release Roadmap

| Phase | Contents | Status |
|---|---|---|
| 1 | Training and evaluation code, patch protocol, sandbox, configs, docs | ✅ Available |
| 2 | Harness-engineer checkpoints for both main-table rows | ✅ Available |
| 3 | Public paper link and citation entry | ⏳ Planned |

## License

Harness-R1 code is released under Apache-2.0. Relax, AgentBench, and the
benchmark environments retain their original notices and licenses; see
[NOTICE](NOTICE).

## Citation

The paper does not yet have a public identifier. A BibTeX entry will be added
here once it does; until then, please cite this repository and the paper title:

```text
Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories.
Shuai Shao, Kangning Zhang, Qingyao Li, Shijian Wang, Hao Wang,
Wenxiang Jiao, Yuan Lu, Yi Guo, Weiwen Liu, Weinan Zhang. 2026.
https://github.com/DeepExperience/Harness-R1
```
