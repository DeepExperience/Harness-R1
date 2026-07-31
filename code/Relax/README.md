# Relax (trimmed snapshot)

[Relax](https://github.com/redai-infra/Relax) is an asynchronous reinforcement
learning engine for omni-modal post-training, open-sourced by the Xiaohongshu AI
Infra Team. It uses Ray Serve for orchestration, Megatron-LM as the training
backend, and SGLang as the inference engine.

This directory is a trimmed source snapshot of Relax carried by the Harness-R1
release so that the online-RL stage runs without a separate patch step. Relative
to upstream it drops the documentation site, framework tests, unrelated example
projects, multimodal training recipes, container assets, and CI configuration,
and it adds the Harness-R1 integration under `examples/harness_r1/`.

For framework-level documentation, installation guidance, and the full source,
see the upstream repository. Relax is licensed under Apache-2.0; see `LICENSE`.

## What Harness-R1 uses

| Path | Role |
|---|---|
| `relax/entrypoints/train.py` | GRPO training entry point invoked by `scripts/train_engineer_rl.sh` |
| `relax/engine/rewards/` | Reward-function plumbing that loads the Harness-R1 reward module |
| `scripts/models/qwen35-9B.sh` | Model/parallelism definition for the engineer checkpoint |
| `scripts/entrypoint/` | Local and Ray-job launchers |
| `examples/harness_r1/` | Harness-R1 rewards, evaluators, and dataset builders |
