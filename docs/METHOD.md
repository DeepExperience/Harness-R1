# Method

Harness-R1 models harness engineering as a single-round contextual decision.
For one batch, the state is the current harness substrate plus compressed
baseline trajectories. The action is one structured code-hook patch. The
environment compiles that patch, reruns the same tasks, and returns the metric
delta.

## One Training Sample

1. Run a target agent without learned harness policies on a fixed task batch.
2. Extract failure evidence: instruction, terminal status/reward, selected
   action-observation events, and benchmark-specific runtime signals.
3. Ask the engineer to analyze recurring failures and emit a reusable patch.
4. Parse `<think>` and `<patch>`, validate the JSON contract, reject leakage,
   compile every hook in the sandbox, and reject runtime no-ops if configured.
5. Rerun the same batch with the patch and the same target model protocol.
   WebShop additionally verifies a canonical manifest hash for every task.
6. Return `patched_metric - baseline_metric`. Invalid output is no-patch and
   therefore receives zero delta.

The engineer does not select tasks, edit model weights, or write task answers.
It edits a small runtime program that mediates interaction between an agent and
an environment.

## Runtime Substrate

Each episode owns a mutable dictionary `nb` (notebook). Hooks receive a
read-only deep copy of runtime context `ctx` and the mutable notebook.

- `on_init` initializes `nb` and may return reusable skills or a tool hint.
- `on_post_step` observes the latest transition and primarily updates `nb`.
- `make_pre_hint` turns current state into a soft, deduplicated hint.
- `on_before_action` applies narrow hard intervention when the benchmark
  runtime supports it.

The substrate is deliberately smaller than the hand-engineered Life-Harness
policy. Built-in H2/H3/H4/H5 modules are disabled in Harness-R1 experiments.

## Safety and Generalization

`code_runner.py` parses hook code with Python AST checks. It rejects imports,
file/network access, dunder access, dynamic evaluation, unbounded loops,
classes, lambdas, nested functions, unsafe builtins, oversized programs, and
benchmark-specific leakage patterns. Runtime execution is line- and
time-bounded; hook failures degrade to no effect instead of crashing the task.

Generalization is encouraged by batch-level evidence, anti-leakage validation,
short reusable state machines, same-batch credit assignment during training,
held-out task splits for model selection, and cross-agent/cross-benchmark
evaluation.

For WebShop, matching integer indices are not accepted as proof of a paired
comparison. Each baseline task records hashes of its instruction, goal, product
price state, and seed. Reward and offline evaluation reject a patched rerun if
any task manifest is missing or different.

## Reward

The released rewards support pass delta and average shaped-reward delta:

```text
delta_pass_rate = (patched_pass - baseline_pass) / batch_size
delta_average_reward = mean(patched_rewards) - mean(baseline_rewards)
```

The main mixed RL path uses `delta_average_reward` with no validity bonus. A
patch must improve behavior, not merely satisfy the output grammar.
