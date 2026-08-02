# Runtime and Reward

Implementation details behind the loop shown in the README.

## Runtime Substrate

Each episode owns a mutable dictionary `nb` (notebook). Hooks receive a
read-only deep copy of runtime context `ctx` and the mutable notebook.

- `on_init` initializes `nb` and may return reusable skills or a tool hint.
- `on_post_step` observes the latest transition and primarily updates `nb`.
- `make_pre_hint` turns current state into a soft, deduplicated hint.
- `on_before_action` applies narrow hard intervention when the benchmark
  runtime supports it.

## Sandbox

`code_runner.py` parses hook code with Python AST checks. It rejects imports,
file/network access, dunder access, dynamic evaluation, unbounded loops,
classes, lambdas, nested functions, unsafe builtins, oversized programs, and
benchmark-specific leakage patterns. Runtime execution is line- and
time-bounded; hook failures degrade to no effect instead of crashing the task.

## Paired-Rerun Identity

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

The main mixed RL path uses `delta_average_reward` with no validity bonus.
Invalid, no-op, or incomplete evaluations receive zero delta, so a patch must
improve behavior rather than merely satisfy the output grammar.
