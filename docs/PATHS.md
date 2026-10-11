# Paths and Interpreters

The supported entry points are under `scripts/`. They locate the checkout from
their own location and can be invoked from another working directory.

| Input | Resolution |
|---|---|
| Generic eval input/output arguments and path environment variables | Relative to the caller's directory, fixed before the wrapper changes directory |
| SFT configuration argument | Relative to the caller; paths inside the YAML follow LLaMA-Factory's working-directory rules (the repository root) |
| RL data/config/save/load path environment variables | Relative to the caller, fixed before entering `code/Relax` |
| RL checkpoint | Existing local paths are made absolute; Hugging Face repository identifiers are preserved |
| `harness_r1_*` paths in the RL reward YAML | Relative to the Harness-R1 repository root, or absolute |
| Python interpreter flags | Preserve virtualenv symlinks; do not replace them with the underlying system binary |
| Worker `PYTHONPATH` | Retained and made absolute before changing the worker's directory |

AgentBench defaults to the bundled `code/life-harness/AgentBench`, without
requiring the private development tree. Python defaults to its `.venv/bin/python`
when installed, otherwise the active interpreter. Existing dedicated benchmark
venvs are discovered; explicit interpreter options take precedence. Install the
required packages in the chosen environment before running a benchmark.

The low-level `harness_r1_batch_debug.py` retains its existing convention that
relative `--dbbench-data-file` values are relative to `--agentbench-dir`.
Top-level evaluators and the main-table scripts pass absolute file paths.
AgentBench's own `data/...` and `configs/...` references are relative to its
working directory; the runner sets that directory when launching workers.

Replace `/path/to/...` entries in configuration templates. In particular,
the SFT templates point to an external LLaMA-Factory DeepSpeed configuration;
the LLaMA-Factory `examples/deepspeed` directory is not part of this repository.
Historical source-data defaults in dataset builders require your own rollout
files; provide `--source-run-root` or the corresponding input option.

The vendored Relax utilities still include container-specific examples such
as `code/Relax/configs/env.yaml`, `code/Relax/scripts/entrypoint/`, and
`code/Relax/relax/utils/external/command_utils.py`. Their `/root/...` defaults
are not used by `scripts/train_engineer_rl.sh`; configure their own paths if
invoking them directly. Standard `/tmp`, `/proc`, JDK, and Docker-internal
paths are not research-machine paths.

The immutable archives under `reproduction/` describe their original standalone
layout. Use the current repository scripts for the latest path fixes. Tests
exercise argument construction from an external directory, virtualenv identity,
and worker imports without launching model services or training.
