"""Exercise release entry points without depending on their checkout location."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/Relax/examples/harness_r1"))
sys.path.insert(0, str(ROOT / "code/life-harness/AgentBench/scripts"))
import eval_alfworld_patches as aw  # noqa: E402
import eval_dbbench_gpt_patches as db  # noqa: E402
import eval_webshop_checkpoint_patches as ws  # noqa: E402
import harness_r1_batch_debug as batch  # noqa: E402
import harness_r1_paths as paths  # noqa: E402


@contextmanager
def in_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def capture_command(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!" + sys.executable + "\nimport json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'pythonpath': os.environ.get('PYTHONPATH'), 'runtime': os.environ.get('RUNTIME_ENV_JSON')}))\n"
    )
    path.chmod(0o700)


class RuntimePathTest(unittest.TestCase):
    def test_evaluator_defaults_use_the_bundled_tree(self) -> None:
        for module in (ws, aw, db):
            self.assertEqual(module.AGENTBENCH_DIR, ROOT / "code/life-harness/AgentBench")
        self.assertTrue(paths.default_python().is_file())
        with tempfile.TemporaryDirectory() as tmp, in_directory(Path(tmp)):
            self.assertEqual(paths.repo_path("outputs/cache"), ROOT / "outputs/cache")

    def test_python_virtualenv_identity_survives_argument_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            environment = directory / "venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            python = environment / "bin/python"
            for module in (ws, aw, db):
                flag = "--prepared-input" if module is db else "--input"
                argv = ["eval", flag, str(directory / "input.jsonl"), "--output-root", str(directory / "out"),
                        "--run-id", "path-test", "--agentbench-python", str(python)]
                with mock.patch.object(sys, "argv", argv):
                    args = module.parse_args()
                self.assertEqual(args.agentbench_python, python)
                prefix = subprocess.check_output([str(args.agentbench_python), "-c", "import sys; print(sys.prefix)"], text=True).strip()
                self.assertEqual(prefix, str(environment))
            self.assertEqual(paths.default_worker_python(directory, "webshop", python), python)

    def test_worker_import_paths_survive_a_different_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            agentbench = directory / "agentbench"
            dependency = directory / "dependency"
            agentbench.mkdir()
            dependency.mkdir()
            (dependency / "portable_dependency.py").write_text("VALUE = 'loaded'\n")
            args = SimpleNamespace(agentbench_dir=agentbench, agentbench_python=Path(sys.executable),
                                   webshop_worker_python=None, alfworld_worker_python=None,
                                   dbbench_worker_python=Path(sys.executable))
            with in_directory(directory), mock.patch.dict(os.environ, {"PYTHONPATH": "dependency"}):
                for bench in ("webshop", "alfworld", "dbbench"):
                    env = batch.worker_env(args, bench)
                    result = subprocess.check_output([sys.executable, "-c", "import portable_dependency; print(portable_dependency.VALUE)"], cwd=agentbench, env=env, text=True)
                    self.assertEqual(result.strip(), "loaded")

    def test_batch_defaults_follow_an_explicit_agentbench_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with mock.patch.object(sys, "argv", ["batch", "--agentbench-dir", str(directory)]):
                args = batch.parse_args()
            self.assertEqual(args.dbbench_data_file, directory / "data/dbbench/standard.jsonl")
            self.assertTrue(args.agentbench_python.is_file())


class ShellPathTest(unittest.TestCase):
    def test_eval_wrappers_preserve_caller_relative_paths_and_json(self) -> None:
        for bench, module in (("webshop", ws), ("alfworld", aw), ("dbbench", db)):
            with self.subTest(bench=bench), tempfile.TemporaryDirectory(prefix="harness paths ") as tmp:
                directory = Path(tmp)
                capture_command(directory / "bin/capture")
                env = dict(os.environ)
                env.update({"PYTHON_BIN": "bin/capture", "AGENTBENCH_PYTHON": "bin/capture",
                            "ALFWORLD_WORKER_PYTHON": "bin/capture", "WEBSHOP_WORKER_PYTHON": "bin/capture",
                            "DBBENCH_WORKER_PYTHON": "bin/capture", "PATCH_SOURCE_ROOT": "patches",
                            "AGENTBENCH_DIR": "agentbench", "DBBENCH_DATA_FILE": "standard.jsonl",
                            "PYTHONPATH": "dependency", "ENGINEER_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":true}',
                            "ROLLOUT_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":false}'})
                env.pop("PROMPT_PREFIX_REFERENCE", None)
                result = subprocess.run(["bash", str(ROOT / "scripts" / ("eval_" + bench + ".sh")), "input.jsonl", "new_result"], cwd=directory, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                captured = json.loads(result.stdout)
                with mock.patch.object(sys, "argv", captured["argv"]):
                    args = module.parse_args()
                input_path = args.prepared_input if bench == "dbbench" else args.input
                self.assertEqual(input_path, directory / "input.jsonl")
                self.assertEqual(args.output_root, directory / "new_result")
                self.assertEqual(args.patch_source_root, directory / "patches")
                self.assertEqual(args.agentbench_python, directory / "bin/capture")
                self.assertEqual(captured["pythonpath"], str(directory / "dependency"))
                self.assertEqual(args.engineer_chat_template_kwargs, {"enable_thinking": True})
                if bench == "dbbench":
                    self.assertEqual(args.dbbench_data_file, directory / "standard.jsonl")

    def test_sft_config_and_executable_are_relative_to_the_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            capture_command(directory / "bin/trainer")
            env = {**os.environ, "LLAMAFACTORY_CLI": "bin/trainer", "PYTHONPATH": "dependency"}
            result = subprocess.run(["bash", str(ROOT / "scripts/train_engineer_sft.sh"), "config.yaml"], cwd=directory, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            captured = json.loads(result.stdout)
            self.assertEqual(captured["argv"], ["train", str(directory / "config.yaml")])
            self.assertEqual(captured["pythonpath"], str(directory / "dependency"))

    def test_rl_paths_are_fixed_before_entering_relax(self) -> None:
        for checkpoint in ("weights", "Qwen/Qwen3.5-9B"):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                checkout = directory / "release"
                (checkout / "scripts").mkdir(parents=True)
                models = checkout / "code/Relax/scripts/models"
                models.mkdir(parents=True)
                for name in ("train_engineer_rl.sh", "path_helpers.sh"):
                    shutil.copyfile(ROOT / "scripts" / name, checkout / "scripts" / name)
                shutil.copyfile(ROOT / "code/Relax/scripts/models/qwen35-9B.sh", models / "qwen35-9B.sh")
                (directory / "weights").mkdir()
                capture_command(directory / "bin/trainer")
                env = dict(os.environ)
                env.update({"PYTHON_BIN": "bin/trainer", "RUN_DIRECT": "1", "QWEN35_HF": checkpoint,
                            "HARNESS_R1_DATA": "prompts.jsonl", "HARNESS_R1_CONFIG": "reward.yaml", "NUM_ROLLOUT": "1",
                            "SAVE_PATH": "saved", "LOAD_PATH": "resume", "EXP_DIR": "experiment",
                            "MODEL_CONFIG_DIR": str(models), "WANDB_API_KEY": "test-only", "PYTHONPATH": "dependency",
                            "MEGATRON": "vendor/megatron", "SGLANG": "vendor/sglang", "RELAX": str(checkout / "code/Relax")})
                env.pop("RUNTIME_ENV_JSON", None)
                result = subprocess.run(["bash", str(checkout / "scripts/train_engineer_rl.sh")], cwd=directory, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                captured = json.loads(result.stdout)
                argv = captured["argv"]
                expected_model = str(directory / "weights") if checkpoint == "weights" else checkpoint
                for flag, expected in (("--hf-checkpoint", expected_model), ("--prompt-data", str(directory / "prompts.jsonl")),
                                       ("--custom-config-path", str(directory / "reward.yaml")), ("--save", str(directory / "saved")),
                                       ("--load", str(directory / "resume"))):
                    self.assertEqual(argv[argv.index(flag) + 1], expected)
                self.assertIn(str(directory / "dependency"), captured["pythonpath"].split(":"))
                runtime = json.loads(captured["runtime"])
                self.assertEqual(runtime["env_vars"]["PYTHONPATH"], captured["pythonpath"])


if __name__ == "__main__":
    unittest.main()
