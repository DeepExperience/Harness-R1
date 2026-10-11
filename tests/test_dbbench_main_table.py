"""Verify archived DBBench records and the public wrapper arguments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reproduction/dbbench_e0"
sys.path.insert(0, str(ROOT / "code/Relax/examples/harness_r1"))
import eval_dbbench_gpt_patches as evaluator  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def capture_python(directory: Path) -> Path:
    path = directory / "capture_args"
    path.write_text(
        "#!" + sys.executable + "\n"
        "import json, subprocess, sys\n"
        "if sys.argv[1] == '-':\n"
        "    raise SystemExit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    path.chmod(0o700)
    return path


class DBBenchReferenceTest(unittest.TestCase):
    def test_reference_files_and_test_data(self) -> None:
        for line in (REFERENCE / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest(), expected, name)
        source = ROOT / "data/dbbench/standard.jsonl"
        provenance = json.loads((REFERENCE / "data/provenance.json").read_text())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), provenance["test_data_sha256"])
        data = jsonl(source)
        tasks = jsonl(REFERENCE / "data/tasks.jsonl")
        self.assertEqual(len(data), 300)
        self.assertEqual([task["index"] for task in tasks], list(range(300)))
        for task, row in zip(tasks, data):
            self.assertEqual(task["task_record_sha256"], canonical_hash(row))
        self.assertEqual(sum(task["baseline_reward"] >= 1 for task in tasks), 183)
        self.assertEqual(sum(task["patched_reward"] >= 1 for task in tasks), 196)

    def test_original_inputs_outputs_and_paired_rewards(self) -> None:
        records = jsonl(REFERENCE / "data/engineer_inputs.jsonl")
        tasks = jsonl(REFERENCE / "data/tasks.jsonl")
        hashes = {row["batch_tag"]: row["prompt_sha256"] for row in json.loads((REFERENCE / "data/prompt_hashes.json").read_text())}
        self.assertEqual(len(records), 30)
        results = []
        for ordinal, record in enumerate(records):
            metadata = record["metadata"]
            tag = metadata["batch_tag"]
            self.assertEqual(metadata["start"], ordinal * 10)
            self.assertEqual(metadata["end"], ordinal * 10 + 10)
            self.assertIsNone(metadata["task_ids"])
            self.assertEqual(metadata["schema_style"], "dbbench_life_multihook_v1")
            self.assertEqual(canonical_hash(record["prompt"]), hashes[tag])
            directory = REFERENCE / "outputs" / tag
            result = json.loads((directory / "result.json").read_text())
            results.append(result)
            patch, error = evaluator.parse_patch((directory / "raw_response.txt").read_text(), "prefill_think_patch")
            self.assertIsNotNone(patch, error)
            self.assertEqual(patch, json.loads((directory / "patch.json").read_text()))
            self.assertEqual(result["eval_status"], "ok")
            for index in range(metadata["start"], metadata["end"]):
                self.assertEqual(tasks[index]["baseline_reward"], metadata["baseline_rewards"][str(index)])
                self.assertEqual(tasks[index]["patched_reward"], result["patched_rewards"][str(index)])
        summary = evaluator.summarize(results)
        self.assertEqual(summary["baseline_pass"], 183)
        self.assertEqual(summary["patched_pass"], 196)
        self.assertEqual(summary["eval_ok_batches"], 30)


class DBBenchWrapperTest(unittest.TestCase):
    def test_generic_wrapper_json_defaults_and_exported_values(self) -> None:
        cases = [
            (None, None),
            ("", ""),
            ('{"enable_thinking":true}', '{"enable_thinking":false}'),
            ('{"enable_thinking":true,"custom":{"value":"}"}}', '{}'),
        ]
        for engineer, rollout in cases:
            with self.subTest(engineer=engineer, rollout=rollout), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                env = dict(os.environ)
                env.update({"PYTHON_BIN": str(capture_python(directory)), "ENGINEER_EXTRA_BODY": '{"add_generation_prompt":false}'})
                for key, value in (("ENGINEER_CHAT_TEMPLATE_KWARGS", engineer), ("ROLLOUT_CHAT_TEMPLATE_KWARGS", rollout)):
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value
                result = subprocess.run([
                    "bash", str(ROOT / "scripts/eval_dbbench.sh"),
                    str(REFERENCE / "data/engineer_inputs.jsonl"), str(directory / "new_result"),
                ], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                with mock.patch.object(sys, "argv", json.loads(result.stdout)):
                    args = evaluator.parse_args()
                self.assertEqual(args.engineer_chat_template_kwargs, json.loads(engineer) if engineer else {"enable_thinking": True})
                self.assertEqual(args.rollout_chat_template_kwargs, json.loads(rollout) if rollout else {"enable_thinking": False})
                self.assertEqual(args.engineer_extra_body, {"add_generation_prompt": False})

    def test_main_table_replay_and_vllm_generation(self) -> None:
        for mode in ("replay", "generate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                env = dict(os.environ)
                env.update({
                    "EVAL_PYTHON": str(capture_python(directory)), "AGENTBENCH_PYTHON": sys.executable,
                    "DBBENCH_WORKER_PYTHON": sys.executable, "DBBENCH_MYSQL_HOST": "127.0.0.1:13306",
                    "TARGET_BASE_URL": "http://127.0.0.1:18000/v1", "TARGET_MODEL": "Qwen3.5-9B",
                    "OUTPUT_ROOT": str(directory / "new_result"), "CONTROLLER_PORT_BASE": "36000",
                    "WORKER_PORT_BASE": "37000", "ENGINEER_BACKEND": "vllm",
                })
                env.pop("ENGINEER_API_KEY_ENV", None)
                if mode == "generate":
                    env.update({"ENGINEER_BASE_URL": "http://127.0.0.1:18001/v1", "ENGINEER_MODEL": "harness-r1"})
                else:
                    env.pop("ENGINEER_BASE_URL", None)
                    env.pop("ENGINEER_MODEL", None)
                result = subprocess.run([
                    "bash", str(ROOT / "scripts/reproduce_dbbench_main_table.sh"), mode, "--limit", "1",
                ], cwd=directory, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                with mock.patch.object(sys, "argv", json.loads(result.stdout)):
                    args = evaluator.parse_args()
                self.assertEqual(args.prepared_input, REFERENCE / "data/engineer_inputs.jsonl")
                self.assertEqual(args.dbbench_data_file, ROOT / "data/dbbench/standard.jsonl")
                self.assertEqual(args.dbbench_env_driver, "manual")
                self.assertEqual(args.dbbench_max_round, 20)
                self.assertEqual(args.engineer_assistant_prefill, "<think>\n")
                self.assertEqual(args.engineer_reasoning_effort, "")
                self.assertEqual(args.engineer_max_tokens, 12288)
                self.assertEqual(args.rollout_max_tokens, 4096)
                records = evaluator.select_records(args)
                self.assertEqual(len(records), 1)
                if mode == "replay":
                    self.assertEqual(args.patch_source_root, REFERENCE / "outputs")
                else:
                    self.assertIsNone(args.patch_source_root)
                    self.assertEqual(args.engineer_extra_body, {"add_generation_prompt": False})


if __name__ == "__main__":
    unittest.main()
