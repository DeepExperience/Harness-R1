"""Check the archived evidence and the public main-table invocation contract."""

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
REFERENCE = ROOT / "reproduction/webshop_e0"
sys.path.insert(0, str(ROOT / "code/Relax/examples/harness_r1"))

import eval_webshop_checkpoint_patches as evaluator  # noqa: E402
from harness_r1_webshop_identity import (  # noqa: E402
    manifest_sha256,
    validate_task_manifest,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class MainTableReferenceTest(unittest.TestCase):
    def test_checksums(self) -> None:
        lines = (REFERENCE / "SHA256SUMS").read_text().splitlines()
        self.assertTrue(lines)
        for line in lines:
            expected, name = line.split("  ", 1)
            path = REFERENCE / name
            self.assertTrue(path.is_relative_to(REFERENCE))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.assertEqual(digest.hexdigest(), expected, name)

    def test_task_coverage_and_paired_rewards(self) -> None:
        tasks = read_jsonl(REFERENCE / "data/tasks.jsonl")
        self.assertEqual([t["index"] for t in tasks], list(range(500)))
        summary = json.loads((REFERENCE / "data/summary.json").read_text())
        self.assertEqual(summary["valid_patches"], 50)
        self.assertEqual(summary["eval_ok_batches"], 50)
        for task in tasks:
            manifest = validate_task_manifest(
                task["manifest"], expected_index=task["index"], expected_goal_seed=233,
            )
            self.assertEqual(manifest["instruction_sha256"], manifest_sha256(task["instruction"]))
        for condition in ("baseline", "patched"):
            rewards = [t[condition + "_reward"] for t in tasks]
            self.assertEqual(sum(r >= 1 for r in rewards), summary[condition + "_pass"])
            self.assertAlmostEqual(sum(rewards) / len(rewards), summary[condition + "_average_reward"])

    def test_prompt_and_patch_provenance(self) -> None:
        inputs = read_jsonl(REFERENCE / "data/engineer_inputs.jsonl")
        tasks = {t["index"]: t for t in read_jsonl(REFERENCE / "data/tasks.jsonl")}
        hashes = {
            row["batch_tag"]: row["prompt_sha256"]
            for row in json.loads((REFERENCE / "data/prompt_hashes.json").read_text())
        }
        self.assertEqual(len(inputs), 50)
        for ordinal, record in enumerate(inputs):
            metadata = record["metadata"]
            tag = metadata["batch_tag"]
            self.assertEqual(metadata["schema_style"], "webshop_code_only")
            self.assertEqual(metadata["response_protocol"], "prefill_think_patch")
            self.assertEqual(metadata["start"], ordinal * 10)
            self.assertEqual(metadata["end"], ordinal * 10 + 10)
            self.assertEqual(manifest_sha256(record["prompt"]), hashes[tag])
            for index in range(metadata["start"], metadata["end"]):
                self.assertEqual(metadata["webshop_task_manifests"][str(index)], tasks[index]["manifest"]["sha256"])
                self.assertEqual(metadata["baseline_rewards"][str(index)], tasks[index]["baseline_reward"])
            patch_dir = REFERENCE / "patches" / tag
            parsed, error = evaluator.parse_patch(
                (patch_dir / "raw_response.txt").read_text(),
                response_protocol="prefill_think_patch", require_code_hook_only=True,
            )
            self.assertIsNotNone(parsed, error)
            self.assertEqual(parsed, json.loads((patch_dir / "patch.json").read_text()))


class MainTableEntryPointTest(unittest.TestCase):
    def invoke(self, directory: Path, mode: str, *, existing_output: bool = False) -> subprocess.CompletedProcess:
        launcher = directory / "capture_args"
        launcher.write_text(
            "#!" + sys.executable + "\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
        )
        launcher.chmod(0o700)
        output = directory / "result"
        if existing_output:
            output.mkdir()
        env = dict(os.environ)
        env.update({
            "EVAL_PYTHON": str(launcher),
            "AGENTBENCH_PYTHON": sys.executable,
            "WEBSHOP_WORKER_PYTHON": sys.executable,
            "TARGET_BASE_URL": "http://127.0.0.1:18000/v1",
            "TARGET_MODEL": "Qwen3.5-9B",
            "OUTPUT_ROOT": str(output),
            "CONTROLLER_PORT_BASE": "30000",
            "WORKER_PORT_BASE": "31000",
            "ENGINEER_BACKEND": "vllm",
        })
        env.pop("ENGINEER_API_KEY_ENV", None)
        if mode == "generate":
            env.update({"ENGINEER_BASE_URL": "http://127.0.0.1:18001/v1", "ENGINEER_MODEL": "harness-r1"})
        else:
            env.pop("ENGINEER_BASE_URL", None)
            env.pop("ENGINEER_MODEL", None)
        return subprocess.run(
            ["bash", str(ROOT / "scripts/reproduce_webshop_main_table.sh"), mode, "--limit", "1"],
            cwd=directory, env=env, text=True, capture_output=True,
        )

    def test_both_modes_use_original_inputs_and_protocol(self) -> None:
        for mode in ("replay", "generate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = self.invoke(Path(directory), mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                with mock.patch.object(sys, "argv", json.loads(result.stdout)):
                    args = evaluator.parse_args()
                self.assertEqual(args.input, REFERENCE / "data/engineer_inputs.jsonl")
                self.assertEqual(args.engineer_assistant_prefill, "<think>\n")
                self.assertEqual(args.engineer_max_tokens, 12288)
                self.assertEqual(args.rollout_max_tokens, 4096)
                self.assertEqual(args.engineer_chat_template_kwargs, {"enable_thinking": True})
                self.assertEqual(args.rollout_chat_template_kwargs, {"enable_thinking": False})
                evaluator.validate_input_records(args, read_jsonl(args.input))
                if mode == "replay":
                    self.assertEqual(args.patch_source_root, REFERENCE / "patches")
                else:
                    self.assertIsNone(args.patch_source_root)
                    self.assertEqual(args.engineer_extra_body, {"add_generation_prompt": False})

    def test_existing_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(Path(directory), "replay", existing_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
