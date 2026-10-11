"""Check the published ALFWorld input/output reference and invocation contract."""

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
REFERENCE = ROOT / "reproduction/alfworld_e0"
sys.path.insert(0, str(ROOT / "code/Relax/examples/harness_r1"))

import eval_alfworld_patches as evaluator  # noqa: E402


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ALFWorldReferenceTest(unittest.TestCase):
    def test_checksums_and_split_identity(self) -> None:
        for line in (REFERENCE / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest(), expected, name)
        split_file = ROOT / "data/alfworld/test_set.json"
        provenance = json.loads((REFERENCE / "data/provenance.json").read_text())
        self.assertEqual(hashlib.sha256(split_file.read_bytes()).hexdigest(), provenance["split_sha256"])
        split = json.loads(split_file.read_text())["mixed"]
        tasks = jsonl(REFERENCE / "data/tasks.jsonl")
        self.assertEqual([t["index"] for t in tasks], list(range(500)))
        self.assertEqual([t["game_file"] for t in tasks], split)
        self.assertEqual(sum(t["baseline_reward"] >= 1 for t in tasks), 203)
        self.assertEqual(sum(t["effective_patched_reward"] >= 1 for t in tasks), 266)

    def test_inputs_and_all_outputs_are_preserved(self) -> None:
        records = jsonl(REFERENCE / "data/engineer_inputs.jsonl")
        hashes = {r["batch_tag"]: r["prompt_sha256"] for r in json.loads((REFERENCE / "data/prompt_hashes.json").read_text())}
        tasks = jsonl(REFERENCE / "data/tasks.jsonl")
        self.assertEqual(len(records), 50)
        results = []
        valid = 0
        for ordinal, record in enumerate(records):
            metadata = record["metadata"]
            tag = metadata["batch_tag"]
            self.assertEqual(metadata["split"], "test_set")
            self.assertEqual(metadata["schema_style"], "alfworld_life_multihook_v1")
            self.assertEqual(metadata["response_protocol"], "prefill_think_patch")
            self.assertEqual(metadata["start"], ordinal * 10)
            self.assertEqual(metadata["end"], ordinal * 10 + 10)
            encoded = json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), hashes[tag])
            directory = REFERENCE / "outputs" / tag
            result = json.loads((directory / "result.json").read_text())
            results.append(result)
            patch, error = evaluator.parse_patch((directory / "raw_response.txt").read_text(), "prefill_think_patch")
            self.assertEqual(patch is not None, result["valid_patch"], error)
            if patch is not None:
                self.assertEqual(patch, json.loads((directory / "patch.json").read_text()))
                self.assertEqual(result["eval_status"], "ok")
                valid += 1
            else:
                self.assertFalse((directory / "patch.json").exists())
                self.assertEqual(result["finish_reason"], "length")
                self.assertEqual(result["patched_pass"], result["baseline_pass"])
            for local_index in range(10):
                task = tasks[ordinal * 10 + local_index]
                self.assertEqual(task["baseline_reward"], metadata["baseline_rewards"][str(local_index)])
                effective = result.get("patched_rewards", metadata["baseline_rewards"])[str(local_index)]
                self.assertEqual(task["effective_patched_reward"], effective)
        self.assertEqual(valid, 43)
        summary = evaluator.summarize(results)
        self.assertEqual(summary["baseline_pass"], 203)
        self.assertEqual(summary["patched_pass"], 266)
        self.assertEqual(summary["eval_ok_batches"], 43)


class ALFWorldEntryPointTest(unittest.TestCase):
    def invoke(self, directory: Path, mode: str, *, wrong_split: bool = False) -> subprocess.CompletedProcess:
        installed = directory / "agentbench/data/alfworld/test_set.json"
        installed.parent.mkdir(parents=True)
        split = json.loads((ROOT / "data/alfworld/test_set.json").read_text())
        if wrong_split:
            split["mixed"] = list(reversed(split["mixed"]))
        installed.write_text(json.dumps(split))
        launcher = directory / "capture_args"
        launcher.write_text(
            "#!" + sys.executable + "\n"
            "import json, subprocess, sys\n"
            "if sys.argv[1] == '-':\n"
            "    raise SystemExit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)\n"
            "print(json.dumps(sys.argv[1:]))\n"
        )
        launcher.chmod(0o700)
        env = dict(os.environ)
        env.update({
            "EVAL_PYTHON": str(launcher), "AGENTBENCH_PYTHON": sys.executable,
            "ALFWORLD_WORKER_PYTHON": sys.executable, "AGENTBENCH_DIR": str(directory / "agentbench"),
            "TARGET_BASE_URL": "http://127.0.0.1:18000/v1", "TARGET_MODEL": "Qwen3.5-9B",
            "OUTPUT_ROOT": str(directory / "new_result"), "CONTROLLER_PORT_BASE": "33000",
            "WORKER_PORT_BASE": "34000", "ENGINEER_BACKEND": "vllm",
        })
        env.pop("ENGINEER_API_KEY_ENV", None)
        if mode == "generate":
            env.update({"ENGINEER_BASE_URL": "http://127.0.0.1:18001/v1", "ENGINEER_MODEL": "harness-r1"})
        else:
            env.pop("ENGINEER_BASE_URL", None)
            env.pop("ENGINEER_MODEL", None)
        return subprocess.run(
            ["bash", str(ROOT / "scripts/reproduce_alfworld_main_table.sh"), mode, "--limit", "1"],
            cwd=directory, env=env, text=True, capture_output=True,
        )

    def test_replay_and_vllm_generation_arguments(self) -> None:
        for mode in ("replay", "generate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = self.invoke(Path(directory), mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                with mock.patch.object(sys, "argv", json.loads(result.stdout)):
                    args = evaluator.parse_args()
                self.assertEqual(args.input, REFERENCE / "data/engineer_inputs.jsonl")
                self.assertEqual(args.alfworld_split, "test_set")
                self.assertEqual(args.alfworld_max_step, 50)
                self.assertEqual(args.engineer_assistant_prefill, "<think>\n")
                self.assertEqual(args.engineer_max_tokens, 12288)
                self.assertEqual(args.rollout_max_tokens, 4096)
                self.assertEqual(args.response_protocol, "prefill_think_patch")
                self.assertEqual(args.engineer_chat_template_kwargs, {"enable_thinking": True})
                self.assertEqual(args.rollout_chat_template_kwargs, {"enable_thinking": False})
                if mode == "replay":
                    self.assertEqual(args.patch_source_root, REFERENCE / "outputs")
                else:
                    self.assertIsNone(args.patch_source_root)
                    self.assertEqual(args.engineer_extra_body, {"add_generation_prompt": False})

    def test_wrong_split_is_rejected_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(Path(directory), "replay", wrong_split=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from the published", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
