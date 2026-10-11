from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = ROOT / "code/Relax/examples/harness_r1"
sys.path.insert(0, str(EVAL_SCRIPTS))

from eval_dbbench_gpt_patches import align_prompt_prefix_to_reference  # noqa: E402
import build_dbbench_rl_dataset as builder  # noqa: E402


class DBBenchPromptAlignmentTest(unittest.TestCase):
    def test_dataset_builder_constructs_records_without_a_prefix_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "source/dbbench/batch_000"
            rollout = batch / "rollout/target/dbbench"
            rollout.mkdir(parents=True)
            (batch / "status.json").write_text(json.dumps({"range": [0, 2]}))
            rows = [
                {"index": 0, "output": {"status": "completed", "result": {
                    "reward": 0.0, "openai_messages": [
                        {"role": "user", "content": "Find the number of rows in this table."},
                        {"role": "tool", "content": "SQL syntax error: check the identifier quoting."},
                    ],
                }}},
                {"index": 1, "output": {"status": "completed", "result": {"reward": 1.0}}},
            ]
            (rollout / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with mock.patch.object(sys, "argv", [
                "build_dbbench_rl_dataset.py", "--output", str(root / "prepared.jsonl"),
                "--batch-size", "2", "--source-run-root", str(root / "source"),
            ]):
                args = builder.parse_args()
            records = builder.build_rows(args, root / "source")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["metadata"]["baseline_pass"], 1)
            self.assertEqual(records[0]["metadata"]["batch_size"], 2)
            self.assertIsNone(records[0]["metadata"]["prompt_prefix_reference"])
            self.assertIn("SQL syntax error", records[0]["prompt"][1]["content"])

    def test_static_prefix_changes_without_replacing_batch_evidence(self) -> None:
        marker = "Observed no-harness rollout evidence:"
        prompt = [
            {"role": "system", "content": "generated system"},
            {
                "role": "user",
                "content": f"generated schema\n\n{marker}\ncurrent batch evidence",
            },
        ]
        reference = [
            {"role": "system", "content": "training system"},
            {
                "role": "user",
                "content": f"training schema\n\n{marker}\nreference row evidence",
            },
        ]

        aligned = align_prompt_prefix_to_reference(prompt, reference)

        self.assertEqual(aligned[0]["content"], "training system")
        self.assertIn("training schema", aligned[1]["content"])
        self.assertIn("current batch evidence", aligned[1]["content"])
        self.assertNotIn("reference row evidence", aligned[1]["content"])
        self.assertEqual(prompt[0]["content"], "generated system")

    def test_missing_evidence_marker_is_rejected(self) -> None:
        prompt = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "no evidence marker"},
        ]
        with self.assertRaisesRegex(ValueError, "missing marker"):
            align_prompt_prefix_to_reference(prompt, prompt)


if __name__ == "__main__":
    unittest.main()
