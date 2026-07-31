from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = ROOT / "code/Relax/examples/harness_r1"
sys.path.insert(0, str(EVAL_SCRIPTS))

from eval_dbbench_gpt_patches import align_prompt_prefix_to_reference  # noqa: E402


class DBBenchPromptAlignmentTest(unittest.TestCase):
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
