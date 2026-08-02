from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "demo/artifacts"


class DemoTest(unittest.TestCase):
    """The offline demo must keep running against the real validator and sandbox."""

    def test_demo_runs_and_reports_expected_effects(self) -> None:
        proc = subprocess.run(
            [sys.executable, "demo/run_demo.py", "--no-color"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        for expected in ("block_and_prompt", "rewrite_action", "passes the action through"):
            self.assertIn(expected, proc.stdout)

    def test_artifacts_carry_no_absolute_workspace_paths(self) -> None:
        for path in sorted(ARTIFACTS.iterdir()):
            self.assertNotIn("/mnt/", path.read_text(), f"{path.name} leaks a local path")

    def test_stored_result_matches_stored_baseline_metadata(self) -> None:
        meta = json.loads((ARTIFACTS / "metadata.json").read_text())
        result = json.loads((ARTIFACTS / "result.json").read_text())
        self.assertEqual(meta["baseline_pass"], result["baseline_pass"])
        self.assertEqual(meta["batch_size"], result["batch_size"])
        self.assertEqual(sorted(meta["baseline_rewards"]), sorted(result["patched_rewards"]))


if __name__ == "__main__":
    unittest.main()
