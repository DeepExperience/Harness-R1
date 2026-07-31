from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/life-harness/AgentBench/scripts"
sys.path.insert(0, str(SCRIPTS))

from harness_r1_patch import (  # noqa: E402
    PatchValidationError,
    compile_patch_to_task_definition,
    normalize_patch,
    require_code_hook_only_patch,
)


class PatchProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = json.loads((ROOT / "examples/webshop_patch.json").read_text())
        self.patch = normalize_patch(raw, bench="webshop")

    def test_example_is_code_hook_only(self) -> None:
        require_code_hook_only_patch(self.patch)
        self.assertEqual(self.patch["benchmark"], "webshop")
        self.assertEqual(len(self.patch["actions"]), 2)

    def test_compile_keeps_legacy_policies_disabled(self) -> None:
        task = {
            "webshop-test": {
                "module": "src.server.tasks.webshop.WebShop",
                "parameters": {
                    "enabled": False,
                    "h2": False,
                    "h3": False,
                    "h4": False,
                    "h5": False,
                },
            }
        }
        compiled = compile_patch_to_task_definition(task, self.patch, "webshop")
        params = compiled["webshop-test"]["parameters"]
        self.assertFalse(params["enabled"])
        self.assertFalse(any(params[name] for name in ("h2", "h3", "h4", "h5")))
        hooks = params["harness_overlay"]["code_hooks"]
        self.assertEqual([item["hook"] for item in hooks], ["on_init", "make_pre_hint"])
        self.assertTrue(params["harness_overlay"]["metadata"]["overlay_only"])

    def test_legacy_action_is_rejected_by_strict_protocol(self) -> None:
        patch = normalize_patch(
            {
                "benchmark": "webshop",
                "actions": [
                    {
                        "type": "add_or_edit_skill",
                        "skill_id": "generic_search",
                        "task_types": ["*"],
                        "keywords": ["search"],
                        "text": "Inspect constraints before buying.",
                    }
                ],
            }
        )
        with self.assertRaises(PatchValidationError):
            require_code_hook_only_patch(patch)


if __name__ == "__main__":
    unittest.main()
