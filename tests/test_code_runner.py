from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTBENCH = ROOT / "code/life-harness/AgentBench"
sys.path.insert(0, str(AGENTBENCH))

from src.server.harness.code_runner import (  # noqa: E402
    HookCompileError,
    compile_hook,
    run_hook,
)


class CodeRunnerTest(unittest.TestCase):
    def test_safe_hook_updates_notebook_and_returns_hint(self) -> None:
        source = """def hook(ctx, nb):
    nb['seen'] = nb.get('seen', 0) + 1
    if 'error' in str(ctx.get('observation', '')).lower():
        return {'message': 'Inspect the latest error before retrying.'}
    return None
"""
        fn = compile_hook(source)
        nb = {}
        effect = run_hook(
            fn,
            {"observation": "ERROR: invalid action"},
            nb,
            hook_name="make_pre_hint",
        )
        self.assertEqual(nb["seen"], 1)
        self.assertEqual(effect, {"message": "Inspect the latest error before retrying."})

    def test_import_and_file_access_are_rejected(self) -> None:
        for source in (
            "import os\ndef hook(ctx, nb):\n    return None\n",
            "def hook(ctx, nb):\n    return open('secret')\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(HookCompileError):
                    compile_hook(source)

    def test_invalid_effect_degrades_to_none(self) -> None:
        fn = compile_hook("def hook(ctx, nb):\n    return {'kind': 'rewrite_action'}\n")
        effect = run_hook(fn, {}, {}, hook_name="on_before_action")
        self.assertIsNone(effect)

    def test_numbered_action_leakage_is_benchmark_scoped(self) -> None:
        source = (
            "def hook(ctx, nb):\n"
            "    return {'message': 'go to page 2 before retrying'}\n"
        )
        compile_hook(source, benchmark="webshop")
        with self.assertRaisesRegex(HookCompileError, "numbered ALFWorld"):
            compile_hook(source, benchmark="alfworld")


if __name__ == "__main__":
    unittest.main()
