from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = REPO_ROOT / "code/life-harness/AgentBench/scripts"
AGENTBENCH = REPO_ROOT / "code/life-harness/AgentBench"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(AGENTBENCH))

from harness_r1_webshop_identity import (  # noqa: E402
    WEBSHOP_TASK_MANIFEST_PROTOCOL,
    WebShopIdentityError,
    assert_matching_task_hashes,
    identity_metadata_from_rows,
    manifest_sha256,
    validate_identity_metadata,
)
from webshop_agentrl_passthrough import inject_webshop_task_manifest  # noqa: E402
from src.server.harness.webshop import (  # noqa: E402
    PAGE_PRODUCT_DETAIL,
    WebShopHarnessConfig,
    WebShopHarnessRuntime,
)


def task_manifest(index: int, goal_seed: int = 233) -> dict:
    payload = {
        "protocol": WEBSHOP_TASK_MANIFEST_PROTOCOL,
        "goal_seed": goal_seed,
        "index": index,
        "instruction_sha256": manifest_sha256(f"instruction-{index}"),
        "goal_sha256": manifest_sha256({"index": index}),
        "product_prices_sha256": manifest_sha256({"shared": True}),
    }
    return {**payload, "sha256": manifest_sha256(payload)}


def rollout_row(index: int, goal_seed: int = 233) -> dict:
    return {
        "index": index,
        "output": {
            "result": {
                "reward": 0.0,
                "webshop_task_manifest": task_manifest(index, goal_seed),
            }
        },
    }


class WebShopIdentityTest(unittest.TestCase):
    def test_set_manifest_hash_is_order_independent(self) -> None:
        left = {"values": {"red", "green", "blue"}}
        right = {"values": set(reversed(["red", "green", "blue"]))}
        self.assertEqual(manifest_sha256(left), manifest_sha256(right))

    def test_round_trip_identity_metadata(self) -> None:
        metadata = {
            "start": 10,
            "end": 13,
            **identity_metadata_from_rows(
                [rollout_row(index) for index in range(10, 13)],
                start=10,
                end=13,
                goal_seed=233,
            ),
        }
        identity = validate_identity_metadata(metadata)
        self.assertEqual(identity["goal_seed"], 233)
        self.assertEqual(set(identity["task_hashes"]), {10, 11, 12})

    def test_missing_manifest_is_rejected(self) -> None:
        rows = [rollout_row(10), {"index": 11, "output": {"result": {"reward": 0}}}]
        with self.assertRaisesRegex(WebShopIdentityError, "missing webshop_task_manifest"):
            identity_metadata_from_rows(rows, start=10, end=12, goal_seed=233)

    def test_baseline_patched_mismatch_is_rejected(self) -> None:
        expected = {10: task_manifest(10)["sha256"]}
        actual = {10: task_manifest(10, goal_seed=234)["sha256"]}
        with self.assertRaisesRegex(WebShopIdentityError, r"mismatched=\[10\]"):
            assert_matching_task_hashes(expected, actual, require_complete=True)


class WebShopRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = WebShopHarnessRuntime(WebShopHarnessConfig(enabled=True))
        self.runtime.init_task("I need a blue shirt in size large")
        self.runtime.page_state.page_type = PAGE_PRODUCT_DETAIL
        self.runtime.page_state.attribute_options = {
            "color": ["blue", "red"],
            "size": ["large", "small"],
        }

    def test_predicate_inspection_has_no_counter_side_effect(self) -> None:
        for _ in range(8):
            message = self.runtime._buy_now_precheck(
                ["buy now", "blue", "large"],
                record_block=False,
                respect_safety_valve=False,
            )
            self.assertIsNotNone(message)
        self.assertEqual(self.runtime._total_buy_now_blocks, 0)
        self.assertEqual(self.runtime._defensive_buy_blocks, 0)

    def test_real_guard_records_one_block(self) -> None:
        message = self.runtime._buy_now_precheck(["buy now", "blue", "large"])
        self.assertIsNotNone(message)
        self.assertEqual(self.runtime._total_buy_now_blocks, 1)
        self.assertEqual(self.runtime._defensive_buy_blocks, 0)


class WebShopWorkerContractTest(unittest.TestCase):
    def test_terminal_manifest_is_whitelisted_into_worker_response(self) -> None:
        class Box:
            pass

        manifest = task_manifest(7)
        running = Box()
        running.session = Box()
        running.session.controller = Box()
        running.session.controller.env_output = Box()
        running.session.controller.env_output.result = {
            "webshop_task_manifest": manifest,
            "private_debug_payload": {"must_not": "leak"},
        }
        response = {"session_id": 3, "env_out": {"reward": 1.0}}

        actual = inject_webshop_task_manifest(response, running)

        self.assertEqual(actual["env_out"]["webshop_task_manifest"], manifest)
        self.assertNotIn("private_debug_payload", actual["env_out"])
        self.assertIsNot(actual["env_out"]["webshop_task_manifest"], manifest)

    def test_interact_keeps_fastapi_request_annotation(self) -> None:
        worker_path = SCRIPTS / "webshop_agentrl_worker.py"
        module = ast.parse(worker_path.read_text(encoding="utf-8"))
        worker_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "SessionToolTaskWorker"
        )
        interact = next(
            node
            for node in worker_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "interact"
        )
        parameters = interact.args.args[1]
        self.assertIsInstance(parameters.annotation, ast.Name)
        self.assertEqual(parameters.annotation.id, "InteractRequest")

    def test_task_uses_episode_local_tool_copy(self) -> None:
        task_path = AGENTBENCH / "src/server/tasks/webshop/task.py"
        source = task_path.read_text(encoding="utf-8")
        self.assertIn("self._base_tools = copy.deepcopy(tools)", source)
        self.assertIn("episode_tools = copy.deepcopy(self._base_tools)", source)
        self.assertIn("session.set_tools(episode_tools)", source)

if __name__ == "__main__":
    unittest.main()
