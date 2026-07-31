from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/life-harness/AgentBench/scripts"
AGENTBENCH = ROOT / "code/life-harness/AgentBench"
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
        rows = [
            rollout_row(10),
            {"index": 11, "output": {"result": {"reward": 0.0}}},
        ]
        with self.assertRaisesRegex(WebShopIdentityError, "missing webshop_task_manifest"):
            identity_metadata_from_rows(
                rows,
                start=10,
                end=12,
                goal_seed=233,
            )

    def test_baseline_patched_mismatch_is_rejected(self) -> None:
        expected = {10: task_manifest(10)["sha256"]}
        actual = {10: task_manifest(10, goal_seed=234)["sha256"]}
        with self.assertRaisesRegex(WebShopIdentityError, r"mismatched=\[10\]"):
            assert_matching_task_hashes(
                expected,
                actual,
                require_complete=True,
            )


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


if __name__ == "__main__":
    unittest.main()
