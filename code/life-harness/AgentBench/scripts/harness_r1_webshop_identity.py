"""Strict WebShop task-identity helpers for baseline and patched reruns."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


WEBSHOP_TASK_MANIFEST_PROTOCOL = "webshop_task_manifest_v1"
WEBSHOP_BATCH_IDENTITY_PROTOCOL = "webshop_batch_identity_v1"


class WebShopIdentityError(ValueError):
    """Raised when baseline and patched WebShop tasks are not identical."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return [
            [str(key), _canonical_value(item)]
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonical_value(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def manifest_sha256(value: Any) -> str:
    canonical = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise WebShopIdentityError(f"{field} must be a 64-character SHA256 digest")
    return text


def validate_task_manifest(
    manifest: Any,
    *,
    expected_index: int | None = None,
    expected_goal_seed: int | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise WebShopIdentityError("missing webshop_task_manifest in rollout result")
    protocol = str(manifest.get("protocol") or "")
    if protocol != WEBSHOP_TASK_MANIFEST_PROTOCOL:
        raise WebShopIdentityError(
            f"unsupported WebShop task manifest protocol: {protocol!r}"
        )
    try:
        index = int(manifest["index"])
        goal_seed = int(manifest["goal_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WebShopIdentityError("task manifest index/goal_seed is invalid") from exc
    if expected_index is not None and index != int(expected_index):
        raise WebShopIdentityError(
            f"task manifest index mismatch: expected {expected_index}, got {index}"
        )
    if expected_goal_seed is not None and goal_seed != int(expected_goal_seed):
        raise WebShopIdentityError(
            f"task manifest goal_seed mismatch: expected {expected_goal_seed}, got {goal_seed}"
        )

    payload = {
        "protocol": protocol,
        "goal_seed": goal_seed,
        "index": index,
        "instruction_sha256": _require_sha256(
            manifest.get("instruction_sha256"), "instruction_sha256"
        ),
        "goal_sha256": _require_sha256(manifest.get("goal_sha256"), "goal_sha256"),
        "product_prices_sha256": _require_sha256(
            manifest.get("product_prices_sha256"), "product_prices_sha256"
        ),
    }
    recorded = _require_sha256(manifest.get("sha256"), "task manifest sha256")
    computed = manifest_sha256(payload)
    if recorded != computed:
        raise WebShopIdentityError(
            f"task manifest digest mismatch for index {index}: recorded={recorded}, computed={computed}"
        )
    return {**payload, "sha256": recorded}


def _result_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    output = row.get("output")
    if not isinstance(output, Mapping):
        return {}
    result = output.get("result")
    return result if isinstance(result, Mapping) else {}


def task_manifest_hashes_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    goal_seed: int,
    expected_indices: Iterable[int] | None = None,
) -> dict[int, str]:
    expected = None if expected_indices is None else {int(index) for index in expected_indices}
    hashes: dict[int, str] = {}
    for row in rows:
        index = row.get("index")
        if not isinstance(index, int):
            continue
        if expected is not None and index not in expected:
            raise WebShopIdentityError(f"unexpected rollout index {index}")
        manifest = validate_task_manifest(
            _result_of(row).get("webshop_task_manifest"),
            expected_index=index,
            expected_goal_seed=goal_seed,
        )
        digest = manifest["sha256"]
        previous = hashes.get(index)
        if previous is not None and previous != digest:
            raise WebShopIdentityError(
                f"conflicting task manifests for duplicate index {index}: {previous} != {digest}"
            )
        hashes[index] = digest

    if expected is not None:
        missing = sorted(expected - set(hashes))
        if missing:
            raise WebShopIdentityError(
                f"missing task manifests for indices: {missing[:20]}"
            )
    return dict(sorted(hashes.items()))


def batch_manifest_sha256(*, goal_seed: int, task_hashes: Mapping[int | str, str]) -> str:
    normalized = [
        [int(index), _require_sha256(digest, f"task manifest {index}")]
        for index, digest in sorted(task_hashes.items(), key=lambda pair: int(pair[0]))
    ]
    return manifest_sha256(
        {
            "protocol": WEBSHOP_BATCH_IDENTITY_PROTOCOL,
            "goal_seed": int(goal_seed),
            "tasks": normalized,
        }
    )


def identity_metadata_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    start: int,
    end: int,
    goal_seed: int,
) -> dict[str, Any]:
    task_hashes = task_manifest_hashes_from_rows(
        rows,
        goal_seed=goal_seed,
        expected_indices=range(start, end),
    )
    return {
        "webshop_identity_protocol": WEBSHOP_BATCH_IDENTITY_PROTOCOL,
        "webshop_goal_seed": int(goal_seed),
        "webshop_task_manifests": {
            str(index): digest for index, digest in task_hashes.items()
        },
        "webshop_task_manifest_sha256": batch_manifest_sha256(
            goal_seed=goal_seed,
            task_hashes=task_hashes,
        ),
    }


def validate_identity_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    protocol = str(metadata.get("webshop_identity_protocol") or "")
    if protocol != WEBSHOP_BATCH_IDENTITY_PROTOCOL:
        raise WebShopIdentityError(
            "WebShop metadata has no strict task identity; regenerate baseline rollouts "
            f"with {WEBSHOP_BATCH_IDENTITY_PROTOCOL}"
        )
    try:
        start = int(metadata["start"])
        end = int(metadata["end"])
        goal_seed = int(metadata["webshop_goal_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WebShopIdentityError("WebShop identity metadata has invalid range/goal_seed") from exc
    if end <= start:
        raise WebShopIdentityError(f"invalid WebShop batch range [{start}, {end})")

    raw_hashes = metadata.get("webshop_task_manifests")
    if not isinstance(raw_hashes, Mapping):
        raise WebShopIdentityError("webshop_task_manifests must be an index-to-SHA256 object")
    task_hashes: dict[int, str] = {}
    for raw_index, raw_digest in raw_hashes.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise WebShopIdentityError(f"invalid task manifest index: {raw_index!r}") from exc
        task_hashes[index] = _require_sha256(raw_digest, f"task manifest {index}")

    expected = set(range(start, end))
    if set(task_hashes) != expected:
        missing = sorted(expected - set(task_hashes))
        extra = sorted(set(task_hashes) - expected)
        raise WebShopIdentityError(
            f"task manifest index set mismatch; missing={missing[:20]}, extra={extra[:20]}"
        )
    recorded = _require_sha256(
        metadata.get("webshop_task_manifest_sha256"),
        "webshop_task_manifest_sha256",
    )
    computed = batch_manifest_sha256(goal_seed=goal_seed, task_hashes=task_hashes)
    if recorded != computed:
        raise WebShopIdentityError(
            f"batch task manifest mismatch: recorded={recorded}, computed={computed}"
        )
    return {
        "protocol": protocol,
        "start": start,
        "end": end,
        "goal_seed": goal_seed,
        "task_hashes": dict(sorted(task_hashes.items())),
        "sha256": recorded,
    }


def assert_matching_task_hashes(
    expected: Mapping[int, str],
    actual: Mapping[int, str],
    *,
    require_complete: bool,
) -> None:
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        index
        for index in set(actual) & set(expected)
        if actual[index] != expected[index]
    )
    missing = sorted(set(expected) - set(actual)) if require_complete else []
    if unexpected or mismatched or missing:
        raise WebShopIdentityError(
            "baseline/patched WebShop identity mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, mismatched={mismatched[:20]}"
        )
