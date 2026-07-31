"""Small runtime DSL for Harness-R1 overlay rules.

This module intentionally evaluates a constrained expression language, not
Python code.  It is used for typed harness patches proposed by a harness
engineer model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def normalize_harness_overlay(raw: Any, benchmark: str = "") -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"skills": [], "guard_rules": [], "recovery_rules": [], "code_hooks": [], "metadata": {}}
    return {
        "benchmark": benchmark,
        "skills": [x for x in raw.get("skills", []) if isinstance(x, dict)],
        "guard_rules": [x for x in raw.get("guard_rules", []) if isinstance(x, dict)],
        "recovery_rules": [x for x in raw.get("recovery_rules", []) if isinstance(x, dict)],
        "code_hooks": [x for x in raw.get("code_hooks", []) if isinstance(x, dict)],
        "metadata": raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), dict) else {},
    }


def overlay_cold_start_hints(
    overlay: Dict[str, Any],
    task_type: Optional[str] = None,
    max_items: int = 3,
) -> List[Dict[str, str]]:
    hints: List[Dict[str, str]] = []
    task_type_norm = (task_type or "").lower()
    for skill in overlay.get("skills", []):
        types = [str(x).lower() for x in skill.get("task_types", [])]
        if types and "*" not in types and task_type_norm and task_type_norm not in types:
            continue
        text = " ".join(str(skill.get("text", "")).split())
        if not text:
            continue
        hints.append(
            {
                "id": str(skill.get("skill_id") or skill.get("id") or "overlay_skill"),
                "text": text,
                "trigger": "overlay_skill",
                "token_cost": str(len(text.split())),
            }
        )
        if len(hints) >= max_items:
            break
    return hints


def apply_harness_rules(
    rules: Iterable[Dict[str, Any]],
    context: Dict[str, Any],
    trigger: str,
) -> Optional[Dict[str, Any]]:
    for rule in rules or []:
        if rule.get("trigger") != trigger:
            continue
        try:
            if eval_condition(rule.get("condition"), context):
                return {"rule_id": rule.get("rule_id") or rule.get("id"), "effect": rule.get("effect") or {}}
        except Exception:
            continue
    return None


def eval_condition(cond: Any, context: Dict[str, Any]) -> bool:
    if not isinstance(cond, dict) or len(cond) != 1:
        return False
    op, value = next(iter(cond.items()))
    if op == "all":
        return all(eval_condition(x, context) for x in value or [])
    if op == "any":
        return any(eval_condition(x, context) for x in value or [])
    if op == "not":
        return not eval_condition(value, context)
    if op == "pred":
        return bool(_get_path(context, f"predicates.{value}"))
    if not isinstance(value, list) or len(value) != 2:
        return False
    left = _resolve_value(value[0], context)
    right = _resolve_value(value[1], context)
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "gt":
        return _num(left) > _num(right)
    if op == "gte":
        return _num(left) >= _num(right)
    if op == "lt":
        return _num(left) < _num(right)
    if op == "lte":
        return _num(left) <= _num(right)
    if op == "contains":
        if isinstance(left, (list, tuple, set)):
            return right in left
        return str(right).lower() in str(left).lower()
    if op == "startswith":
        return str(left).lower().startswith(str(right).lower())
    if op == "endswith":
        return str(left).lower().endswith(str(right).lower())
    return False


def _resolve_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith(("state.", "action.", "task.", "predicates.")):
        return _get_path(context, value)
    return value


def _get_path(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
