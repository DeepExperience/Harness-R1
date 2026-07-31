#!/usr/bin/env python3
"""Harness-R1 typed patch schema and compiler utilities.

The patch language is intentionally narrower than arbitrary Python edits.  It
lets a harness engineer model propose structured changes that can be validated,
compiled into AgentBench task configs, and audited before rollout.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ACTION_TYPES = {
    "set_config",
    "edit_tool_hint",
    "add_or_edit_skill",
    "add_guard_rule",
    "add_recovery_rule",
    "add_code_hook",
}

TOOLS = {
    "webshop": {"search_action", "click_action"},
    "alfworld": {"take_action"},
    "dbbench": {"execute_sql", "commit_final_answer"},
}

CONFIG_FIELDS: dict[str, dict[str, tuple[type, float | int | None, float | int | None]]] = {
    "webshop": {
        "enabled": (bool, None, None),
        "h2": (bool, None, None),
        "h3": (bool, None, None),
        "h4": (bool, None, None),
        "h5": (bool, None, None),
        "h2_click_similarity_threshold": (float, 0.0, 1.0),
        "h2_repeat_click_block_after": (int, 1, 10),
        "h4_search_loop_threshold": (int, 1, 20),
        "h4_duplicate_search_threshold": (int, 1, 10),
        "h4_product_stall_turns": (int, 1, 20),
        "h4_hint_max_words": (int, 10, 120),
        "h4_warn_threshold": (int, 1, 20),
        "h4_force_threshold": (int, 1, 20),
        "price_tolerance": (float, 0.0, 0.5),
        "h5_top_k": (int, 0, 5),
        "h5_cold_start_max_words": (int, 5, 120),
        "h5_score_threshold": (float, 0.0, 100.0),
    },
    "alfworld": {
        "enabled": (bool, None, None),
        "h2": (bool, None, None),
        "h3": (bool, None, None),
        "h4": (bool, None, None),
        "h5": (bool, None, None),
        "action_similarity_threshold": (float, 0.0, 1.0),
        "invalid_block_after": (int, 1, 10),
        "h3_max_words": (int, 5, 80),
        "h4_stall_window": (int, 2, 12),
        "h4_soft_intervention_rounds": (int, 0, 8),
        "h4_min_rounds_before_stall": (int, 1, 30),
        "h4_post_put_grace": (int, 0, 12),
        "h5_top_k": (int, 0, 5),
        "h5_cold_start_max_words": (int, 5, 120),
        "h5_step_hint_max_words": (int, 5, 80),
        "h4_nothing_happens_window": (int, 2, 12),
        "h4_nothing_happens_threshold": (int, 1, 10),
        "h2_empty_turn_threshold": (int, 1, 8),
        "h4_budget_warn_threshold": (int, 1, 30),
        "h4_budget_force_threshold": (int, 1, 20),
        "h4_budget_warn_threshold_multistep": (int, 1, 40),
    },
    "dbbench": {
        "enabled": (bool, None, None),
        "h2": (bool, None, None),
        "h3": (bool, None, None),
        "h4": (bool, None, None),
        "h5": (bool, None, None),
        "h2_repeat_sql_block_after": (int, 1, 10),
        "h4_stall_window": (int, 2, 12),
        "h4_empty_threshold": (int, 1, 10),
        "h4_budget_warn_threshold": (int, 1, 20),
        "h4_budget_force_threshold": (int, 1, 20),
        "h4_hint_max_words": (int, 5, 120),
        "h5_top_k": (int, 0, 5),
        "h5_cold_start_max_words": (int, 5, 120),
    },
}

TARGET_ALIASES = {
    "h2_enabled": "h2",
    "h3_enabled": "h3",
    "h4_enabled": "h4",
    "h5_enabled": "h5",
}
MODULE_CONFIG_FIELDS = {"enabled", "h2", "h3", "h4", "h5"}
PROMPT_CONFIG_FIELDS = {
    # In overlay-only WebShop patches, this is the only scalar config that
    # directly changes a predicate exposed to the model.
    "webshop": {"price_tolerance"},
    # ALFWorld patches should not tune the Life Harness H2/H4/H5 hand-written
    # policy thresholds. The first overlay-only action space is rule/text only.
    "alfworld": set(),
    # DBBench code-hook experiments use the same overlay-only rule: do not tune
    # the hand-written DBBench H2/H4/H5 policy thresholds through set_config.
    "dbbench": set(),
}

CONDITION_OPS = {"all", "any", "not", "eq", "ne", "gt", "gte", "lt", "lte", "contains", "startswith", "endswith", "pred"}
EFFECT_KINDS = {"block_and_prompt", "inject_hint", "force_action", "rewrite_action"}
TEXT_MAX_CHARS = 900
CODE_HOOK_MAX_CHARS = 8000
CODE_HOOKS = {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}

CONDITION_PATHS = {
    "webshop": {
        "action.tool",
        "action.value",
        "action.value_normalized",
        "action.final_action",
        "state.page_type",
        "state.has_search_bar",
        "state.clickables",
        "state.buy_now_available",
        "state.back_to_search_count",
        "state.duplicate_search_count",
        "state.product_stall_turns",
        "state.same_click_count",
        "state.remaining_steps",
        "state.current_price",
        "state.price_max",
        "task.task_type",
        "task.required_color",
        "task.required_size",
        "task.required_material",
    },
    "alfworld": {
        "action.raw",
        "action.normalized",
        "action.final_action",
        "action.in_admissible",
        "state.nothing_happens_count",
        "state.repeated_observation_count",
        "state.repeated_action_count",
        "state.invalid_action_count",
        "state.remaining_steps",
        "task.task_type",
        "task.target_type",
        "task.destination_type",
    },
    "dbbench": {
        "action.tool",
        "action.query",
        "action.answers",
        "action.h2_action",
        "action.h2_blocked_reason",
        "action.commit_gate_action",
        "action.commit_gate_reason",
        "state.sql_count",
        "state.last_sql",
        "state.last_result",
        "state.last_error_kind",
        "state.last_error_text",
        "state.last_result_was_error",
        "state.error_streak",
        "state.empty_streak",
        "state.text_only_streak",
        "state.loop_streak",
        "state.mutation_attempted",
        "state.candidate_answer",
        "state.candidate_answer_shape",
        "state.candidate_implausible",
        "state.remaining_rounds",
        "task.task_type",
        "task.answer_shape",
        "task.target_table",
    },
}

CONDITION_PATH_ALIASES = {
    "webshop": {
        "state.parsed_budget_constraint": "state.price_max",
        "state.budget": "state.price_max",
        "state.budget_max": "state.price_max",
        "state.max_price": "state.price_max",
    },
    "alfworld": {},
    "dbbench": {},
}

CONDITION_PREDICATES = {
    "webshop": {
        "required_options_unselected",
        "product_price_over_budget",
        "same_click_repeated",
        "duplicate_search_repeated",
        "search_loop_detected",
        "product_page_stalled",
        "buy_now_available",
        "search_not_available",
        "action_not_admissible",
    },
    "alfworld": {
        "action_not_admissible",
        "nothing_happens_repeated",
        "same_observation_repeated",
        "same_action_repeated",
        "remaining_steps_low",
    },
    "dbbench": {
        "execute_sql",
        "commit_final_answer",
        "no_sql_yet",
        "commit_before_sql",
        "commit_empty_answer",
        "mutation_task",
        "mutation_not_attempted",
        "last_result_error",
        "last_result_empty",
        "unknown_column_error",
        "syntax_error",
        "repeated_sql",
        "candidate_answer_available",
        "remaining_rounds_low",
        "text_only_loop",
    },
    }

FORBIDDEN_TEXT_PATTERNS = [
    re.compile(r"\b[bB]0[0-9a-zA-Z]{8}\b"),  # WebShop ASIN-like IDs
    re.compile(r"\b(?:task|index|sample)\s*#?\s*\d{1,5}\b", re.IGNORECASE),
    re.compile(r"\bwebshop[-_\s]*\d{1,5}\b", re.IGNORECASE),
    re.compile(r"\balfworld[-_\s]*\d{1,5}\b", re.IGNORECASE),
]

# ALFWorld admissible commands often include instance ids, e.g.
# "go to sinkbasin 1" or "take egg 2". Letting a reusable patch force/rewrite
# those strings encourages copying batch-specific trajectories. Tool hints and
# skills may mention generic task concepts; harness-executed action effects may
# not target numbered instances.
ALFWORLD_INSTANCE_ACTION_RE = re.compile(
    r"\b(?:go to|take|pick up|put|open|close|toggle|clean|heat|cool|slice|examine|use)\b.*\b\d+\b",
    re.IGNORECASE,
)


class PatchValidationError(ValueError):
    pass


def _prompt_config_fields(bench: str) -> set[str]:
    return PROMPT_CONFIG_FIELDS.get(
        bench,
        {name for name in CONFIG_FIELDS[bench] if name not in MODULE_CONFIG_FIELDS},
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from <patch>, plain text, or a fenced code block."""
    text = text.strip()

    def candidates(segment: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", segment):
            try:
                obj, _ = decoder.raw_decode(segment[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                objects.append(obj)
        return objects

    def choose(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
        patch_like = [
            obj
            for obj in objects
            if isinstance(obj.get("actions"), list) and isinstance(obj.get("benchmark"), str)
        ]
        if patch_like:
            return patch_like[-1]
        if objects:
            return objects[-1]
        return None

    def patch_blocks(segment: str) -> list[str]:
        blocks = [
            match.group(1)
            for match in re.finditer(r"<patch>\s*(.*?)\s*</patch>", segment, re.DOTALL | re.IGNORECASE)
        ]
        if not blocks:
            match = re.search(r"<patch>\s*(.*)", segment, re.DOTALL | re.IGNORECASE)
            if match:
                blocks.append(match.group(1))
        return blocks

    # Relax/SGLang reward paths can pass the full rendered chat, including the
    # prompt schema examples.  Prefer the final assistant segment when present,
    # then prefer an explicit <patch> block inside that segment.
    segments: list[str] = []
    assistant_tag = "<|im_start|>assistant"
    search_segment = text
    if assistant_tag in text:
        search_segment = text.rsplit(assistant_tag, 1)[-1]
    segments.extend(patch_blocks(search_segment))
    segments.append(search_segment)
    segments.extend(match.group(1) for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", search_segment, re.DOTALL))
    if search_segment != text:
        segments.append(text)

    for segment in segments:
        obj = choose(candidates(segment))
        if obj is not None:
            return obj
    raise PatchValidationError("no JSON object found in model output")


def extract_think_patch_json_object(text: str) -> dict[str, Any]:
    """Extract patch JSON only from a complete <think>...</think><patch>...</patch> response."""
    text = text.strip()
    assistant_tag = "<|im_start|>assistant"
    if assistant_tag in text:
        text = text.rsplit(assistant_tag, 1)[-1]

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match is None:
        raise PatchValidationError("missing complete <think>...</think> block")
    patch_matches = list(re.finditer(r"<patch>\s*(.*?)\s*</patch>", text, re.DOTALL | re.IGNORECASE))
    if not patch_matches:
        raise PatchValidationError("missing complete <patch>...</patch> block")
    patch_text = patch_matches[-1].group(1).strip()
    if not patch_text:
        raise PatchValidationError("<patch> block is empty")
    return extract_json_object(f"<patch>\n{patch_text}\n</patch>")


def extract_prefilled_think_patch_json_object(text: str) -> dict[str, Any]:
    """Extract patch JSON from a Qwen3.5 completion whose prompt already opened <think>."""
    text = text.strip()
    assistant_tag = "<|im_start|>assistant"
    if assistant_tag in text:
        text = text.rsplit(assistant_tag, 1)[-1].lstrip()
    if re.match(r"(?is)^<think\b", text):
        raise PatchValidationError("prefilled-think response must not start with another <think> tag")
    return extract_think_patch_json_object("<think>\n" + text)


def load_patch(path: Path, bench: str | None = None) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return normalize_patch(raw, bench=bench)


def require_code_hook_only_patch(patch: dict[str, Any]) -> None:
    """Reject legacy DSL actions in code-hook-only experiment protocols."""
    if not isinstance(patch, dict):
        raise PatchValidationError("patch must be a JSON object")
    actions = patch.get("actions")
    if not isinstance(actions, list) or not actions:
        raise PatchValidationError("patch.actions must be a non-empty list")
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            raise PatchValidationError(f"action {idx} must be an object")
        if action.get("type") != "add_code_hook":
            raise PatchValidationError(
                f"action {idx} type {action.get('type')!r} is not allowed in code-hook-only protocol"
            )


def normalize_patch(raw: dict[str, Any], bench: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PatchValidationError("patch must be a JSON object")
    patch = copy.deepcopy(raw)
    benchmark = str(patch.get("benchmark") or bench or "").lower()
    if benchmark not in CONFIG_FIELDS:
        raise PatchValidationError(f"unsupported benchmark: {benchmark!r}")
    actions = patch.get("actions")
    if not isinstance(actions, list) or not actions:
        raise PatchValidationError("patch.actions must be a non-empty list")
    if len(actions) > 12:
        raise PatchValidationError("patch.actions has more than 12 actions")
    normalized = {
        "schema_version": "harness-r1-patch-v1",
        "benchmark": benchmark,
        "description": _clean_text(str(patch.get("description", "")), field="description", required=False),
        "actions": [],
    }
    for idx, action in enumerate(actions):
        normalized["actions"].append(_normalize_action(benchmark, action, idx))
    return normalized


def _normalize_action(bench: str, action: Any, idx: int) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise PatchValidationError(f"action {idx} must be an object")
    typ = action.get("type")
    if typ not in ACTION_TYPES:
        raise PatchValidationError(f"action {idx} has unsupported type: {typ!r}")
    if typ == "set_config":
        return _normalize_set_config(bench, action)
    if typ == "edit_tool_hint":
        return _normalize_tool_hint(bench, action)
    if typ == "add_or_edit_skill":
        return _normalize_skill(bench, action)
    if typ == "add_guard_rule":
        return _normalize_rule(bench, action, kind="guard")
    if typ == "add_recovery_rule":
        return _normalize_rule(bench, action, kind="recovery")
    if typ == "add_code_hook":
        return _normalize_code_hook(bench, action)
    raise AssertionError(typ)


def _compile_code_hook_for_validation(source: str, bench: str) -> None:
    agentbench_root = Path(__file__).resolve().parents[1]
    if str(agentbench_root) not in sys.path:
        sys.path.insert(0, str(agentbench_root))
    try:
        from src.server.harness.code_runner import HookCompileError, compile_hook

        compile_hook(source, benchmark=bench)
    except HookCompileError as exc:
        raise PatchValidationError(str(exc)) from exc
    except Exception as exc:
        raise PatchValidationError(f"code hook failed sandbox validation: {exc}") from exc


def _normalize_code_hook(bench: str, action: dict[str, Any]) -> dict[str, Any]:
    if bench not in {"alfworld", "webshop", "dbbench"}:
        raise PatchValidationError("add_code_hook is supported only for ALFWorld, WebShop, and DBBench")
    hook = str(action.get("hook") or "").strip()
    if hook not in CODE_HOOKS:
        raise PatchValidationError(f"unsupported code hook: {hook!r}")
    code = action.get("code")
    if not isinstance(code, str) or not code.strip():
        raise PatchValidationError("code hook must include non-empty code")
    if len(code) > CODE_HOOK_MAX_CHARS:
        raise PatchValidationError(f"code hook exceeds {CODE_HOOK_MAX_CHARS} chars")
    _compile_code_hook_for_validation(code, bench)
    return {"type": "add_code_hook", "hook": hook, "code": code}


def _field_from_target(bench: str, target: str) -> str:
    field = str(target or "").strip()
    if "." in field:
        prefix, field = field.split(".", 1)
        if prefix not in {bench, "harness", "config"}:
            raise PatchValidationError(f"config target prefix {prefix!r} does not match {bench}")
    field = TARGET_ALIASES.get(field, field)
    if field not in CONFIG_FIELDS[bench]:
        raise PatchValidationError(f"unsupported config target for {bench}: {target!r}")
    if field in MODULE_CONFIG_FIELDS:
        raise PatchValidationError(
            f"{field!r} cannot be changed by a Harness-R1 patch; "
            "patches edit only the overlay substrate, not built-in harness modules"
        )
    if field not in _prompt_config_fields(bench):
        raise PatchValidationError(
            f"{field!r} is not an exposed config target for {bench}; "
            "patches may only set config fields listed in the Harness-R1 prompt"
        )
    return field


def _normalize_set_config(bench: str, action: dict[str, Any]) -> dict[str, Any]:
    field = _field_from_target(bench, str(action.get("target") or action.get("field") or ""))
    expected, min_value, max_value = CONFIG_FIELDS[bench][field]
    value = action.get("value")
    if expected is bool:
        if not isinstance(value, bool):
            raise PatchValidationError(f"{field} expects boolean")
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PatchValidationError(f"{field} expects integer")
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PatchValidationError(f"{field} expects number")
        value = float(value)
    if min_value is not None and value < min_value:
        raise PatchValidationError(f"{field} below minimum {min_value}")
    if max_value is not None and value > max_value:
        raise PatchValidationError(f"{field} above maximum {max_value}")
    return {"type": "set_config", "target": field, "value": value}


def _normalize_tool_hint(bench: str, action: dict[str, Any]) -> dict[str, Any]:
    tool = str(action.get("tool") or "").strip()
    if tool not in TOOLS[bench]:
        raise PatchValidationError(f"unsupported tool for {bench}: {tool!r}")
    operation = str(action.get("operation") or "append").strip()
    if operation not in {"append", "prepend", "replace"}:
        raise PatchValidationError(f"unsupported edit_tool_hint operation: {operation!r}")
    return {
        "type": "edit_tool_hint",
        "tool": tool,
        "operation": operation,
        "text": _clean_text(action.get("text", ""), field="edit_tool_hint.text"),
    }


def _normalize_skill(bench: str, action: dict[str, Any]) -> dict[str, Any]:
    skill_id = str(action.get("skill_id") or action.get("id") or "").strip()
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,80}", skill_id):
        raise PatchValidationError(f"invalid skill_id: {skill_id!r}")
    task_types = action.get("task_types") or ["*"]
    keywords = action.get("keywords") or []
    if not isinstance(task_types, list) or len(task_types) > 16:
        raise PatchValidationError("skill.task_types must be a list of <=16 strings")
    if not isinstance(keywords, list) or len(keywords) > 32:
        raise PatchValidationError("skill.keywords must be a list of <=32 strings")
    return {
        "type": "add_or_edit_skill",
        "skill_id": skill_id,
        "task_types": [_clean_token(x, "task_type") for x in task_types],
        "keywords": [_clean_keyword(x) for x in keywords],
        "text": _clean_text(action.get("text", ""), field="skill.text"),
    }


def _normalize_rule(bench: str, action: dict[str, Any], kind: str) -> dict[str, Any]:
    rule_id = str(action.get("rule_id") or action.get("id") or "").strip()
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,80}", rule_id):
        raise PatchValidationError(f"invalid rule_id: {rule_id!r}")
    trigger_default = "before_action" if kind == "guard" else "post_step"
    trigger = str(action.get("trigger") or trigger_default).strip()
    allowed_triggers = _allowed_rule_triggers(bench, kind)
    if trigger not in allowed_triggers:
        raise PatchValidationError(f"unsupported {kind} trigger: {trigger!r}")
    condition = _normalize_condition(bench, action.get("condition"), depth=0)
    effect = _normalize_effect(
        action.get("effect") or {},
        bench=bench,
        rule_kind=kind,
    )
    out_type = "add_guard_rule" if kind == "guard" else "add_recovery_rule"
    return {
        "type": out_type,
        "rule_id": rule_id,
        "trigger": trigger,
        "condition": condition,
        "effect": effect,
    }


def _allowed_rule_triggers(bench: str, kind: str) -> set[str]:
    if kind == "guard":
        return {"before_action"}
    # WebShop has a budget_check runtime hook. ALFWorld currently evaluates
    # overlay recovery rules only after env.step, so exposing budget_check would
    # validate patches that never execute.
    if bench == "webshop":
        return {"post_step", "budget_check"}
    return {"post_step"}


def _allowed_effect_kinds(bench: str, rule_kind: str) -> set[str]:
    if rule_kind == "guard":
        return {"block_and_prompt", "force_action", "rewrite_action"}
    if bench in {"webshop", "alfworld"}:
        return {"inject_hint", "force_action"}
    return EFFECT_KINDS


def _normalize_effect(effect: dict[str, Any], bench: str, rule_kind: str) -> dict[str, Any]:
    if not isinstance(effect, dict):
        raise PatchValidationError("rule.effect must be an object")
    kind = str(effect.get("kind") or "").strip()
    if kind not in EFFECT_KINDS:
        raise PatchValidationError(f"unsupported effect kind: {kind!r}")
    allowed_kinds = _allowed_effect_kinds(bench, rule_kind)
    if kind not in allowed_kinds:
        raise PatchValidationError(
            f"effect kind {kind!r} is not supported for {bench} {rule_kind} rules"
        )
    out: dict[str, Any] = {"kind": kind}
    if kind in {"block_and_prompt", "inject_hint"}:
        out["message"] = _clean_text(effect.get("message", ""), field="effect.message")
    if kind in {"force_action", "rewrite_action"}:
        out["action"] = _clean_text(effect.get("action", ""), field="effect.action")
        if bench == "alfworld":
            _validate_alfworld_effect_action(out["action"])
        if effect.get("message"):
            out["message"] = _clean_text(effect.get("message", ""), field="effect.message", required=False)
    return out


def _validate_alfworld_effect_action(action: str) -> None:
    action_norm = " ".join(str(action or "").lower().split())
    if not action_norm:
        raise PatchValidationError("ALFWorld force/rewrite action must be non-empty")
    if ALFWORLD_INSTANCE_ACTION_RE.search(action_norm):
        raise PatchValidationError(
            "ALFWorld force/rewrite actions may not contain numbered object/location "
            "instances such as 'sinkbasin 1' or 'egg 2'; use inject_hint or a broad "
            "guard instead"
        )


def _normalize_condition(bench: str, cond: Any, depth: int) -> dict[str, Any]:
    if depth > 8:
        raise PatchValidationError("condition nesting is too deep")
    if not isinstance(cond, dict) or len(cond) != 1:
        raise PatchValidationError("condition must be an object with exactly one operator")
    op, value = next(iter(cond.items()))
    if op not in CONDITION_OPS:
        raise PatchValidationError(f"unsupported condition operator: {op!r}")
    if op in {"all", "any"}:
        if not isinstance(value, list) or not value:
            raise PatchValidationError(f"condition.{op} must be a non-empty list")
        return {op: [_normalize_condition(bench, child, depth + 1) for child in value]}
    elif op == "not":
        return {op: _normalize_condition(bench, value, depth + 1)}
    elif op == "pred":
        if not isinstance(value, str) or not value:
            raise PatchValidationError("condition.pred must be a predicate name")
        if value not in CONDITION_PREDICATES[bench]:
            raise PatchValidationError(f"unsupported predicate for {bench}: {value!r}")
        return {op: value}
    else:
        if not isinstance(value, list) or len(value) != 2:
            raise PatchValidationError(f"condition.{op} must be a two-item list")
        normalized_value = []
        aliases = CONDITION_PATH_ALIASES.get(bench, {})
        for operand in value:
            if isinstance(operand, str):
                operand = aliases.get(operand, operand)
            if isinstance(operand, str) and operand.startswith(("action.", "state.", "task.")):
                if operand not in CONDITION_PATHS[bench]:
                    raise PatchValidationError(f"unsupported condition path for {bench}: {operand!r}")
            normalized_value.append(operand)
        return {op: normalized_value}


def _clean_text(value: Any, field: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise PatchValidationError(f"{field} must be a string")
    text = " ".join(value.strip().split())
    if required and not text:
        raise PatchValidationError(f"{field} cannot be empty")
    if len(text) > TEXT_MAX_CHARS:
        raise PatchValidationError(f"{field} exceeds {TEXT_MAX_CHARS} chars")
    for pat in FORBIDDEN_TEXT_PATTERNS:
        if pat.search(text):
            raise PatchValidationError(f"{field} appears to contain task-specific leakage")
    return text


def _clean_token(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise PatchValidationError(f"{field} must be a string")
    token = value.strip().lower()
    if token != "*" and not re.fullmatch(r"[a-z0-9_.:-]{1,64}", token):
        raise PatchValidationError(f"invalid {field}: {value!r}")
    return token


def _clean_keyword(value: Any) -> str:
    if not isinstance(value, str):
        raise PatchValidationError("keyword must be a string")
    token = re.sub(r"\s+", "_", value.strip().lower())
    if token != "*" and not re.fullmatch(r"[a-z0-9_.:-]{1,64}", token):
        raise PatchValidationError(f"invalid keyword: {value!r}")
    return token


def compile_patch_to_task_definition(
    task_definition: dict[str, Any],
    patch: dict[str, Any] | None,
    bench: str,
) -> dict[str, Any]:
    """Return a copy of an AgentBench task definition with patch actions applied."""
    if not patch:
        return copy.deepcopy(task_definition)
    patch = normalize_patch(patch, bench=bench)
    out = copy.deepcopy(task_definition)
    if len(out) != 1:
        raise PatchValidationError("task definition must contain exactly one task")
    task_name = next(iter(out))
    params = out[task_name].setdefault("parameters", {})
    overlay = copy.deepcopy(params.get("harness_overlay") or {})
    overlay.setdefault("skills", [])
    overlay.setdefault("guard_rules", [])
    overlay.setdefault("recovery_rules", [])
    overlay.setdefault("code_hooks", [])
    overlay.setdefault("metadata", {})
    overlay["metadata"]["source_patch_description"] = patch.get("description", "")
    overlay["metadata"]["overlay_only"] = True
    for action in patch["actions"]:
        typ = action["type"]
        if typ == "set_config":
            params[action["target"]] = action["value"]
        elif typ == "edit_tool_hint":
            _apply_tool_hint(params, action)
        elif typ == "add_or_edit_skill":
            overlay["skills"] = _upsert_by_id(overlay["skills"], action, id_key="skill_id")
        elif typ == "add_guard_rule":
            overlay["guard_rules"] = _upsert_by_id(overlay["guard_rules"], action, id_key="rule_id")
        elif typ == "add_recovery_rule":
            overlay["recovery_rules"] = _upsert_by_id(overlay["recovery_rules"], action, id_key="rule_id")
        elif typ == "add_code_hook":
            overlay["code_hooks"].append(
                {"type": "add_code_hook", "hook": action["hook"], "code": action["code"]}
            )
    if any(overlay.get(key) for key in ("skills", "guard_rules", "recovery_rules", "code_hooks")):
        params["harness_overlay"] = overlay
    return out


def _apply_tool_hint(params: dict[str, Any], action: dict[str, Any]) -> None:
    tools = params.get("tools") or []
    for tool in tools:
        fn = tool.get("function") or {}
        if fn.get("name") != action["tool"]:
            continue
        old = str(fn.get("description") or "")
        text = action["text"]
        if action["operation"] == "append":
            new = (old.rstrip() + " " + text).strip()
        elif action["operation"] == "prepend":
            new = (text.rstrip() + " " + old.lstrip()).strip()
        else:
            new = text
        fn["description"] = new
        tool["function"] = fn
        return
    raise PatchValidationError(f"tool not found in task definition: {action['tool']}")


def _upsert_by_id(items: list[dict[str, Any]], item: dict[str, Any], id_key: str) -> list[dict[str, Any]]:
    result = [x for x in items if x.get(id_key) != item.get(id_key)]
    result.append(copy.deepcopy(item))
    return result


def schema_prompt(
    bench: str,
    response_protocol: str = "full_think_patch",
    schema_style: str = "example",
) -> str:
    bench = bench.lower()
    if bench not in CONFIG_FIELDS:
        raise PatchValidationError(f"unsupported benchmark: {bench}")
    if response_protocol not in {"full_think_patch", "prefill_think_patch"}:
        raise PatchValidationError(f"unsupported response protocol: {response_protocol!r}")
    if schema_style not in {
        "example",
        "schema_only",
        "qwen25_sft",
        "webshop_code_only",
        "webshop_action_balanced_v1",
        "webshop_life_multihook_v1",
        "dbbench_life_multihook_v1",
        "alfworld_life_rubric",
        "alfworld_life_template",
        "alfworld_life_onehook",
        "alfworld_life_onehook_v2",
        "alfworld_life_twohook",
        "alfworld_life_multihook_v1",
    }:
        raise PatchValidationError(f"unsupported schema style: {schema_style!r}")
    if schema_style in {
        "webshop_code_only",
        "webshop_action_balanced_v1",
        "webshop_life_multihook_v1",
    } and bench != "webshop":
        raise PatchValidationError(f"schema_style={schema_style!r} is supported only for WebShop")
    if schema_style == "dbbench_life_multihook_v1" and bench != "dbbench":
        raise PatchValidationError(f"schema_style={schema_style!r} is supported only for DBBench")
    prompt_fields = _prompt_config_fields(bench)
    fields = "\n".join(
        f"- {name}: {spec[0].__name__}"
        for name, spec in CONFIG_FIELDS[bench].items()
        if name in prompt_fields
    )
    tools = ", ".join(sorted(TOOLS[bench]))
    fields_text = fields or f"- none; set_config is not available for {bench} overlay-only patches."
    set_config_required = (
        "- set_config: type, target, value"
        if prompt_fields
        else f"- set_config: not available for {bench}; do not use"
    )
    set_config_grammar = (
        """
SET_CONFIG ::= {
  "type": "set_config",
  "target": CONFIG_TARGET,
  "value": JSON_SCALAR
}
""".strip()
        if prompt_fields
        else ""
    )
    code_hook_available = bench in {"alfworld", "webshop", "dbbench"}
    action_grammar_union = (
        "SET_CONFIG | EDIT_TOOL_HINT | ADD_OR_EDIT_SKILL |\n"
        "           ADD_GUARD_RULE | ADD_RECOVERY_RULE"
        if prompt_fields
        else "EDIT_TOOL_HINT | ADD_OR_EDIT_SKILL | ADD_GUARD_RULE | ADD_RECOVERY_RULE"
    )
    if code_hook_available:
        action_grammar_union = action_grammar_union + " | ADD_CODE_HOOK"
    add_code_hook_required = (
        "- add_code_hook: type, hook in {on_init, make_pre_hint, on_before_action, on_post_step}, code\n"
        if code_hook_available
        else ""
    )
    add_code_hook_grammar = (
        """
ADD_CODE_HOOK ::= {
  "type": "add_code_hook",
  "hook": "on_init" | "make_pre_hint" | "on_before_action" | "on_post_step",
  "code": "def helper(...):\n    ...\n\ndef hook(ctx, nb):\n    ...\n    return {...}"
}
""".strip()
        if code_hook_available
        else ""
    )
    if code_hook_available and bench == "alfworld":
        code_hook_notes = """
ALFWorld code hooks:
- code must be no more than 8000 characters, define exactly one top-level
  def hook(ctx, nb), plus at most 5 top-level helper functions with no imports
  or global state. Keep the AST simple; do not build large helper forests.
- ctx is read-only evidence for the current episode; nb is a per-episode scratch dict.
- ctx["world"] is a read-only Life-Harness world-model fact snapshot:
  current_location, inventory, object_at, visited, unvisited, target_found,
  target_location, placed_count, placed_items, placed_locations, lamp_location.
- ctx["task"] includes task_type, target_type, destination_type. Use
  ctx["observation"] and nb to maintain your own stage/subgoal state; planner
  outputs and built-in subgoals are not exposed.
- When ctx["world"] facts are enough to choose a safe next action, prefer a
  narrow force_action or rewrite_action selected from ctx["admissible"] over a
  generic hint.
- on_init may return {"skills":[{"text": TEXT}], "tool_hint": TEXT}.
- make_pre_hint may return {"message": TEXT}.
- on_before_action may return block_and_prompt, rewrite_action, or force_action.
- on_post_step may return inject_hint or force_action.
- Available ctx keys include observation, admissible, output, step, max_step,
  remaining_steps, action.*, state.*, task.*, predicates.*, world.*.
- Code may use basic Python, re, math, and SequenceMatcher. try/except is
  allowed only as "except Exception"; do not use else/finally.
- Do not import, use while loops, open files, use eval/exec, access names beginning with
  underscore, or hard-code numbered actions such as "go to sinkbasin 1";
  choose from ctx["admissible"] when an exact action is needed.
""".strip()
    elif code_hook_available and bench == "webshop":
        code_hook_notes = """
WebShop code hooks:
- Keep total code compact: target <=6000 characters and hard max 8000
  characters. Code must define exactly one top-level def hook(ctx, nb), plus
  at most 5 top-level helper functions with no imports or global state. Keep
  the AST simple; do not build large helper forests.
- ctx is read-only evidence for the current episode; nb is a per-episode scratch dict.
- Available ctx keys include observation, step, max_step, remaining_steps,
  action.*, state.*, task.*, predicates.*, and webshop.*.
- You may read ctx either as dictionaries, e.g. ctx["state"]["page_type"], or
  as attributes, e.g. ctx.state.page_type.
- state.page_type summarizes the current page; state.clickables is the
  lowercase clickable list; state.current_price and state.price_max expose
  budget evidence when available.
- action.tool is exactly "search_action" or "click_action"; action.value is
  the raw search query or click value, and action.value_normalized is lowercase.
- task.required_color, task.required_size, and task.required_material expose
  parsed product requirements when available.
- webshop.search_queries, webshop.asins_visited, webshop.current_asin,
  webshop.selected_attributes, and webshop.attribute_options expose a
  read-only WebShop state snapshot.
- on_init may return {"skills":[{"text": TEXT}], "tool_hint": TEXT}.
- make_pre_hint may return {"message": TEXT}.
- on_before_action may return block_and_prompt, rewrite_action, or force_action.
- on_post_step may return inject_hint or force_action.
- For on_before_action, action strings may be raw values such as "buy now" or
  complete tool actions such as "click[buy now]" / "search[query]"; the
  runtime wraps raw values according to the current tool.
- For on_post_step force_action, prefer complete click[...] actions or raw
  clickable values. Use force_action only for narrow, evidence-backed fixes.
- Code may use basic Python, re, math, and SequenceMatcher. try/except is
  allowed only as "except Exception"; do not use else/finally.
- Do not use generic reflection helpers or forbidden builtins such as getattr,
  setattr, type, dir, vars, eval, exec, or open; access ctx fields directly.
- Helper functions cannot see local hook variables. If a helper needs ctx or
  nb, pass them explicitly, e.g. helper(ctx, value), not helper().
- Do not import, use while loops, open files, use eval/exec, access names
  beginning with underscore, or hard-code product IDs, exact product titles,
  ASINs, task indices, or test-set answers.
""".strip()
    elif code_hook_available and bench == "dbbench":
        code_hook_notes = """
DBBench code hooks:
- Keep total code compact: target <=6000 characters and hard max 8000
  characters. Code must define exactly one top-level def hook(ctx, nb), plus
  at most 5 top-level helper functions with no imports or global state.
- Do not define helper functions inside hook; nested functions are not allowed.
- Helper function names and local variable names must not begin with "_".
- ctx is read-only evidence for the current episode; nb is a per-episode scratch dict.
- Available ctx keys include action.*, state.*, task.*, predicates.*, and dbbench.*.
- action.tool is exactly "execute_sql" or "commit_final_answer"; action.query
  is the raw SQL string, and action.answers is the submitted answer list.
- state exposes runtime observations such as sql_count, last_sql, last_result,
  last_error_kind, last_error_text, last_result_was_error, error_streak,
  empty_streak, text_only_streak, loop_streak, mutation_attempted,
  candidate_answer, candidate_answer_shape, candidate_implausible, and
  remaining_rounds. Treat candidate_answer as a noisy SQL-derived hint, not as
  an answer oracle.
- task exposes task_type, answer_shape, target_table, and description from the
  task/schema parser.
- dbbench exposes sql_history, sql_history_raw, discovered_columns,
  db_response, and round.
- predicates exposes narrow boolean signals such as no_sql_yet,
  commit_before_sql, commit_empty_answer, mutation_task,
  mutation_not_attempted, last_result_error, last_result_empty,
  unknown_column_error, syntax_error, repeated_sql,
  candidate_answer_available, remaining_rounds_low, and text_only_loop.
- state.last_error_kind is one of runtime-derived labels such as unknown_col,
  syntax, empty, null_agg, or a blank string; use predicates for narrow checks.
- on_init may return {"skills":[{"text": TEXT}], "tool_hint": TEXT}.
- make_pre_hint may return {"message": TEXT}.
- on_before_action may return block_and_prompt only.
- on_post_step may return inject_hint only, or update nb and return None.
- DBBench v1 intentionally ignores rewrite_action and force_action because
  generic action rewrites are unsafe for SQL strings and literal casing.
- Code may use basic Python, re, math, and SequenceMatcher. try/except is
  allowed only as "except Exception"; do not use else/finally.
- Do not import, use while loops, open files, use eval/exec, access names
  beginning with underscore, or hard-code table cell values, exact final
  answers, task indices, or ground-truth SQL copied from traces.
""".strip()
    else:
        code_hook_notes = ""
    json_scalar_note = (
        "- JSON_SCALAR means a string, number, boolean, or null accepted by the listed\n"
        "  config target.\n"
        if prompt_fields
        else ""
    )
    config_target_note = (
        "- CONFIG_TARGET must be one of the listed allowed config targets below."
        if prompt_fields
        else "- CONFIG_TARGET is not available for this benchmark; do not use set_config."
    )
    config_value_note = (
        "For WebShop, price_tolerance must be a number from 0.0 to 0.5."
        if bench == "webshop"
        else f"For {bench}, no config target is exposed in the overlay-only action space."
    )
    if bench == "webshop":
        contains_example = '- {{"contains": ["state.clickables", "buy now"]}}, startswith, endswith'
        valid_example = (
            '{"benchmark":"webshop","description":"Block premature buy when required options are still unselected.",'
            '"actions":[{"type":"add_guard_rule","rule_id":"block_buy_missing_options",'
            '"trigger":"before_action","condition":{"all":[{"eq":["action.tool","click_action"]},'
            '{"eq":["action.value_normalized","buy now"]},{"pred":"required_options_unselected"}]},'
            '"effect":{"kind":"block_and_prompt","message":"Select required product options before buying."}}]}'
        )
        bench_specific_rules = (
            "- Do not mention task indices, ASINs, exact product titles, or test-set answers."
        )
    elif bench == "alfworld":
        contains_example = '- {{"contains": ["action.normalized", "look"]}}, startswith, endswith'
        valid_example = (
            '{"benchmark":"alfworld","description":"Block actions that are not listed as available.",'
            '"actions":[{"type":"add_guard_rule","rule_id":"block_non_admissible_action",'
            '"trigger":"before_action","condition":{"pred":"action_not_admissible"},'
            '"effect":{"kind":"block_and_prompt","message":"Choose an action exactly from AVAILABLE ACTIONS."}}]}'
        )
        bench_specific_rules = (
            "- Do not mention task indices, exact task answers, or numbered object/location instances.\n"
            "- For ALFWorld, force_action and rewrite_action must not contain numbered instances such as "
            "'sinkbasin 1' or 'egg 2'; prefer inject_hint for reusable recovery."
        )
    else:
        contains_example = '- {{"contains": ["state.last_result", "Unknown column"]}}, startswith, endswith'
        valid_example = (
            '{"benchmark":"dbbench","description":"Block empty commit before any SQL.",'
            '"actions":[{"type":"add_code_hook","hook":"on_before_action",'
            '"code":"def hook(ctx, nb):\\n    if ctx.predicates.commit_before_sql:\\n        return {'
            '\\"kind\\":\\"block_and_prompt\\",\\"message\\":\\"Run SQL before committing.\\"}\\n    return None"}]}'
        )
        bench_specific_rules = (
            "- Do not mention task indices, exact final answers, table cell values, or ground-truth SQL."
        )
    if bench == "webshop":
        variables = """
Paths:
- action.tool, action.value, action.value_normalized, action.final_action
- state.page_type in {home, search_results, product_detail, unknown}
- state.has_search_bar, state.buy_now_available, state.clickables
- state.back_to_search_count, state.duplicate_search_count, state.product_stall_turns
- state.same_click_count, state.remaining_steps, state.current_price, state.price_max
- task.task_type, task.required_color, task.required_size, task.required_material

Predicates:
required_options_unselected, product_price_over_budget, same_click_repeated,
duplicate_search_repeated, search_loop_detected, product_page_stalled,
buy_now_available, search_not_available, action_not_admissible.

Notes: use state.price_max/current_price for budget. required_options_unselected
is checked for buy-now option omissions; product_price_over_budget uses
price_tolerance.
"""
    elif bench == "alfworld":
        variables = """
Paths:
- action.raw, action.normalized, action.final_action, action.in_admissible
- state.nothing_happens_count, state.repeated_observation_count
- state.repeated_action_count, state.invalid_action_count, state.remaining_steps
- task.task_type, task.target_type, task.destination_type

Predicates:
action_not_admissible, nothing_happens_repeated, same_observation_repeated,
same_action_repeated, remaining_steps_low.
"""
    else:
        variables = """
Paths:
- action.tool in {execute_sql, commit_final_answer}
- action.query, action.answers, action.h2_action, action.h2_blocked_reason
- state.sql_count, state.last_sql, state.last_result, state.last_error_kind
- state.error_streak, state.empty_streak, state.text_only_streak, state.loop_streak
- state.mutation_attempted, state.candidate_answer, state.candidate_answer_shape
- state.candidate_implausible, state.remaining_rounds
- task.task_type, task.answer_shape, task.target_table, task.description
- dbbench.sql_history, dbbench.sql_history_raw, dbbench.discovered_columns

Predicates:
execute_sql, commit_final_answer, no_sql_yet, commit_before_sql,
commit_empty_answer, mutation_task, mutation_not_attempted, last_result_error,
last_result_empty, unknown_column_error, syntax_error, repeated_sql,
candidate_answer_available, remaining_rounds_low, text_only_loop.
"""
    if response_protocol == "prefill_think_patch":
        output_contract = """
After your reasoning, output exactly one <patch> block containing a single
JSON patch object with top-level keys benchmark, description, and actions.
Output nothing after the </patch> block.
"""
    else:
        output_contract = """
Output exactly two XML-style blocks, in this order:
- A <think> block with concise recurring-failure analysis.
- A <patch> block containing exactly one valid JSON object with keys
  benchmark, description, actions.
"""

    if bench == "webshop" and schema_style == "webshop_action_balanced_v1":
        base = schema_prompt(
            bench,
            response_protocol=response_protocol,
            schema_style="webshop_code_only",
        )
        mechanism_policy = """
Mechanism selection policy:
- Choose hooks by the causal point at which the recurring failure becomes
  observable. Do not default every failure to on_before_action.
- on_init is for a short task-level shopping skill or tool hint that is useful
  before the first action.
- make_pre_hint is for state-dependent, deduplicated guidance immediately
  before the agent chooses its next action.
- on_before_action is for a narrow correction or safety guard on the current
  proposed action. It should not be used as a generic shopping policy.
- on_post_step is for updating nb from the observed transition and recovering
  from loops, stalls, or failed progress after an action has executed.
- Use the least invasive mechanism that addresses the evidence. A patch may
  contain 1 to 3 hooks, each hook at most once; omit unsupported hooks.
- When multiple hooks are needed, use nb to share only compact stage or
  deduplication state. Avoid repeated hints and broad unconditional effects.
""".strip()
        marker = "\nRules:\n"
        if marker not in base:
            raise PatchValidationError("WebShop code-only schema has no Rules marker")
        return base.replace(marker, f"\n\n{mechanism_policy}{marker}", 1)

    if bench == "webshop" and schema_style == "webshop_code_only":
        return f"""
{output_contract}

Do not copy grammar placeholder tokens.

Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "webshop"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Active WebShop action space for this experiment: use add_code_hook only.
- Legacy DSL actions are sleeping in this prompt. Do not use set_config,
  edit_tool_hint, add_or_edit_skill, add_guard_rule, or add_recovery_rule.

ACTION ::= ADD_CODE_HOOK
{add_code_hook_grammar}

Hook return contracts:
- on_init returns {{"skills": [{{"text": TEXT}}, ...], "tool_hint": TEXT}} or None.
- make_pre_hint returns {{"message": TEXT}} or None.
- on_before_action returns
  {{"kind": "block_and_prompt", "message": TEXT}},
  {{"kind": "rewrite_action", "action": TEXT, "message": TEXT}}, or
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}.
- on_post_step returns
  {{"kind": "inject_hint", "message": TEXT}} or
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}.

Field notes:
- code must define exactly one top-level def hook(ctx, nb), plus at most 5
  top-level helper functions with no imports or global state.
- ctx is read-only; nb is a per-episode scratch dict.
- You may read ctx either as dictionaries, e.g. ctx["state"]["page_type"], or
  as attributes, e.g. ctx.state.page_type.
- Do not use generic reflection helpers or forbidden builtins such as getattr,
  setattr, type, dir, vars, eval, exec, or open; access ctx fields directly.
- Helper functions cannot see local hook variables. If a helper needs ctx or
  nb, pass them explicitly, e.g. helper(ctx, value), not helper().
- Prefer narrow, evidence-backed hooks over broad always-on hints or actions.
- Use on_before_action for correcting or blocking the current action.
- Use on_post_step for deduplicated recovery hints or a carefully forced next
  click when the current page evidence makes it safe.
- For WebShop action strings, raw values like "buy now" are allowed for the
  current tool; complete "click[...]" and "search[...]" actions are also allowed.
- action.tool is exactly "search_action" or "click_action"; action.value is
  the raw search query or click value, and action.value_normalized is lowercase.
- Do not hard-code product IDs, exact product titles, ASINs, task indices, or
  test-set answers.

{variables}
{code_hook_notes}

Rules:
{bench_specific_rules}
- Use add_code_hook as the only action type for WebShop code-patch experiments.
- Include only hooks justified by recurring evidence; keep code short and general.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    if bench == "webshop" and schema_style == "webshop_life_multihook_v1":
        return f"""
{output_contract}

Do not copy grammar placeholder tokens.

Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "webshop"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Active WebShop action space for this experiment: use add_code_hook only.
- Legacy DSL actions are kept only for backward compatibility and are sleeping
  in this prompt. Do not use set_config, edit_tool_hint, add_or_edit_skill,
  add_guard_rule, or add_recovery_rule.

Multi-hook WebShop patch contract:
- actions may contain 1 to 4 add_code_hook actions.
- Use each hook at most once.
- Preferred order:
  1. "on_init" for a very short reusable shopping skill/tool_hint or nb flags;
  2. "on_post_step" to update nb["stage"] / nb flags from page evidence;
  3. "make_pre_hint" for deduplicated soft stage hints;
  4. "on_before_action" only for narrow block/rewrite/force interventions.
- A single on_before_action hook is acceptable for a narrow buy/options guard.
- Do not output hooks that are not needed by the recurring evidence.
- Keep each hook compact; total Python code across hooks should stay under
  9000 characters.
- Do not hard-code product IDs, exact product titles, ASINs, task indices, or
  test-set answers.

ACTION ::= ADD_CODE_HOOK
{add_code_hook_grammar}

Hook return contracts:
- on_init returns {{"skills": [{{"text": TEXT}}], "tool_hint": TEXT}} or None.
- on_post_step returns {{"kind": "inject_hint", "message": TEXT}},
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}, or None.
  Prefer updating nb over returning an effect.
- make_pre_hint returns {{"message": TEXT}} or None.
- on_before_action returns
  {{"kind": "block_and_prompt", "message": TEXT}},
  {{"kind": "rewrite_action", "action": TEXT, "message": TEXT}},
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}, or None.

Life-Harness-style design target for WebShop:
- Treat the patch as a small runtime program, not a one-step product chooser.
- Maintain reusable page/task state in nb, for example:
  search_query, inspect_results, inspect_product, select_options, budget_check,
  buy, recover_search_loop.
- Update nb["stage"] mainly in on_post_step using:
  ctx["task"], ctx["state"], ctx["webshop"], ctx["observation"], and
  ctx["action"].
- Use make_pre_hint for soft guidance when the next shopping subgoal is clear
  but forcing could disrupt already-successful trajectories.
- Use on_before_action hard interventions only when the condition is narrow:
  buy-now with required options unselected, buy-now over budget, repeated
  identical click/search loops, or a directly observable invalid action.
- Use on_post_step force_action only when current page evidence makes it safe,
  for example a deduplicated "buy now" after required options and budget checks
  are already satisfied.
- Avoid broad always-on hints. Deduplicate hints with nb["last_hint"] or a
  similar field.
- Do not force clicks to product IDs or exact product titles copied from traces.

<think> quality contract:
- Use 4 short bullets only:
  1. recurring failure pattern;
  2. state you will maintain in nb;
  3. which hook owns which part of the logic;
  4. regression risk avoided.
- The reasoning should explain why this patch is reusable across WebShop tasks.
- Do not include JSON inside <think>.

{variables}
{code_hook_notes}

Rules:
{bench_specific_rules}
- Include only interventions justified by recurring evidence.
- Prefer no-op or soft hint over broad hard force when the baseline already
  succeeds on many tasks.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    if bench == "dbbench" and schema_style == "dbbench_life_multihook_v1":
        return f"""
{output_contract}

Do not copy grammar placeholder tokens.

Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "dbbench"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Active DBBench action space for this experiment: use add_code_hook only.
- Do not use set_config, edit_tool_hint, add_or_edit_skill, add_guard_rule,
  or add_recovery_rule.

Multi-hook teacher patch contract:
- actions may contain 2 to 4 add_code_hook actions.
- Use each hook at most once.
- Preferred order:
  1. "on_init" for a very short SQL task-order skill/tool_hint or initial nb flags;
  2. "on_post_step" to update nb["stage"] / nb flags from SQL result evidence;
  3. "make_pre_hint" for deduplicated soft stage hints;
  4. "on_before_action" only for narrow block interventions.
- Do not output hooks that are not needed by the recurring evidence.
- DBBench v1 code hooks do not execute SQL rewrites or forced commits.
  Do not return rewrite_action or force_action; they are ignored for safety.
- Keep each hook compact; total Python code across hooks should stay under 9000 characters.
- Do not hard-code table cell values, exact final answers, task indices, or
  ground-truth SQL copied from traces.

ACTION ::= ADD_CODE_HOOK
{add_code_hook_grammar}

Hook return contracts:
- on_init returns {{"skills": [{{"text": TEXT}}], "tool_hint": TEXT}} or None.
- on_post_step returns {{"kind": "inject_hint", "message": TEXT}} or None.
  Prefer updating nb over returning an effect.
- make_pre_hint returns {{"message": TEXT}} or None.
- on_before_action returns
  {{"kind": "block_and_prompt", "message": TEXT}} or None.

Life-Harness-style design target:
- Treat the patch as a small runtime program, not a one-step SQL solver.
- Maintain a reusable stage/subgoal state in nb, for example:
  need_schema_probe, fixing_unknown_column, fixing_empty_result,
  mutation_needs_execution, ready_to_commit_candidate, repeated_query_loop.
- Update nb["stage"] mainly in on_post_step using:
  ctx["task"], ctx["state"], ctx["dbbench"], and ctx["predicates"].
- Use make_pre_hint for soft guidance when the next SQL debugging step is clear:
  DESCRIBE/SHOW TABLES after schema errors, inspect sample rows after empty
  SELECT, cast TEXT numerics for ranking/aggregation, verify mutations before
  commit.
- Use on_before_action only for narrow safety guards such as commit before any
  SQL, empty commit answers, or mutation commit before mutation execution.
- Since DBBench v1 cannot safely rewrite SQL strings, prefer a soft recovery
  hint or a narrow block over attempting to force a query.
- Avoid broad always-on hints. Deduplicate hints with nb["last_hint"] or a
  similar field.
- Do not emit exact SQL from the evidence unless it is a generic schema probe
  pattern; hints should describe reusable SQL debugging strategy.

<think> quality contract:
- Use 4 short bullets only:
  1. recurring failure pattern;
  2. subgoal state you will maintain;
  3. which hook owns which part of the logic;
  4. regression risk avoided.
- The reasoning should explain why this patch is reusable across tasks.
- Do not include JSON inside <think>.

{variables}
{code_hook_notes}

Rules:
{bench_specific_rules}
- Include only interventions justified by recurring evidence.
- Prefer no-op or soft hint over broad hard force when the baseline already
  succeeds on many tasks.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    if bench == "alfworld" and schema_style == "alfworld_life_multihook_v1":
        return f"""
{output_contract}

Do not copy grammar placeholder tokens.

Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "alfworld"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Active ALFWorld action space for this experiment: use add_code_hook only.
- Do not use edit_tool_hint, add_or_edit_skill, add_guard_rule,
  add_recovery_rule, or set_config.

Multi-hook teacher patch contract:
- actions may contain 2 to 4 add_code_hook actions.
- Use each hook at most once.
- Preferred order:
  1. "on_init" for a very short task-order skill/tool_hint or initial nb flags;
  2. "on_post_step" to update nb["stage"] / nb flags from action and observation;
  3. "make_pre_hint" for deduplicated soft stage hints;
  4. "on_before_action" only for narrow block/rewrite/force interventions.
- Do not output hooks that are not needed by the recurring evidence.
- Keep each hook compact; total Python code across hooks should stay under 9000 characters.
- Select exact actions only from ctx["admissible"].
- Do not hard-code numbered instance actions copied from traces.

ACTION ::= ADD_CODE_HOOK
{add_code_hook_grammar}

Hook return contracts:
- on_init returns {{"skills": [{{"text": TEXT}}], "tool_hint": TEXT}} or None.
- on_post_step returns {{"kind": "inject_hint", "message": TEXT}},
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}, or None.
  Prefer updating nb over returning an effect.
- make_pre_hint returns {{"message": TEXT}} or None.
- on_before_action returns
  {{"kind": "block_and_prompt", "message": TEXT}},
  {{"kind": "rewrite_action", "action": TEXT, "message": TEXT}},
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}, or None.

Life-Harness-style design target:
- Treat the patch as a small runtime program, not a one-step planner.
- Maintain a reusable stage/subgoal state in nb, for example:
  find_target, take_target, transform, go_destination, put_target, wrong_inventory.
- Update nb["stage"] mainly in on_post_step using:
  ctx["task"], ctx["world"], ctx["observation"], ctx["action"], and
  ctx["admissible"].
- Use make_pre_hint for soft guidance when the next subgoal is clear but forcing
  would risk disrupting already-successful trajectories.
- Use on_before_action hard interventions only when the condition is narrow:
  exact target take is admissible, exact required transform is admissible, or
  exact destination put is admissible under loop/low-budget evidence.
- If inventory is non-empty but does not match target_type, do not force a
  destination put. Prefer a soft recovery hint or no intervention.
- For pick_two_obj, use placed_items, placed_locations, and placed_count to
  avoid taking back already placed items and to prefer the same destination.
- Avoid broad always-on hints. Deduplicate hints with nb["last_hint"] or a
  similar field.

<think> quality contract:
- Use 4 short bullets only:
  1. recurring failure pattern;
  2. subgoal state you will maintain;
  3. which hook owns which part of the logic;
  4. regression risk avoided.
- The reasoning should explain why this patch is reusable across tasks.
- Do not include JSON inside <think>.

{variables}
{code_hook_notes}

Rules:
{bench_specific_rules}
- Include only interventions justified by recurring evidence.
- Prefer no-op or soft hint over broad hard force when the baseline already
  succeeds on many tasks.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    if bench == "alfworld" and schema_style in {"alfworld_life_onehook", "alfworld_life_onehook_v2", "alfworld_life_twohook"}:
        if schema_style in {"alfworld_life_onehook", "alfworld_life_onehook_v2"}:
            action_shape = """
- actions must contain exactly one add_code_hook action.
- The only allowed hook is "on_before_action".
- Do not output on_init, make_pre_hint, or on_post_step.
""".strip()
        else:
            action_shape = """
- actions must contain exactly two add_code_hook actions in this order:
  1. one "on_init" hook with a short reusable skill/tool_hint;
  2. one "on_before_action" hook with the actual stage correction.
- Do not output make_pre_hint or on_post_step.
""".strip()
        if schema_style == "alfworld_life_onehook_v2":
            target_stage_logic = """
- target_type and destination_type are semantic types, not exact instance ids.
- Build a small candidate target set from target_type. Generic aliases are
  allowed when semantically necessary: for cloth-like tasks, treat cloth,
  towel, and handtowel as target candidates. Do not hard-code numbered
  instances such as "towel 1".
- If no inventory and an admissible action starts with "take " and contains a
  target candidate, force that exact action.
- If holding an object that matches no target candidate, return None.
""".strip()
            conservation_note = """
- If several baseline failures share a pattern but the patch could affect many
  already-successful trajectories, prefer a narrow no-op or block only repeated
  invalid put actions. A positive patch should preserve successes.
""".strip()
        else:
            target_stage_logic = """
- target_type and destination_type are semantic types, not exact instance ids.
- If no inventory and an admissible action starts with "take " and contains
  target_type, force that exact action.
- If holding an object that is not target_type, return None.
""".strip()
            conservation_note = ""
        return f"""
{output_contract}

Do not copy grammar placeholder tokens.

Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "alfworld"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Active ALFWorld action space for this experiment: use add_code_hook only.
- Do not use edit_tool_hint, add_or_edit_skill, add_guard_rule,
  add_recovery_rule, set_config, make_pre_hint, or on_post_step.

Compact teacher patch contract:
{action_shape}
- Keep total Python code under 3500 characters.
- Use at most 1 helper function; zero helpers is preferred.
- The hook must be a small stage machine, not a general planner.
- Select exact actions only from ctx["admissible"].
- Do not hard-code numbered instance actions copied from traces.

Hook return contracts:
- on_init returns {{"skills": [{{"text": TEXT}}], "tool_hint": TEXT}} or None.
- on_before_action returns
  {{"kind": "block_and_prompt", "message": TEXT}},
  {{"kind": "rewrite_action", "action": TEXT, "message": TEXT}}, or
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}.

Use this compact stage logic:
- Read task = ctx.get("task") or {{}}, world = ctx.get("world") or {{}},
  admissible = ctx.get("admissible") or [].
{target_stage_logic}
- For clean/heat/cool tasks, maintain nb["transform_done"] from the latest
  observation and block target put until the required transform is done.
- If holding target and the exact required transform action is admissible,
  force that exact action.
- If holding target and transformed/no-transform, force a destination put only
  under loop or low-budget evidence; otherwise avoid over-control.
- For two-object tasks, do not take back placed_items; use placed_count and
  placed_locations only as generic state, not as copied task answers.
{conservation_note}

<think> quality contract:
- Use 3 short bullets only:
  1. recurring failure pattern;
  2. chosen stage-machine correction;
  3. regression risk avoided.
- The reasoning should explain why this patch is reusable across tasks.
- Do not include JSON inside <think>.

{variables}
{code_hook_notes}

Rules:
{bench_specific_rules}
- Include only interventions justified by recurring evidence.
- Prefer a narrow no-op over broad always-on hints.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    if bench == "alfworld" and schema_style in {
        "schema_only",
        "example",
        "alfworld_life_rubric",
        "alfworld_life_template",
    }:
        copy_instruction = "Do not copy grammar placeholder tokens."
        life_rubric = ""
        if schema_style in {"alfworld_life_rubric", "alfworld_life_template"}:
            life_rubric = """
Life-Harness-style design rubric for ALFWorld:
- Prefer hint-first patches: use make_pre_hint for subgoal guidance, and use
  on_before_action force_action only for narrow evidence-backed corrections.
- Model the task as a small stage machine in nb: find target, take target,
  required transform if any, go to destination, final put. Use ctx["task"] for
  task_type/target_type/destination_type and ctx["world"] for inventory,
  target_found, target_location, placed_items, and placed_locations.
- Safe force_action cases are limited to:
  1. a target take action is exactly present in ctx["admissible"];
  2. the held object is the task target and the required clean/heat/cool action
     is exactly present in ctx["admissible"];
  3. the held object is the task target and a destination put action is exactly
     present, preferably under low-budget or loop evidence.
- Never force placement of a non-target object. If inventory is non-empty but
  does not contain target_type, do not treat destination put as progress.
- Never break loops by choosing the first arbitrary different admissible action.
  Choose an action only if it matches the current task stage; otherwise emit a
  short hint.
- Avoid over-active hooks. A useful patch should preserve already-successful
  trajectories while fixing recurring failures.
- In <think>, explain the failure pattern, the stage logic, and the regression
  risks you are avoiding before writing the JSON patch.
""".strip()
        life_template = ""
        if schema_style == "alfworld_life_template":
            life_template = """
Constrained patch family for this teacher run:
- Use add_code_hook only. Prefer exactly these hooks:
  1. on_init for one short task-order skill/tool hint;
  2. on_post_step only to update nb flags such as transform_done from observations;
  3. make_pre_hint for deduplicated stage hints; store nb["last_hint"] and do
     not repeat the same hint on consecutive turns;
  4. on_before_action for narrow force/block only.
- In make_pre_hint, emit a hint only when it is stage-specific and changed:
  visible/known target to take, held target needs transform, held transformed
  target needs destination, or loop/low-budget task-order reminder.
- In on_before_action, use this safety order:
  a. If no inventory and an exact target take action appears, force that take.
  b. If inventory is non-empty but does not contain target_type, return None;
     do not force destination placement for a non-target object.
  c. If holding target and required transform is not done, force the exact
     transform action when present; otherwise block target put/wrong transform.
  d. If holding transformed/no-transform target and a destination put action is
     present, force it only under loop or low-budget evidence.
- For pick_two_obj, use placed_items/placed_locations/placed_count to avoid
  taking back already placed items and to prefer the same destination instance.
- Do not force a put action repeatedly when the agent just tried the same put.
  If same_action_repeated is true and the same final action is already the put,
  use a block_and_prompt or no intervention instead.
- Do not use broad always-on messages such as "target is held; go to destination"
  every step. Over-prompting successful tasks is a regression risk.
""".strip()
        format_contract = f"""
Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "alfworld"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Active ALFWorld action space for this experiment: use add_code_hook only.
- Legacy DSL actions are kept for backward compatibility but are sleeping in
  this prompt. Do not use edit_tool_hint, add_or_edit_skill, add_guard_rule,
  add_recovery_rule, or set_config.

ACTION ::= ADD_CODE_HOOK
{add_code_hook_grammar}

Hook return contracts:
- on_init returns {{"skills": [{{"text": TEXT}}, ...], "tool_hint": TEXT}} or None.
- make_pre_hint returns {{"message": TEXT}} or None.
- on_before_action returns
  {{"kind": "block_and_prompt", "message": TEXT}},
  {{"kind": "rewrite_action", "action": TEXT, "message": TEXT}}, or
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}.
- on_post_step returns
  {{"kind": "inject_hint", "message": TEXT}} or
  {{"kind": "force_action", "action": TEXT, "message": TEXT}}.

Field notes:
- code must define exactly one top-level def hook(ctx, nb), plus at most 5
  top-level helper functions with no imports or global state.
- ctx is read-only; nb is a per-episode scratch dict.
- ctx["world"] contains read-only facts from the Life-Harness world model:
  current_location, inventory, object_at, visited, unvisited, target_found,
  target_location, placed_count, placed_items, placed_locations, lamp_location.
- Build any task stage or subgoal logic yourself in nb using ctx["task"] and
  ctx["observation"]; built-in planner outputs/subgoals are not exposed.
- If world facts identify a safe next action, choose it from ctx["admissible"]
  with force_action/rewrite_action instead of emitting only a broad hint.
- Prefer general control logic over long natural-language hints.
- If choosing an exact action, choose from ctx["admissible"]; do not hard-code
  numbered instance actions copied from a trace.
- Keep each hook short and reusable.

{variables}
{code_hook_notes}
{life_rubric}
{life_template}
""".strip()
        return f"""
{output_contract}

{copy_instruction}

{format_contract}

Rules:
{bench_specific_rules}
- Use add_code_hook as the only action type for ALFWorld code-patch experiments.
- Include only hooks justified by recurring evidence; keep code short and general.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    if schema_style == "qwen25_sft":
        copy_instruction = "Do not copy placeholder text."
        format_contract = f"""
Each ACTION must be a flat JSON object with key "type". Do not use
"action_type" or nest an action body under the action name.
Required fields by type:
{set_config_required}
- edit_tool_hint: type, tool in {{{tools}}}, operation in {{append, prepend, replace}}, text
- add_or_edit_skill: type, skill_id, task_types, keywords, text
- add_guard_rule: type, rule_id, trigger=before_action, condition, effect
- add_recovery_rule: type, rule_id, trigger=post_step, condition, effect
{add_code_hook_required.rstrip()}
Guard (before_action) rule effects: block_and_prompt, force_action, or
rewrite_action. Recovery (post_step) rule effects: inject_hint or force_action.

Allowed config targets:
{fields_text}

Patches edit only the overlay substrate, not built-in H2/H3/H4/H5 modules.
Do not target enabled, h2, h3, h4, h5, or *_enabled aliases.

COND operators:
- {{"all": [COND, COND]}}, {{"any": [COND, COND]}}, {{"not": COND}}
- {{"eq": ["state.x", "literal"]}}, ne, gt, gte, lt, lte
{contains_example}
- {{"pred": "predicate_name"}}
{variables}
{code_hook_notes}
""".strip()
    elif schema_style == "example":
        copy_instruction = "Do not copy grammar placeholder tokens."
        format_contract = f"""
Patch JSON top-level contract:
- The <patch> block must contain one JSON object with top-level keys
  "benchmark", "description", and "actions".
- "benchmark" must be "{bench}".
- "actions" must be a non-empty array of ACTION objects.
- Never use an action type as a top-level key. This is invalid:
  {{"add_guard_rule": {{"type": "add_guard_rule"}}}}
- Never output a bare ACTION object. This is invalid:
  {{"type": "add_guard_rule", "rule_id": "x"}}
- Valid format example:
  {valid_example}

Each ACTION must be a flat JSON object with key "type". Do not use
"action_type" or nest an action body under the action name.
Required fields by type:
{set_config_required}
- edit_tool_hint: type, tool in {{{tools}}}, operation in {{append, prepend, replace}}, text
- add_or_edit_skill: type, skill_id, task_types, keywords, text
- add_guard_rule: type, rule_id, trigger=before_action, condition, effect
- add_recovery_rule: type, rule_id, trigger=post_step, condition, effect
{add_code_hook_required.rstrip()}
Guard (before_action) rule effects: block_and_prompt, force_action, or
rewrite_action. Recovery (post_step) rule effects: inject_hint or force_action.

Allowed config targets:
{fields_text}

Patches edit only the overlay substrate, not built-in H2/H3/H4/H5 modules.
Do not target enabled, h2, h3, h4, h5, or *_enabled aliases.

COND operators:
- {{"all": [COND, COND]}}, {{"any": [COND, COND]}}, {{"not": COND}}
- {{"eq": ["state.x", "literal"]}}, ne, gt, gte, lt, lte
{contains_example}
- {{"pred": "predicate_name"}}
{variables}
{code_hook_notes}
""".strip()
    else:
        copy_instruction = "Do not copy grammar placeholder tokens."
        format_contract = f"""
Patch JSON top-level contract:
- The <patch> block must contain exactly one JSON object.
- Top-level fields:
  - benchmark: string, exactly "{bench}"
  - description: short general string
  - actions: non-empty array of ACTION objects
- Never use an action type as a top-level key.
- Never output a bare ACTION object; actions must be inside the top-level
  "actions" array.

JSON grammar, with descriptive placeholders rather than values to copy:
PATCH ::= {{
  "benchmark": "{bench}",
  "description": TEXT,
  "actions": [ACTION, ...]
}}
ACTION ::= {action_grammar_union}
{set_config_grammar}
EDIT_TOOL_HINT ::= {{
  "type": "edit_tool_hint",
  "tool": TOOL_NAME,
  "operation": "append" | "prepend" | "replace",
  "text": TEXT
}}
ADD_OR_EDIT_SKILL ::= {{
  "type": "add_or_edit_skill",
  "skill_id": ID,
  "task_types": [TASK_TYPE_TOKEN, ...],
  "keywords": [TEXT, ...],
  "text": TEXT
}}
ADD_GUARD_RULE ::= {{
  "type": "add_guard_rule",
  "rule_id": ID,
  "trigger": "before_action",
  "condition": CONDITION,
  "effect": GUARD_EFFECT
}}
ADD_RECOVERY_RULE ::= {{
  "type": "add_recovery_rule",
  "rule_id": ID,
  "trigger": "post_step",
  "condition": CONDITION,
  "effect": RECOVERY_EFFECT
}}
{add_code_hook_grammar}
CONDITION ::= {{"all": [CONDITION, ...]}}
            | {{"any": [CONDITION, ...]}}
            | {{"not": CONDITION}}
            | {{"eq": [OPERAND, OPERAND]}}
            | {{"ne": [OPERAND, OPERAND]}}
            | {{"gt": [OPERAND, OPERAND]}}
            | {{"gte": [OPERAND, OPERAND]}}
            | {{"lt": [OPERAND, OPERAND]}}
            | {{"lte": [OPERAND, OPERAND]}}
            | {{"contains": [OPERAND, OPERAND]}}
            | {{"startswith": [OPERAND, OPERAND]}}
            | {{"endswith": [OPERAND, OPERAND]}}
            | {{"pred": PREDICATE_NAME}}
GUARD_EFFECT ::= {{"kind": "block_and_prompt", "message": TEXT}}
               | {{"kind": "force_action", "action": TEXT, "message"?: TEXT}}
               | {{"kind": "rewrite_action", "action": TEXT, "message"?: TEXT}}
RECOVERY_EFFECT ::= {{"kind": "inject_hint", "message": TEXT}}
                  | {{"kind": "force_action", "action": TEXT, "message"?: TEXT}}
Guard rules run before an action; add_guard_rule.effect must be a GUARD_EFFECT
(block_and_prompt, force_action, or rewrite_action).
Recovery rules run after a step; add_recovery_rule.effect must be a
RECOVERY_EFFECT (inject_hint or force_action).
To message the agent before it acts, use block_and_prompt; inject_hint runs
only after a step (recovery), never in a guard rule.

Field notes:
- TEXT values should be short, reusable, and independent of individual task
  indices, product IDs, exact product titles, or answers.
- ID values should be stable snake_case names for the reusable patch component.
- TASK_TYPE_TOKEN values must be "*" or short lowercase tokens using only
  letters, digits, underscore, dot, colon, or dash.
{json_scalar_note.rstrip()}
- TOOL_NAME must be one of: {tools}.
{config_target_note}
- OPERAND may be a listed path string, a predicate-compatible literal string,
  number, boolean, or null.
- PREDICATE_NAME must be one of the listed predicates below.
- force_action and rewrite_action "action" values are harness-executed tool
  action strings; use them only for narrow, evidence-backed corrections.
- The grammar tokens PATCH, ACTION, TEXT, ID, TASK_TYPE_TOKEN, TOOL_NAME,
  CONFIG_TARGET, JSON_SCALAR, CONDITION, GUARD_EFFECT, RECOVERY_EFFECT,
  OPERAND, and PREDICATE_NAME are not literal output values.

Allowed config targets:
{fields_text}
{config_value_note}

Patches edit only the overlay substrate, not built-in H2/H3/H4/H5 modules.
Do not target enabled, h2, h3, h4, h5, or *_enabled aliases.
{variables}
{code_hook_notes}
""".strip()

    if schema_style == "qwen25_sft":
        return f"""
{output_contract.strip()}

{copy_instruction}

{format_contract}


Rules:
{bench_specific_rules}
- Use only listed action types, names, paths, predicates, tools, and config targets.
- Do not put action.* conditions in post_step recovery rules.
- Include only actions justified by recurring evidence; keep text short and general.
- Put all JSON inside the <patch> block; no markdown.
""".strip()

    return f"""
{output_contract}

{copy_instruction}

{format_contract}

Rules:
{bench_specific_rules}
- Use only listed action types, names, paths, predicates, tools, and config targets.
- Do not put action.* conditions in post_step recovery rules.
- Include only actions justified by recurring evidence; keep text short and general.
- Put all JSON inside the <patch> block; no markdown.
""".strip()
