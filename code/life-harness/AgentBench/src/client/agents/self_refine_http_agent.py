import ast
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .http_agent import HTTPAgent


DEFAULT_SELF_CRITIQUE_PROMPT = """Critique the candidate action before it is executed.

Use the task, conversation history, latest observation, currently available actions, and tool definitions. Identify a concrete error only when one is present. The candidate has not been executed; do not assume its effects occurred.

Return only one JSON object with this shape:
{{"verdict":"keep|revise","issue":"specific error or empty string","suggested_action":"concise correction or empty string"}}

The feedback must be specific and actionable. Use verdict "keep" only when the candidate is a valid and appropriate immediate next action. Do not call a tool and do not give the final task answer.

Available tools:
{tools}

Candidate action:
{candidate}
"""


DEFAULT_SELF_REFINE_PROMPT = """Refine the candidate action using the critic feedback.

Use the task, conversation history, latest observation, currently available actions, tool definitions, candidate, and feedback. The candidate has not been executed. Emit exactly one currently valid tool call and no plain-text answer.

Candidate action:
{candidate}

Critic feedback:
{feedback}
"""


def _compact_candidate(message: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep reviewer-relevant fields and strip generated tool-call ids."""
    if not isinstance(message, dict):
        return {}
    candidate: Dict[str, Any] = {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        candidate["content"] = content

    tool_calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        tool_calls.append(
            {
                "name": function.get("name"),
                "arguments": function.get("arguments", "{}"),
            }
        )
    if tool_calls:
        candidate["tool_calls"] = tool_calls
    return candidate


def _single_tool_call(message: Dict[str, Any]) -> Dict[str, Any]:
    """The supported interactive benchmarks execute one action per step."""
    normalized = copy.deepcopy(message)
    calls = normalized.get("tool_calls") or []
    if calls:
        normalized["tool_calls"] = [calls[0]]
    return normalized


def _add_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: sum(int(usage.get(key, 0) or 0) for usage in usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _action_signature(message: Optional[Dict[str, Any]]) -> tuple[Any, Any]:
    candidate = _compact_candidate(message)
    calls = candidate.get("tool_calls") or []
    if not calls:
        return None, None
    call = calls[0]
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            pass
    return call.get("name"), arguments


def _tool_summary(tools: Optional[List[Dict[str, Any]]]) -> str:
    summaries = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        description = function.get("description")
        line = f"- {function['name']}"
        if isinstance(description, str) and description.strip():
            line += f": {description.strip()}"
        summaries.append(line)
    return "\n".join(summaries) if summaries else "- No tool metadata provided."


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = stripped.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _critic_verdict(
    feedback: Optional[Dict[str, Any]], raw_text: str = ""
) -> Optional[str]:
    if not isinstance(feedback, dict):
        match = re.search(
            r'["\']verdict["\']\s*:\s*["\'](keep|revise)["\']',
            raw_text,
            flags=re.IGNORECASE,
        )
        return match.group(1).lower() if match else None
    verdict = feedback.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower()
        if verdict in {"keep", "revise"}:
            return verdict
    return None


def _parse_arguments(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    return {
        "string": lambda: isinstance(value, str),
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }.get(expected, lambda: True)()


def _validate_tool_schema(
    name: Any,
    arguments: Optional[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, str]:
    if not isinstance(name, str) or not name:
        return False, "missing_tool_name"
    if arguments is None:
        return False, "invalid_tool_arguments_json"

    spec = None
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            spec = function
            break
    if spec is None:
        return False, "unknown_tool"

    schema = spec.get("parameters") or {}
    required = schema.get("required") or []
    for key in required:
        if key not in arguments:
            return False, f"missing_required_argument:{key}"
    if schema.get("additionalProperties") is False:
        allowed = set((schema.get("properties") or {}).keys())
        extras = set(arguments) - allowed
        if extras:
            return False, f"unexpected_argument:{sorted(extras)[0]}"
    for key, value in arguments.items():
        expected = ((schema.get("properties") or {}).get(key) or {}).get("type")
        if expected is not None and not _matches_json_type(value, expected):
            return False, f"argument_type:{key}"
    return True, "tool_schema_valid"


def _latest_message_with(messages: List[Dict[str, Any]], marker: str) -> Optional[str]:
    marker_lower = marker.lower()
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and marker_lower in content.lower():
            return content
    return None


def _alfworld_action_valid(action: Any, messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not isinstance(action, str) or not action.strip():
        return False, "empty_alfworld_action"
    content = _latest_message_with(messages, "AVAILABLE ACTIONS:")
    if content is None:
        return True, "alfworld_actions_unavailable"
    tail = content.lower().rsplit("available actions:", 1)[1]
    available = {line.strip() for line in tail.splitlines() if line.strip()}
    normalized = action.strip().lower().splitlines()[0]
    if normalized not in available:
        return False, "alfworld_action_not_available"
    return True, "alfworld_action_available"


def _webshop_state(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    content = _latest_message_with(messages, "Available Actions:")
    if content is None:
        return None
    marker = "available actions:"
    marker_at = content.lower().rfind(marker)
    if marker_at < 0:
        return None
    # Preserve Python literal casing (notably True/False) for literal_eval.
    raw = content[marker_at + len(marker) :].strip()
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _webshop_action_valid(
    name: str,
    arguments: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    state = _webshop_state(messages)
    if state is None:
        return True, "webshop_actions_unavailable"
    if name == "search_action":
        keywords = arguments.get("keywords")
        if not state.get("has_search_bar"):
            return False, "webshop_search_not_available"
        if not isinstance(keywords, str) or not keywords.strip():
            return False, "webshop_empty_search"
        return True, "webshop_search_available"
    if name == "click_action":
        value = arguments.get("value")
        if not isinstance(value, str) or not value.strip():
            return False, "webshop_empty_click"
        clickables = {
            str(item).strip().casefold() for item in state.get("clickables", [])
        }
        if value.strip().casefold() not in clickables:
            return False, "webshop_click_not_available"
        return True, "webshop_click_available"
    return True, "webshop_nonstandard_tool"


def _action_valid(
    message: Optional[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, str]:
    if not isinstance(message, dict):
        return False, "missing_message"
    calls = message.get("tool_calls") or []
    if not calls:
        return False, "missing_tool_call"
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    arguments = _parse_arguments(call)
    schema_valid, schema_reason = _validate_tool_schema(name, arguments, tools)
    if not schema_valid:
        return False, schema_reason
    assert isinstance(name, str) and isinstance(arguments, dict)
    if name == "take_action":
        return _alfworld_action_valid(arguments.get("action"), messages)
    if name in {"search_action", "click_action"}:
        return _webshop_action_valid(name, arguments, messages)
    return True, "tool_schema_valid"


class SelfRefineHTTPAgent(HTTPAgent):
    """Generate, critique, refine, and gate one action before execution."""

    def __init__(
        self,
        *args,
        self_critique_prompt: str = DEFAULT_SELF_CRITIQUE_PROMPT,
        self_refine_prompt: str = DEFAULT_SELF_REFINE_PROMPT,
        self_refine_fallback_to_draft: bool = True,
        self_critique_max_tokens: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.self_critique_prompt = self_critique_prompt
        self.self_refine_prompt = self_refine_prompt
        self.self_refine_fallback_to_draft = bool(self_refine_fallback_to_draft)

        critique_body = copy.deepcopy(self.body)
        critique_body.pop("tool_choice", None)
        critique_body.pop("parallel_tool_calls", None)
        critique_body["max_tokens"] = int(self_critique_max_tokens)
        self._critique_client = HTTPAgent(
            url=self.url,
            proxies=copy.deepcopy(self.proxies),
            body=critique_body,
            headers=copy.deepcopy(self.headers),
            timeout=self.timeout,
        )

    def _critique_messages(
        self,
        messages: List[Dict[str, Any]],
        draft: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        prompt = self.self_critique_prompt.format(
            candidate=json.dumps(_compact_candidate(draft), ensure_ascii=False, sort_keys=True),
            tools=_tool_summary(tools),
        )
        return copy.deepcopy(messages) + [{"role": "user", "content": prompt}]

    def _refine_messages(
        self,
        messages: List[Dict[str, Any]],
        draft: Dict[str, Any],
        feedback_text: str,
    ) -> List[Dict[str, Any]]:
        prompt = self.self_refine_prompt.format(
            candidate=json.dumps(_compact_candidate(draft), ensure_ascii=False, sort_keys=True),
            feedback=feedback_text,
        )
        return copy.deepcopy(messages) + [{"role": "user", "content": prompt}]

    def inference_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        draft, draft_usage = super().inference_openai(messages, tools)
        draft_valid, draft_validity_reason = _action_valid(draft, messages, tools)

        feedback_text = ""
        feedback_obj = None
        feedback_usage: Dict[str, Any] = {}
        feedback_error = None
        try:
            feedback_message, feedback_usage = self._critique_client.inference_openai(
                self._critique_messages(messages, draft, tools),
                tools=None,
            )
            content = feedback_message.get("content")
            if isinstance(content, str):
                feedback_text = content.strip()
                feedback_obj = _extract_json_object(feedback_text)
        except Exception as exc:
            feedback_error = repr(exc)

        verdict = _critic_verdict(feedback_obj, feedback_text)
        refined = None
        refine_usage: Dict[str, Any] = {}
        refined_valid = False
        refined_validity_reason = "not_generated"
        refine_error = None

        if feedback_error is not None or not feedback_text:
            if not self.self_refine_fallback_to_draft:
                if feedback_error is not None:
                    raise RuntimeError(feedback_error)
                raise RuntimeError("Self-Refine critic returned empty feedback")
            selected = _single_tool_call(draft)
            decision = (
                "draft_fallback_critic_error"
                if feedback_error is not None
                else "draft_fallback_critic_empty"
            )
        elif verdict == "keep" and draft_valid:
            selected = _single_tool_call(draft)
            decision = "critic_keep_draft"
        else:
            try:
                refined, refine_usage = super().inference_openai(
                    self._refine_messages(messages, draft, feedback_text),
                    tools,
                )
                refined_valid, refined_validity_reason = _action_valid(
                    refined, messages, tools
                )
                if refined_valid:
                    selected = _single_tool_call(refined)
                    changed = _action_signature(refined) != _action_signature(draft)
                    decision = "refined_changed" if changed else "refined_unchanged"
                elif draft_valid and self.self_refine_fallback_to_draft:
                    selected = _single_tool_call(draft)
                    decision = "draft_gate_refined_invalid"
                elif self.self_refine_fallback_to_draft:
                    selected = _single_tool_call(draft)
                    decision = "draft_fallback_both_invalid"
                else:
                    selected = _single_tool_call(refined)
                    decision = "refined_invalid_fail_open"
            except Exception as exc:
                if not self.self_refine_fallback_to_draft:
                    raise
                refine_error = repr(exc)
                selected = _single_tool_call(draft)
                decision = "draft_fallback_refiner_error"

        changed = _action_signature(selected) != _action_signature(draft)
        usage = _add_usage(draft_usage, feedback_usage, refine_usage)
        usage["_agentbench_inference_metadata"] = {
            "method": "action_self_refine",
            "protocol_version": "feedback_refine_v1",
            "decision": decision,
            "changed": changed,
            "draft": _compact_candidate(draft),
            "draft_valid": draft_valid,
            "draft_validity_reason": draft_validity_reason,
            "feedback": feedback_text or None,
            "feedback_parsed": feedback_obj,
            "feedback_verdict": verdict,
            "refined": _compact_candidate(refined),
            "refined_valid": refined_valid,
            "refined_validity_reason": refined_validity_reason,
            "selected": _compact_candidate(selected),
            "draft_usage": _add_usage(draft_usage),
            "feedback_usage": _add_usage(feedback_usage),
            "refine_usage": _add_usage(refine_usage),
            "critic_error": feedback_error,
            "refiner_error": refine_error,
        }
        return selected, usage
