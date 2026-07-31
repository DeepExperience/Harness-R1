import copy
from typing import Any, Dict, List, Optional, Tuple

from .http_agent import HTTPAgent


DEFAULT_REFLECTION_PROMPT = """Reflect on the task trajectory before choosing the next action.

Use the task, all actions and observations so far, and the available tools to identify:
- what has been established,
- any mistake, loop, or unmet constraint,
- the best immediate strategy for the next action.

Return only a concise private reflection. Do not call a tool and do not give the final task answer.

Available tools:
{tools}
"""


DEFAULT_ACTION_PROMPT = """Use the private trajectory reflection below to choose the next action.

The reflection is guidance, not a new environment observation. Do not repeat it and do not assume that any unexecuted action has happened. Emit exactly one available tool call and no plain-text answer.

<private_reflection>
{reflection}
</private_reflection>
"""


def _add_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: sum(int(usage.get(key, 0) or 0) for usage in usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _compact_action(message: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        compact["content"] = content
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        calls.append(
            {
                "name": function.get("name"),
                "arguments": function.get("arguments", "{}"),
            }
        )
    if calls:
        compact["tool_calls"] = calls
    return compact


def _single_tool_call(message: Dict[str, Any]) -> Dict[str, Any]:
    selected = copy.deepcopy(message)
    calls = selected.get("tool_calls") or []
    if calls:
        selected["tool_calls"] = [calls[0]]
    return selected


def _tool_summary(tools: Optional[List[Dict[str, Any]]]) -> str:
    summaries = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = function.get("description")
        if isinstance(description, str) and description.strip():
            summaries.append(f"- {name}: {description.strip()}")
        else:
            summaries.append(f"- {name}")
    return "\n".join(summaries) if summaries else "- No tool metadata provided."


class ReflectionHTTPAgent(HTTPAgent):
    """Reflect on the current trajectory once, then generate one action.

    Reflection and action prompts are ephemeral. Only the selected action is
    returned to TaskClient and therefore only that action enters real history.
    """

    def __init__(
        self,
        *args,
        reflection_prompt: str = DEFAULT_REFLECTION_PROMPT,
        reflection_action_prompt: str = DEFAULT_ACTION_PROMPT,
        reflection_max_tokens: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reflection_prompt = reflection_prompt
        self.reflection_action_prompt = reflection_action_prompt

        reflection_body = copy.deepcopy(self.body)
        reflection_body.pop("tool_choice", None)
        reflection_body.pop("parallel_tool_calls", None)
        reflection_body["max_tokens"] = int(reflection_max_tokens)
        self._reflection_client = HTTPAgent(
            url=self.url,
            proxies=copy.deepcopy(self.proxies),
            body=reflection_body,
            headers=copy.deepcopy(self.headers),
            timeout=self.timeout,
        )

    def _reflection_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        prompt = self.reflection_prompt.replace("{tools}", _tool_summary(tools))
        return copy.deepcopy(messages) + [{"role": "user", "content": prompt}]

    def _action_messages(
        self,
        messages: List[Dict[str, Any]],
        reflection: str,
    ) -> List[Dict[str, Any]]:
        prompt = self.reflection_action_prompt.replace("{reflection}", reflection)
        return copy.deepcopy(messages) + [{"role": "user", "content": prompt}]

    def inference_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        reflection = ""
        reflection_usage: Dict[str, Any] = {}
        reflection_error = None
        try:
            reflection_message, reflection_usage = (
                self._reflection_client.inference_openai(
                    self._reflection_messages(messages, tools),
                    tools=None,
                )
            )
            content = reflection_message.get("content")
            if isinstance(content, str):
                reflection = content.strip()
        except Exception as exc:
            reflection_error = repr(exc)

        if reflection:
            action_messages = self._action_messages(messages, reflection)
            decision = "action_with_reflection"
        elif reflection_error is not None:
            action_messages = copy.deepcopy(messages)
            decision = "action_without_reflection_error"
        else:
            action_messages = copy.deepcopy(messages)
            decision = "action_without_reflection_empty"

        action, action_usage = super().inference_openai(action_messages, tools)
        selected = _single_tool_call(action)
        usage = _add_usage(reflection_usage, action_usage)
        usage["_agentbench_inference_metadata"] = {
            "method": "trajectory_reflection",
            "decision": decision,
            "reflection": reflection or None,
            "selected": _compact_action(selected),
            "reflection_usage": _add_usage(reflection_usage),
            "action_usage": _add_usage(action_usage),
            "reflection_error": reflection_error,
        }
        return selected, usage
