import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from .http_agent import HTTPAgent


DEFAULT_PLAN_PROMPT = """[REACT_PLAN_STAGE_V1]
Plan the single best next environment action from the task, interaction history, latest observation, available actions, and tool definitions.

Return only a concise, concrete plan in one to three sentences. Do not call a tool, emit tool-call syntax, give a final answer, or claim that the planned action has already happened.

Available tool catalog:
{tools}
"""


DEFAULT_ACT_PROMPT = """[REACT_ACTION_STAGE_V1]
Execute the next step using the plan below. Issue exactly one currently valid structured tool call from the provided tools. Do not return a plain-text action, repeat the plan, or call multiple tools.

Plan:
{plan}
"""


DEFAULT_OBSERVATION_LAST_ACT_PROMPT = """[REACT_ACTION_STAGE_V2]
Use the current-step plan only as guidance. Issue exactly one structured tool call that is valid for the current observation. If the plan conflicts with the current observation or its available actions, follow the current observation. Do not repeat the plan or return a plain-text action.

Current-step plan:
{plan}

Current observation:
{observation}"""


def _message_text(message: Dict[str, Any]) -> str:
    for key in ("content", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _usage_sum(*usages: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: sum(int(usage.get(key, 0) or 0) for usage in usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _tool_catalog(tools: Optional[List[Dict[str, Any]]]) -> str:
    functions = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict):
            functions.append(function)
    return json.dumps(functions, ensure_ascii=False, sort_keys=True)


def _one_tool_call(message: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(message)
    calls = normalized.get("tool_calls") or []
    if calls:
        normalized["tool_calls"] = calls[:1]
    return normalized


class PlanActHTTPAgent(HTTPAgent):
    """Two-call ReAct: generate a visible plan, then condition one tool action on it."""

    def __init__(
        self,
        *args,
        plan_prompt: str = DEFAULT_PLAN_PROMPT,
        act_prompt: str = DEFAULT_ACT_PROMPT,
        observation_last_act_prompt: str = DEFAULT_OBSERVATION_LAST_ACT_PROMPT,
        action_context_mode: str = "append_plan_v1",
        plan_max_tokens: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.plan_prompt = str(plan_prompt).strip()
        self.act_prompt = str(act_prompt).strip()
        self.observation_last_act_prompt = str(observation_last_act_prompt).strip()
        self.action_context_mode = str(action_context_mode).strip()
        if self.action_context_mode not in {
            "append_plan_v1",
            "observation_last_v2",
        }:
            raise ValueError(
                "action_context_mode must be append_plan_v1 or observation_last_v2"
            )

        plan_body = copy.deepcopy(self.body)
        plan_body.pop("tool_choice", None)
        plan_body.pop("parallel_tool_calls", None)
        plan_body["max_tokens"] = int(plan_max_tokens)
        plan_body["continue_final_message"] = True
        plan_body["stop"] = ["<tool_call>"]
        chat_template_kwargs = copy.deepcopy(
            plan_body.get("chat_template_kwargs") or {}
        )
        chat_template_kwargs["enable_thinking"] = False
        plan_body["chat_template_kwargs"] = chat_template_kwargs
        self._plan_client = HTTPAgent(
            url=self.url,
            proxies=copy.deepcopy(self.proxies),
            body=plan_body,
            headers=copy.deepcopy(self.headers),
            timeout=self.timeout,
        )

    def inference_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        plan_messages = copy.deepcopy(messages) + [
            {
                "role": "user",
                "content": self.plan_prompt.format(tools=_tool_catalog(tools)),
            },
            {"role": "assistant", "content": "Plan:"},
        ]
        plan_message, plan_usage = self._plan_client.inference_openai(
            plan_messages, tools=None
        )
        plan = _message_text(plan_message)

        fallback_plan = (
            plan
            or "Select the safest valid next action from the latest observation."
        )
        if self.action_context_mode == "observation_last_v2":
            action_messages = copy.deepcopy(messages)
            if action_messages and isinstance(
                action_messages[-1].get("content"), str
            ):
                observation = action_messages[-1]["content"]
                action_messages[-1]["content"] = (
                    self.observation_last_act_prompt.format(
                        plan=fallback_plan,
                        observation=observation,
                    )
                )
            else:
                action_messages.append(
                    {
                        "role": "user",
                        "content": self.observation_last_act_prompt.format(
                            plan=fallback_plan,
                            observation="No textual observation was provided.",
                        ),
                    }
                )
            protocol_version = "visible_plan_then_observation_action_v2"
        else:
            action_messages = copy.deepcopy(messages) + [
                {
                    "role": "assistant",
                    "content": f"Plan: {plan}" if plan else "Plan: unavailable",
                },
                {
                    "role": "user",
                    "content": self.act_prompt.format(plan=fallback_plan),
                },
            ]
            protocol_version = "visible_plan_then_structured_action_v1"
        action_message, action_usage = super().inference_openai(
            action_messages, tools
        )
        selected = _one_tool_call(action_message)

        usage = _usage_sum(plan_usage, action_usage)
        usage["_agentbench_inference_metadata"] = {
            "method": "plan_act",
            "protocol_version": protocol_version,
            "action_context_mode": self.action_context_mode,
            "plan": plan or None,
            "plan_nonempty": bool(plan),
            "plan_chars": len(plan),
            "action_tool_call_count": len(selected.get("tool_calls") or []),
            "plan_usage": _usage_sum(plan_usage),
            "action_usage": _usage_sum(action_usage),
        }
        return selected, usage
