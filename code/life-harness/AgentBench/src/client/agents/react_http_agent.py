import copy
from typing import Any, Dict, List, Optional, Tuple

from .http_agent import HTTPAgent


DEFAULT_REACT_PROMPT = """Use the ReAct procedure for every environment step.

Thought: reason step by step from the task, interaction history, latest observation, available actions, and tool definitions. Keep this reasoning concise and specific to the immediate decision.
Action: after the thought, issue exactly one valid structured tool call from the provided tools. Put the action in message.tool_calls rather than writing an action as plain text. Never invent an unavailable action or call multiple tools in one step.
Observation: treat the returned tool result as the next observation, then repeat Thought followed by one Action until the task is complete. When a final-answer or finish tool is available, use that tool rather than returning the answer only as plain text.
A direct tool call without preceding reasoning is not ReAct; always produce the Thought before the structured Action.
"""


def _inject_react_prompt(
    messages: List[Dict[str, Any]], prompt: str
) -> List[Dict[str, Any]]:
    prepared = copy.deepcopy(messages)
    marker = "[REACT_BASELINE_V1]"
    block = f"{marker}\n{prompt.strip()}"
    for message in prepared:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and marker not in content:
            message["content"] = f"{content.rstrip()}\n\n{block}"
        elif not isinstance(content, str):
            message["content"] = block
        return prepared
    prepared.insert(0, {"role": "system", "content": block})
    return prepared


class ReActHTTPAgent(HTTPAgent):
    """Single-call direct agent with an explicit visible Thought/Action contract."""

    def __init__(
        self,
        *args,
        react_prompt: str = DEFAULT_REACT_PROMPT,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.react_prompt = str(react_prompt).strip()

    def inference_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prepared = _inject_react_prompt(messages, self.react_prompt)
        message, usage = super().inference_openai(prepared, tools)
        content = message.get("content")
        visible_thought = content.strip() if isinstance(content, str) else ""
        normalized_usage = copy.deepcopy(usage)
        normalized_usage["_agentbench_inference_metadata"] = {
            "method": "react",
            "protocol_version": "visible_thought_structured_tool_v1",
            "visible_thought": visible_thought or None,
            "visible_thought_chars": len(visible_thought),
            "tool_call_count": len(message.get("tool_calls") or []),
        }
        return message, normalized_usage
