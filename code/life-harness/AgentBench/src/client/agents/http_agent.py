import contextlib
import copy
import json
import re
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional

import requests
from urllib3.exceptions import InsecureRequestWarning

from src.typings import *
from src.utils import *
from ..agent import AgentClient

old_merge_environment_settings = requests.Session.merge_environment_settings


_QWEN_CODER_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_]\w*)>\s*(.*?)</function>\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_QWEN_CODER_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z_]\w*)>\s*(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)


def _allowed_tool_names(tools: Optional[List[Dict[str, Any]]]) -> set[str]:
    allowed: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name:
            allowed.add(name)
    return allowed


def _tool_parameter_schemas(tools: Optional[List[Dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        name = function.get("name")
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        if isinstance(name, str) and isinstance(properties, dict):
            schemas[name] = properties
    return schemas


def _coerce_xml_parameter(raw_value: str, schema: Any) -> Any:
    value = raw_value.strip()
    expected = schema.get("type") if isinstance(schema, dict) else None
    if expected in {"array", "object", "integer", "number", "boolean"}:
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _make_openai_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {
        "id": f"compat_call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _json_payload_to_tool_calls(payload: Any, allowed: set[str]) -> list[dict[str, Any]]:
    items = payload if isinstance(payload, list) else [payload]
    calls: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        name = item.get("name") or item.get("tool_name")
        arguments = item.get("arguments", item.get("parameters", {}))
        if isinstance(function, dict):
            name = function.get("name", name)
            arguments = function.get("arguments", arguments)
        if not isinstance(name, str) or (allowed and name not in allowed):
            continue
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                parsed_arguments = {"value": arguments}
            arguments = parsed_arguments
        calls.append(_make_openai_tool_call(name, arguments))
    return calls


def _json_tool_text_to_tool_calls(content: str, allowed: set[str]) -> list[dict[str, Any]]:
    text = _strip_json_fence(content)
    try:
        payload = json.loads(text)
    except Exception:
        return []
    return _json_payload_to_tool_calls(payload, allowed)


def _tool_call_blocks_to_tool_calls(content: str, allowed: set[str]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block_match in _TOOL_CALL_BLOCK_RE.finditer(content):
        block = _strip_json_fence(block_match.group(1))
        try:
            payload = json.loads(block)
        except Exception:
            continue
        calls.extend(_json_payload_to_tool_calls(payload, allowed))
    return calls


def _qwen_coder_xml_to_tool_calls(
    msg: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Normalize Qwen3-Coder XML tool text into OpenAI tool_calls.

    Some local SGLang deployments return content like
    <tool_call><function=foo><parameter=x>...</parameter></function></tool_call>
    instead of filling message.tool_calls. This is a serving compatibility shim;
    it is intentionally independent from any task harness logic.
    """
    if msg.get("tool_calls"):
        return msg
    content = msg.get("content")
    if not isinstance(content, str) or "<tool_call>" not in content.lower():
        return msg
    allowed = _allowed_tool_names(tools)
    calls = _tool_call_blocks_to_tool_calls(content, allowed)
    if calls:
        normalized = dict(msg)
        normalized["tool_calls"] = calls
        return normalized
    parameter_schemas = _tool_parameter_schemas(tools)
    calls = []
    for match in _QWEN_CODER_TOOL_RE.finditer(content):
        name = match.group(1).strip()
        if allowed and name not in allowed:
            continue
        body = match.group(2)
        arguments = {
            param.group(1).strip(): _coerce_xml_parameter(
                param.group(2),
                (parameter_schemas.get(name) or {}).get(param.group(1).strip()),
            )
            for param in _QWEN_CODER_PARAM_RE.finditer(body)
        }
        calls.append(_make_openai_tool_call(name, arguments))
    if not calls:
        return msg
    normalized = dict(msg)
    normalized["tool_calls"] = calls
    return normalized


def _drop_null_openai_message_fields(msg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(msg)
    for key in ("tool_calls", "function_call", "audio", "refusal"):
        if normalized.get(key) is None:
            normalized.pop(key, None)
    return normalized


def _limit_to_single_tool_call(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool_calls for templates that require exactly one when present."""
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return msg
    normalized = dict(msg)
    if not tool_calls:
        normalized.pop("tool_calls", None)
    elif len(tool_calls) > 1:
        normalized["tool_calls"] = tool_calls[:1]
    else:
        return msg
    return normalized


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[Dict[str, Any], None]):
        # check if prompter_name is a method and its variable
        if not prompter:
            return Prompter.default()
        assert isinstance(prompter, dict)
        prompter_name = prompter.get("name", None)
        prompter_args = prompter.get("args", {})
        if hasattr(Prompter, prompter_name) and callable(
            getattr(Prompter, prompter_name)
        ):
            return getattr(Prompter, prompter_name)(**prompter_args)
        return Prompter.default()

    @staticmethod
    def default():
        return Prompter.role_content_dict()

    @staticmethod
    def batched_role_content_dict(*args, **kwargs):
        base = Prompter.role_content_dict(*args, **kwargs)

        def batched(messages):
            result = base(messages)
            return {key: [result[key]] for key in result}

        return batched

    @staticmethod
    def role_content_dict(
        message_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        user_role: str = "user",
        agent_role: str = "agent",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal message_key, role_key, content_key, user_role, agent_role
            role_dict = {
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return {message_key: prompt}

        return prompter

    @staticmethod
    def prompt_string(
        prefix: str = "",
        suffix: str = "AGENT:",
        user_format: str = "USER: {content}\n\n",
        agent_format: str = "AGENT: {content}\n\n",
        prompt_key: str = "prompt",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal prefix, suffix, user_format, agent_format, prompt_key
            prompt = prefix
            for item in messages:
                if item["role"] == "user":
                    prompt += user_format.format(content=item["content"])
                else:
                    prompt += agent_format.format(content=item["content"])
            prompt += suffix
            print(prompt)
            return {prompt_key: prompt}

        return prompter

    @staticmethod
    def claude():
        return Prompter.prompt_string(
            prefix="",
            suffix="Assistant:",
            user_format="Human: {content}\n\n",
            agent_format="Assistant: {content}\n\n",
        )

    @staticmethod
    def palm():
        def prompter(messages):
            return {"instances": [
                Prompter.role_content_dict("messages", "author", "content", "user", "bot")(messages)
            ]}
        return prompter


def check_context_limit(content: str):
    content = content.lower()
    and_words = [
        ["prompt", "context", "tokens"],
        [
            "limit",
            "exceed",
            "max",
            "long",
            "much",
            "many",
            "reach",
            "over",
            "up",
            "beyond",
        ],
    ]
    rule = AndRule(
        [
            OrRule([ContainRule(word) for word in and_words[i]])
            for i in range(len(and_words))
        ]
    )
    return rule.check(content)


class HTTPAgent(AgentClient):
    def __init__(
        self,
        url,
        proxies=None,
        body=None,
        headers=None,
        return_format="{response}",
        prompter=None,
        timeout=120,
        single_tool_call_only=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.proxies = proxies or {}
        self.headers = headers or {}
        self.body = body or {}
        self.return_format = return_format
        self.timeout = int(timeout)
        self.single_tool_call_only = bool(single_tool_call_only)
        self.prompter = Prompter.get_prompter(prompter)
        if not self.url:
            raise Exception("Please set 'url' parameter")

    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)

    def inference(self, history: List[dict]) -> str:
        for _ in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))
                with no_ssl_verification():
                    resp = requests.post(
                        self.url, json=body, headers=self.headers, proxies=self.proxies, timeout=self.timeout
                    )
                # print(resp.status_code, resp.text)
                if resp.status_code != 200:
                    # print(resp.text)
                    if check_context_limit(resp.text):
                        raise AgentContextLimitException(resp.text)
                    else:
                        raise Exception(
                            f"Invalid status code {resp.status_code}:\n\n{resp.text}"
                        )
            except AgentClientException as e:
                raise e
            except Exception as e:
                print("Warning: ", e)
                pass
            else:
                resp = resp.json()
                return self.return_format.format(response=resp)
            time.sleep(_ + 2)
        raise Exception("Failed.")

    def inference_openai(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        For agentrl controller protocol: call chat/completions with full OpenAI-style
        messages and optional tools, return the assistant message dict (may include tool_calls).
        """
        for _ in range(3):
            try:
                body = copy.deepcopy(self.body)
                body["messages"] = messages
                if tools:
                    body["tools"] = tools
                with no_ssl_verification():
                    resp = requests.post(
                        self.url, json=body, headers=self.headers, proxies=self.proxies, timeout=self.timeout
                    )
                if resp.status_code != 200:
                    if check_context_limit(resp.text):
                        raise AgentContextLimitException(resp.text)
                    raise Exception(
                        f"Invalid status code {resp.status_code}:\n\n{resp.text}"
                    )
            except AgentClientException as e:
                raise e
            except Exception as e:
                print("Warning: ", e)
                pass
            else:
                data = resp.json()
                msg = data["choices"][0]["message"]
                if not isinstance(msg, dict):
                    raise Exception(f"Unexpected message shape: {msg!r}")
                msg = _qwen_coder_xml_to_tool_calls(msg, tools)
                if not msg.get("tool_calls") and isinstance(msg.get("content"), str):
                    rescued_tool_calls = _json_tool_text_to_tool_calls(
                        msg["content"],
                        _allowed_tool_names(tools),
                    )
                    if rescued_tool_calls:
                        msg = dict(msg)
                        msg["tool_calls"] = rescued_tool_calls
                msg = _drop_null_openai_message_fields(msg)
                if self.single_tool_call_only:
                    msg = _limit_to_single_tool_call(msg)
                usage = data.get("usage") or {}
                return msg, usage
            time.sleep(_ + 2)
        raise Exception("Failed.")
