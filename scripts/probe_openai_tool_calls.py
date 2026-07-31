#!/usr/bin/env python3
"""Verify that an OpenAI-compatible endpoint returns structured tool calls."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_response(response: dict[str, Any]) -> tuple[bool, str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "missing choices"
    choice = choices[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = str(message.get("content") or "")
        return False, f"missing message.tool_calls; content_prefix={content[:160]!r}"
    function = tool_calls[0].get("function") or {}
    if function.get("name") != "lookup_weather":
        return False, f"unexpected tool name: {function.get('name')!r}"
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return False, f"tool arguments are not JSON: {exc}"
    if not isinstance(arguments, dict) or arguments.get("city") != "Paris":
        return False, f"unexpected tool arguments: {arguments!r}"
    if choice.get("finish_reason") not in {"tool_calls", "stop"}:
        return False, f"unexpected finish_reason: {choice.get('finish_reason')!r}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tool-choice", choices=("auto", "required"), required=True)
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Call lookup_weather exactly once for Paris. Do not answer in plain text.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": args.tool_choice,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": chat_template_kwargs,
    }

    records: list[dict[str, Any]] = []
    passed = 0
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    for attempt in range(1, args.attempts + 1):
        started = time.monotonic()
        try:
            response = post_json(endpoint, payload, args.timeout)
            ok, detail = validate_response(response)
            records.append(
                {
                    "attempt": attempt,
                    "ok": ok,
                    "detail": detail,
                    "elapsed_seconds": time.monotonic() - started,
                    "response": response,
                }
            )
            passed += int(ok)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            records.append(
                {
                    "attempt": attempt,
                    "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.monotonic() - started,
                }
            )

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "tool_choice": args.tool_choice,
        "chat_template_kwargs": chat_template_kwargs,
        "attempts": args.attempts,
        "passed": passed,
        "all_passed": passed == args.attempts,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
