from src.client.agents.http_agent import _limit_to_single_tool_call


def test_limit_to_single_tool_call_preserves_first_without_mutating_input():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-1", "function": {"name": "first"}},
            {"id": "call-2", "function": {"name": "second"}},
        ],
    }

    normalized = _limit_to_single_tool_call(message)

    assert [call["id"] for call in normalized["tool_calls"]] == ["call-1"]
    assert [call["id"] for call in message["tool_calls"]] == ["call-1", "call-2"]


def test_limit_to_single_tool_call_keeps_single_call_message_identity():
    message = {
        "role": "assistant",
        "tool_calls": [{"id": "call-1", "function": {"name": "only"}}],
    }

    assert _limit_to_single_tool_call(message) is message


def test_limit_to_single_tool_call_removes_empty_tool_calls():
    message = {"role": "assistant", "content": "I need to search.", "tool_calls": []}

    normalized = _limit_to_single_tool_call(message)

    assert "tool_calls" not in normalized
    assert message["tool_calls"] == []
