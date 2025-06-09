"""Tests for correct callback parameter mapping in trigger_event."""

from typing import Any

from llmproc.callbacks import CallbackEvent
from tests.conftest import create_test_llmprocess_directly


def test_tool_start_and_response_parameters():
    """Callback methods should receive correctly mapped parameters."""
    process = create_test_llmprocess_directly()

    calls: dict[str, Any] = {}

    class MyCallback:
        def tool_start(self, tool_name: str, args: Any) -> None:
            calls["tool_name"] = tool_name
            calls["args"] = args

        def response(self, content: str) -> None:
            calls["content"] = content

    process.add_callback(MyCallback())
    process.trigger_event(CallbackEvent.TOOL_START, "calc", {"x": 1})
    process.trigger_event(CallbackEvent.RESPONSE, "done")

    assert calls["tool_name"] == "calc"
    assert calls["args"] == {"x": 1}
    assert calls["content"] == "done"
