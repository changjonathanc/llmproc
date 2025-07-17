"""Unit tests for trigger_event callback dispatch."""

from typing import Any

import pytest

from llmproc.plugin.events import CallbackEvent
from tests.conftest import create_test_llmprocess_directly

EVENT_PARAMS = [
    (CallbackEvent.TOOL_START, {"tool_name": "calc", "tool_args": {"x": 1}}),
    (CallbackEvent.TOOL_END, {"tool_name": "calc", "result": "ok"}),
    (CallbackEvent.API_STREAM_BLOCK, {"block": object()}),
    (CallbackEvent.API_REQUEST, {"api_request": {"url": "https://api"}}),
    (CallbackEvent.API_RESPONSE, {"response": {"status": 200}}),
    (CallbackEvent.TURN_START, {"run_result": {"iteration": 1}}),
    (CallbackEvent.TURN_END, {"response": {"text": "done"}, "tool_results": []}),
    (CallbackEvent.RUN_END, {"run_result": {"ok": True}}),
]


class CallbackWithProcess:
    """Callback object with process parameter in methods."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def tool_start(self, tool_name, tool_args, *, process):
        self.calls.append(("tool_start", process))

    def api_stream_block(self, block, *, process):
        self.calls.append(("api_stream_block", process))

    def tool_end(self, tool_name, result, *, process):
        self.calls.append(("tool_end", process))

    def response(self, content, *, process):
        self.calls.append(("response", process))

    def api_request(self, api_request, *, process):
        self.calls.append(("api_request", process))

    def api_response(self, response, *, process):
        self.calls.append(("api_response", process))

    def turn_start(self, *, process, run_result):
        self.calls.append(("turn_start", process))

    def turn_end(self, response, tool_results, *, process):
        self.calls.append(("turn_end", process))

    def run_end(self, run_result, *, process):
        self.calls.append(("run_end", process))



class CallbackWithoutProcess:
    """Callback object without process parameter."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def tool_start(self, tool_name, tool_args):
        self.calls.append("tool_start")

    def api_stream_block(self, block):
        self.calls.append("api_stream_block")

    def tool_end(self, tool_name, result):
        self.calls.append("tool_end")

    def response(self, content):
        self.calls.append("response")

    def api_request(self, api_request):
        self.calls.append("api_request")

    def api_response(self, response):
        self.calls.append("api_response")

    def turn_start(self, run_result):
        self.calls.append("turn_start")

    def turn_end(self, response, tool_results):
        self.calls.append("turn_end")

    def run_end(self, run_result):
        self.calls.append("run_end")



@pytest.mark.asyncio
@pytest.mark.parametrize("event,kwargs", EVENT_PARAMS)
async def test_trigger_event_process_injection(event, kwargs):
    """All callbacks should run without missing-argument errors."""
    process = create_test_llmprocess_directly()

    cb_with = CallbackWithProcess()
    cb_without = CallbackWithoutProcess()

    process.add_plugins(cb_with)
    process.add_plugins(cb_without)

    await process.trigger_event(event, **kwargs)

    assert len(cb_with.calls) == 1
    assert cb_with.calls[0][0] == event.value
    assert cb_with.calls[0][1] is process

    assert len(cb_without.calls) == 1
    assert cb_without.calls[0] == event.value
