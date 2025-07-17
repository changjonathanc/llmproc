"""Tests for the TURN_START and TURN_END callback events."""

from llmproc.plugin.events import CallbackEvent


class DummyProcess:
    """Simplified process that forwards events to callbacks."""

    def __init__(self) -> None:
        self.callbacks = []

    def trigger_event(self, event: CallbackEvent, *args, **kwargs) -> None:
        event_name = event.value
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(*args, **kwargs)
            elif callable(cb):
                cb(event, *args, **kwargs)


def test_turn_callbacks():
    """Verify TURN_START and TURN_END events reach registered callbacks."""
    recorded = []

    class CallbackRecorder:
        def turn_start(self, *, process, run_result=None):
            recorded.append((CallbackEvent.TURN_START, (process,)))

        def turn_end(self, response, tool_results, *, process):
            recorded.append((CallbackEvent.TURN_END, (process, response, tool_results)))

    proc = DummyProcess()
    proc.callbacks.append(CallbackRecorder())

    response = {"text": "hi"}
    tool_results = ["tool_result"]
    proc.trigger_event(CallbackEvent.TURN_START, process=proc)
    proc.trigger_event(CallbackEvent.TURN_END, response, tool_results, process=proc)

    assert recorded[0][0] == CallbackEvent.TURN_START
    assert recorded[0][1][0] is proc
    assert recorded[1][0] == CallbackEvent.TURN_END
    assert recorded[1][1] == (proc, response, tool_results)
