"""Tests for the TURN_START and TURN_END callback events."""

from llmproc.callbacks import CallbackEvent


class DummyProcess:
    """Simplified process that forwards events to callbacks."""

    def __init__(self) -> None:
        self.callbacks = []

    def trigger_event(self, event: CallbackEvent, *args) -> None:
        event_name = event.value
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(*args)
            elif callable(cb):
                cb(event, *args)


def test_turn_callbacks():
    """Verify TURN_START and TURN_END events reach registered callbacks."""
    recorded = []

    def callback(event: CallbackEvent, *args) -> None:
        recorded.append((event, args))

    proc = DummyProcess()
    proc.callbacks.append(callback)

    response = {"text": "hi"}
    tool_results = ["tool_result"]
    proc.trigger_event(CallbackEvent.TURN_START, proc)
    proc.trigger_event(CallbackEvent.TURN_END, proc, response, tool_results)

    assert recorded[0][0] == CallbackEvent.TURN_START
    assert recorded[0][1][0] is proc
    assert recorded[1][0] == CallbackEvent.TURN_END
    assert recorded[1][1] == (proc, response, tool_results)
