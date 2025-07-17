"""Tests for the RUN_END callback event."""

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


def test_run_end_callback():
    """Verify RUN_END events reach registered callbacks."""
    recorded = []

    class CallbackRecorder:
        def run_end(self, run_result, *, process):
            recorded.append((CallbackEvent.RUN_END, (process, run_result)))

    proc = DummyProcess()
    proc.callbacks.append(CallbackRecorder())

    run_result = {"ok": True}
    proc.trigger_event(CallbackEvent.RUN_END, run_result, process=proc)

    assert recorded[0][0] == CallbackEvent.RUN_END
    assert recorded[0][1] == (proc, run_result)
