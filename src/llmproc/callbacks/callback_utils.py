import inspect
import logging
from collections.abc import Callable
from typing import Any

from llmproc.callbacks import CallbackEvent

logger = logging.getLogger(__name__)

# Mapping of callback events to their expected parameter names
CALLBACK_EVENT_PARAMETERS = {
    CallbackEvent.TOOL_START: ["tool_name", "tool_args"],
    CallbackEvent.TOOL_END: ["tool_name", "result"],
    CallbackEvent.TURN_START: ["process", "run_result"],
    CallbackEvent.TURN_END: ["process", "response", "tool_results"],
    CallbackEvent.API_REQUEST: ["api_request"],
    CallbackEvent.API_RESPONSE: ["response"],
    CallbackEvent.RESPONSE: ["text"],
    CallbackEvent.STDERR_WRITE: ["message"],
}


def filter_callback_parameters(method: Callable, available_params: dict[str, Any]) -> dict[str, Any]:
    """Filter available parameters based on callback method signature."""
    try:
        sig = inspect.signature(method)
        filtered_params: dict[str, Any] = {}
        has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if has_var_keyword:
            return available_params

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in available_params:
                filtered_params[param_name] = available_params[param_name]
            elif param.default == param.empty:
                logger.debug(
                    "Required parameter '%s' not available for callback %s",
                    param_name,
                    method.__name__,
                )

        if not has_var_keyword and len(available_params) > len(filtered_params):
            extra_params = set(available_params.keys()) - set(filtered_params.keys()) - {"self"}
            if extra_params:
                logger.warning(
                    "Callback %s should accept **kwargs for forward compatibility. Ignoring parameters: %s",
                    method.__name__,
                    extra_params,
                )

        return filtered_params
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Error filtering parameters for callback %s: %s", method.__name__, e)
        return {}


def add_callback(process: Any, callback: Callable) -> None:
    """Add a callback to the process."""
    process.callbacks.append(callback)


def trigger_event(process: Any, event: CallbackEvent, *args, **kwargs) -> None:
    """Trigger an event to all registered callbacks."""
    if not process.callbacks:
        return

    event_name = event.value
    event_kwargs = kwargs.copy()
    if event in CALLBACK_EVENT_PARAMETERS and args:
        param_names = CALLBACK_EVENT_PARAMETERS[event]
        for i, arg in enumerate(args):
            if i < len(param_names):
                event_kwargs[param_names[i]] = arg

    for callback in process.callbacks:
        try:
            method = None
            pass_event = False

            if hasattr(callback, event_name):
                method = getattr(callback, event_name)
            elif callable(callback):
                method = callback
                pass_event = True

            if method is None:
                continue

            if not pass_event:
                filtered_kwargs = filter_callback_parameters(method, event_kwargs)
                if inspect.iscoroutinefunction(method):
                    fut = process._submit_to_loop(method(**filtered_kwargs))
                    fut.add_done_callback(
                        lambda f, en=event.name: logger.warning("Error in %s callback: %s", en, f.exception())
                        if f.exception()
                        else None
                    )
                else:
                    result = method(**filtered_kwargs)
                    if inspect.isawaitable(result):
                        fut = process._submit_to_loop(result)
                        fut.add_done_callback(
                            lambda f, en=event.name: logger.warning("Error in %s callback: %s", en, f.exception())
                            if f.exception()
                            else None
                        )
            else:
                if inspect.iscoroutinefunction(method):
                    fut = process._submit_to_loop(method(event, *args, **kwargs))
                    fut.add_done_callback(
                        lambda f, en=event.name: logger.warning("Error in %s callback: %s", en, f.exception())
                        if f.exception()
                        else None
                    )
                else:
                    result = method(event, *args, **kwargs)
                    if inspect.isawaitable(result):
                        fut = process._submit_to_loop(result)
                        fut.add_done_callback(
                            lambda f, en=event.name: logger.warning("Error in %s callback: %s", en, f.exception())
                            if f.exception()
                            else None
                        )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Error in %s callback: %s", event.name, e)
