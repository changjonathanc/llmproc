# Callback System

LLMProc provides a minimal, flexible callback system for monitoring process execution events.

## Overview

The callback system is designed to:

1. Support both function-based and class-based callbacks
2. Provide a minimal set of event types covering core operations
3. Propagate events from child processes to parent process callbacks

## Event Types

The system supports several standard event types as defined in the `CallbackEvent` enum:

- `TOOL_START` - Called when a tool is about to be executed
- `TOOL_END` - Called when a tool execution completes
- `RESPONSE` - Called when model generates a response
- `API_REQUEST` - Called right before an API request is sent
- `API_RESPONSE` - Called when an API response is received
- `TURN_START` - Called at the start of each turn
- `TURN_END` - Called at the end of each turn
- `STDERR_WRITE` - Called when text is appended to the stderr log

## Using Callbacks

### Registering Callbacks

Register callbacks with a process after creation:

```python
from llmproc import LLMProgram
from llmproc.callbacks import CallbackEvent

# Create a process
program = LLMProgram.from_toml("config.toml")
process = await program.start()

# Add a function-based callback
process.add_callback(my_callback_function)

# Add a class-based callback
process.add_callback(MyCallbackClass())

# Chain methods
process = await program.start().add_callback(my_callback)
```

### Function-Based Callbacks

Function-based callbacks receive the event enum and all event arguments:

```python
def my_callback(event, *args, **kwargs):
    if event == CallbackEvent.TOOL_START:
        tool_name, tool_args = args
        print(f"Tool started: {tool_name}")
    elif event == CallbackEvent.TOOL_END:
        tool_name, result = args
        print(f"Tool completed: {tool_name}")
    elif event == CallbackEvent.RESPONSE:
        text = args[0]
        print(f"Response: {text[:30]}...")
```

### Class-Based Callbacks

Class-based callbacks can implement methods named after the event string values:

```python
class MyCallbacks:
    def tool_start(self, tool_name, tool_args):
        print(f"Tool started: {tool_name}")

    def tool_end(self, tool_name, result):
        print(f"Tool completed: {tool_name}")

    def response(self, text):
        print(f"Response: {text[:30]}...")
```

### Async Callbacks

Callbacks can also be asynchronous. Both function-based and class-based callbacks
may be defined as ``async`` and will automatically run on the process event
loop. A callback class may freely mix synchronous and asynchronous methods.

```python
async def my_async_callback(event, *args, **kwargs):
    await log_event(event)

class MixedCallbacks:
    def tool_start(self, tool_name, tool_args):
        print("sync start")

    async def response(self, text):
        await log_response(text)
```

## Callback Arguments

Each event type passes specific arguments to callbacks:

- `TOOL_START` - `(tool_name, tool_args)`
- `TOOL_END` - `(tool_name, result)`
- `RESPONSE` - `(text)`
- `API_REQUEST` - `(payload_dict)`
- `API_RESPONSE` - `(response_obj)`
- `TURN_START` - `(process, run_result=None)` **(Enhanced in v0.9.3)**
- `TURN_END` - `(process, response_obj, tool_results)`
- `STDERR_WRITE` - `(text)`

## Smart Parameter Passing (v0.9.3+)

LLMProc now supports intelligent parameter filtering based on callback method signatures. This means:

- **Backward Compatibility**: Existing callbacks continue to work unchanged
- **Smart Filtering**: Callbacks only receive parameters they can accept
- **Enhanced Features**: New parameters like `run_result` can be used for advanced functionality

### TURN_START with RunResult Support

The `TURN_START` event now supports an optional `run_result` parameter for real-time cost monitoring:

```python
class CostMonitoringCallback:
    def __init__(self, cost_limit=5.0):
        self.cost_limit = cost_limit

    def turn_start(self, process, run_result=None):
        """Monitor costs before each turn starts."""
        if run_result and run_result.usd_cost > self.cost_limit:
            raise Exception(f"Cost ${run_result.usd_cost:.4f} exceeds limit ${self.cost_limit:.2f}")

        if run_result:
            print(f"Current cost: ${run_result.usd_cost:.4f}")

    # Old-style callback still works
    def turn_end(self, process, response_obj, tool_results):
        print("Turn completed")
```

### Legacy Compatibility

Existing callbacks continue to work without modification:

```python
# This old-style callback still works perfectly
def old_turn_start(process):
    print("Starting new turn")

# This new-style callback gets additional features
def new_turn_start(process, run_result=None):
    print(f"Starting turn, cost so far: ${run_result.usd_cost if run_result else 0:.4f}")
```

## Example: Timing Callback

Here's a simple callback for tracking tool execution time:

```python
class TimingCallback:
    def __init__(self):
        self.tools = {}
        self.current_tools = {}

    def tool_start(self, tool_name, tool_args):
        if tool_name not in self.tools:
            self.tools[tool_name] = {"calls": 0, "total_time": 0}

        self.tools[tool_name]["calls"] += 1
        self.current_tools[tool_name] = time.time()

    def tool_end(self, tool_name, result):
        if tool_name in self.current_tools:
            elapsed = time.time() - self.current_tools[tool_name]
            self.tools[tool_name]["total_time"] += elapsed
            del self.current_tools[tool_name]

    def get_stats(self):
        stats = {}
        for name, data in self.tools.items():
            calls = data["calls"]
            total = data["total_time"]
            avg = total / calls if calls > 0 else 0
            stats[name] = {
                "calls": calls,
                "total_time": total,
                "avg_time": avg
            }
        return stats
```

## Example: API Logging Callback

Use a callback to capture the raw request and response data for each API call:

```python
class APILogger:
    def __init__(self):
        self.calls = []

    def api_request(self, payload):
        self.calls.append({"request": payload})

    def api_response(self, response):
        if self.calls:
            self.calls[-1]["response"] = response
```

## Callback Inheritance

Callbacks registered on a parent process are automatically propagated to child processes created via fork:

```python
# Register callback on parent
parent = await program.start()
parent.add_callback(my_callback)

# Fork the process
child = await parent.fork_process()
# Child inherits callbacks automatically
```

See `examples/scripts/callback_demo.py` for a complete demonstration.
