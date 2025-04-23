# Callback System

LLMProc provides a minimal, flexible callback system for monitoring process execution events.

## Overview

The callback system is designed to:

1. Support both function-based and class-based callbacks
2. Provide a minimal set of event types covering core operations
3. Propagate events from child processes to parent process callbacks

## Event Types

The system supports three standard event types as defined in the `CallbackEvent` enum:

- `TOOL_START` - Called when a tool is about to be executed
- `TOOL_END` - Called when a tool execution completes
- `RESPONSE` - Called when model generates a response

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

## Callback Arguments

Each event type passes specific arguments to callbacks:

- `TOOL_START` - `(tool_name, tool_args)`
- `TOOL_END` - `(tool_name, result)`
- `RESPONSE` - `(text)`

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

See `examples/scripts/callback_demo_new.py` for a complete demonstration.