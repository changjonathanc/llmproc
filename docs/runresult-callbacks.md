# RunResult and Callbacks

## Overview

The LLMProc library now provides detailed metrics and real-time monitoring through two key features:

1. **RunResult**: A structured object that captures metrics and diagnostic information about each run
2. **Callbacks**: A mechanism to monitor and respond to events during process execution

These features enable better diagnostics, resource tracking, and integration with other systems.

## RunResult Class

The `RunResult` class captures detailed information about each execution of an LLMProcess:

```python
class RunResult:
    """Contains metadata about a process run."""

    api_call_infos: List[Dict[str, Any]]  # Raw API response data
    api_calls: int                        # Number of API calls made
    start_time: float                     # When the run started
    end_time: Optional[float]             # When the run completed
    duration_ms: int                      # Duration in milliseconds
```

### Usage Example

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml("examples/minimal.toml")
    process = await program.start()

    # Run returns a RunResult object
    run_result = await process.run("Hello, how are you?")

    # Get the response content
    response = process.get_last_message()
    print(f"Response: {response}")

    # Use the run metrics
    print(f"Run completed in {run_result.duration_ms}ms")
    print(f"API calls: {run_result.api_calls}")

    # Examine detailed API information
    for i, api_info in enumerate(run_result.api_call_infos):
        print(f"API call {i+1}:")
        if "model" in api_info:
            print(f"  Model: {api_info['model']}")
        if "usage" in api_info:
            print(f"  Usage: {api_info['usage']}")

asyncio.run(main())
```

## Callbacks

Callbacks allow you to monitor and respond to events during execution, including:

- Tool start events
- Tool completion events
- Model response events

### Available Callbacks

```python
callbacks = {
    # Called when a tool starts execution
    "on_tool_start": lambda tool_name, args: print(f"Starting tool: {tool_name}"),

    # Called when a tool completes execution
    "on_tool_end": lambda tool_name, result: print(f"Tool completed: {tool_name}"),

    # Called when a model generates a response
    "on_response": lambda content: print(f"Response: {content[:30]}...")
}
```

### Using Callbacks

```python
import asyncio
from llmproc import LLMProgram
import time

async def main():
    program = LLMProgram.from_toml("examples/mcp.toml")
    process = await program.start()

    # Define stateful callbacks with progress tracking
    active_tools = set()

    def on_tool_start(tool_name, args):
        active_tools.add(tool_name)
        print(f"⚙️ Running tool: {tool_name} ({len(active_tools)} active tools)")

    def on_tool_end(tool_name, result):
        if tool_name in active_tools:
            active_tools.remove(tool_name)
        print(f"✓ Completed tool: {tool_name} ({len(active_tools)} active tools)")

    callbacks = {
        "on_tool_start": on_tool_start,
        "on_tool_end": on_tool_end,
        "on_response": lambda content: print(f"💬 Response: {content[:50]}...")
    }

    # Start time for tracking
    start_time = time.time()

    # Run with callbacks
    run_result = await process.run(
        "Use available tools to search for information about Python asyncio.",
        callbacks=callbacks
    )

    # Calculate elapsed time
    elapsed = time.time() - start_time

    # Get the assistant's response
    response = process.get_last_message()
    print("\nFinal response:")
    print(response)

    # Display run metrics
    print(f"\nRun completed in {elapsed:.2f}s using {run_result.api_calls} API calls")

asyncio.run(main())
```

## Advanced Usage: Progress Tracking

You can implement a spinner or progress tracker for tool execution:

```python
import asyncio
import sys
import time
from llmproc import LLMProgram

class ToolProgressTracker:
    def __init__(self):
        self.active_tools = set()
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_idx = 0
        self.should_run = False
        self.task = None

    def start(self):
        self.should_run = True
        self.task = asyncio.create_task(self._run_spinner())

    async def _run_spinner(self):
        while self.should_run and self.active_tools:
            tools_str = ", ".join(self.active_tools)
            spinner = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
            sys.stdout.write(f"\r{spinner} Processing: {tools_str}")
            sys.stdout.flush()
            self.spinner_idx += 1
            await asyncio.sleep(0.1)

        # Clear the spinner line
        if self.active_tools:
            sys.stdout.write("\r" + " " * (20 + sum(len(t) for t in self.active_tools)) + "\r")
            sys.stdout.flush()

    def stop(self):
        self.should_run = False
        if self.task:
            self.task.cancel()

        # Clear any remaining spinner
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    def on_tool_start(self, tool_name, tool_args):
        """Callback for when a tool starts execution"""
        self.active_tools.add(tool_name)
        if not self.task or self.task.done():
            self.start()

    def on_tool_end(self, tool_name, result):
        """Callback for when a tool completes execution"""
        if tool_name in self.active_tools:
            self.active_tools.remove(tool_name)

        if not self.active_tools:
            self.stop()

async def main():
    # Create the progress tracker
    tracker = ToolProgressTracker()

    # Load and start the program
    program = LLMProgram.from_toml("examples/mcp.toml")
    process = await program.start()

    # Set up callbacks with the tracker
    callbacks = {
        "on_tool_start": tracker.on_tool_start,
        "on_tool_end": tracker.on_tool_end,
        "on_response": lambda content: print(f"\nResponse: {content[:50]}...")
    }

    # Run with callbacks
    print("Working on your request...")
    run_result = await process.run("Search for information using the available tools", callbacks=callbacks)

    # Get the assistant's response
    response = process.get_last_message()
    print(f"\nFinal response: {response}")

    # Display run metrics
    print(f"\nRun completed in {run_result.duration_ms}ms using {run_result.api_calls} API calls")

asyncio.run(main())
```

## Implementation Details

### RunResult Implementation

- Created at the start of each run
- Updated throughout execution with API call information
- Tracks time with start_time, end_time, and duration_ms
- Completed with the complete() method at the end of execution
- Returned from all process.run() calls

### Callbacks Implementation

- Registered at the run() method call
- Executed during process execution at specific points
- Made optional to maintain a clean API
- Designed to handle exceptions gracefully
- Support both simple and complex callback functions

## Best Practices

1. **Error Handling**: Make your callbacks robust with proper error handling
2. **Performance**: Keep callbacks lightweight to avoid slowing down execution
3. **Stateful Tracking**: Use stateful callbacks to track complex progress indicators
4. **Tool Monitoring**: Focus on tool execution, as this is typically the most time-consuming part
5. **Resource Tracking**: Use RunResult to track API usage for cost monitoring
6. **Diagnostic Logging**: Store RunResult information for later analysis and optimization
