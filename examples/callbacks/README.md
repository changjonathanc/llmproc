# Callback Examples

This directory demonstrates LLMProc's **flexible callback signature** capabilities using Flask/pytest-style parameter injection.

## Key Benefits

- **Clean signatures** - Declare only the parameters you actually need
- **Performance** - No unnecessary parameter processing with caching
- **Compatibility** - Legacy `*, process` signatures still work
- **Flexibility** - Mix different signature styles freely

## Examples

### [flexible_signatures_cookbook.py](flexible_signatures_cookbook.py)
Comprehensive cookbook showing all callback patterns:
- Basic patterns (minimal signatures)
- Selective patterns (choose what you need)
- Advanced patterns (full context when needed)
- Legacy compatibility patterns
- Mixed patterns (use different styles together)
- Performance comparison

### [progressive_complexity.py](progressive_complexity.py)
Shows how to start simple and gradually add complexity:
- Level 1: Basic logging
- Level 2: Simple metrics
- Level 3: Timing and performance
- Level 4: Process monitoring
- Level 5: Advanced analytics
- Level 6: Adaptive callbacks

### [protocol_example.py](protocol_example.py)
Demonstrates IDE support with `PluginProtocol`:
- Type hints for autocomplete
- Different inheritance patterns
- Performance considerations

## Quick Start

```python
from llmproc import LLMProgram

class MyCallbacks:
    def tool_start(self, tool_name):                    # Basic: just the tool name
        print(f"🔧 Starting {tool_name}")

    def tool_end(self, tool_name, result):              # Selective: name and result
        print(f"✅ {tool_name} completed")

    def response(self, content, process):               # Full context when needed
        tokens = process.count_tokens()
        print(f"💬 Response: {len(content)} chars, {tokens} tokens")

# Use with any LLM program
program = LLMProgram.from_yaml("config.yaml")
process = await program.start()
process.add_plugins(MyCallbacks())
await process.run("Your prompt here")
```

## Available Parameters by Event

- **tool_start**: `tool_name`, `tool_args`, `process`
- **tool_end**: `tool_name`, `result`, `process`  
- **response**: `content`, `process`
- **turn_start**: `process`, `run_result` (optional)
- **turn_end**: `response`, `tool_results`, `process`
- **api_request**: `api_request`, `process`
- **api_response**: `response`, `process`
- **run_end**: `run_result`, `process`

## Run Examples

```bash
# Comprehensive patterns cookbook
python examples/callbacks/flexible_signatures_cookbook.py

# Progressive complexity demonstration  
python examples/callbacks/progressive_complexity.py

# Protocol and IDE support
python examples/callbacks/protocol_example.py
```
