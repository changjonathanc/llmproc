# Tool Registration Callback

The `register_tool` decorator supports an optional `on_register` callback. This function is executed once when the tool is registered with the `ToolManager`.

## Purpose

`on_register` allows a tool to perform initialization logic or modify the `ToolManager` when it becomes available. This avoids scattering feature flags or ad-hoc checks throughout the codebase.

The callback receives the tool's name and the `ToolManager` instance. It can set attributes or perform any other setup required for the tool to work correctly.

This mechanism runs **only during registration**. It is separate from the runtime [callback system](callbacks.md) that reports events such as tool execution and model responses.

## Example: Enabling message IDs for the `goto` tool

The `goto` tool requires message IDs to be enabled so it can reference previous conversation points. Using `on_register`, the tool can enable this functionality automatically:

```python
from llmproc.tools.function_tools import register_tool

# Callback executed when the tool is registered
def enable_message_ids(tool_name, tool_manager):
    tool_manager.message_ids_enabled = True

@register_tool(
    name="goto",
    description="Reset conversation to a previous message by ID",
    on_register=enable_message_ids,
)
async def handle_goto(position: str, message: str, runtime_context: dict | None = None):
    """Reset to a previous message and continue from there."""
    ...  # implementation omitted
```

When `ToolManager.process_function_tools()` registers `handle_goto`, the callback sets `message_ids_enabled` on the manager. This eliminates the need for a global feature flag or additional configuration steps.

Use `on_register` whenever a tool needs to perform one-time initialization or customize the tool manager during setup.
