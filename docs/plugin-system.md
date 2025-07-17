# Plugin System

LLMProc exposes a single plugin system that covers both observational callbacks
and behavioral hooks. Plugins are regular Python objects added with
`process.add_plugins(obj)` or `program.add_plugins(obj)` before `start()`.

Each plugin may implement any combination of callback or hook methods. Missing
methods are ignored. Hook methods can modify data by returning new values while
callback methods typically return ``None`` and simply observe events.

## Event Categories

Events are defined in `llmproc.plugin.events` and classified as either
`OBSERVATIONAL` or `BEHAVIORAL`:

- Observational events map to the former callback system and log errors without
  stopping execution.
- Behavioral events correspond to hook methods and propagate exceptions for
  fail-fast behavior.

See `CallbackEvent` and `HookEvent` in `llmproc.plugin.events` for the
available names.

## Usage Example

```python
from llmproc import LLMProgram
from llmproc.plugin import PluginEventRunner

class MyPlugin:
    def tool_start(self, tool_name, tool_args, *, process):
        print(f"Tool {tool_name} started")

    async def hook_user_input(self, user_input, process):
        return user_input.strip()

program = LLMProgram(model_name="model", provider="openai")
program.add_plugins(MyPlugin())
process = await program.start()
```

Return values from hook methods replace the previous value. If a method returns
``None`` the value is unchanged. Callback methods that raise exceptions only
produce log warnings so execution continues.

---
[← Back to Documentation Index](index.md)
