# Release Notes - v0.10.0

## 🎉 Major Features

### Unified Process Configuration
- **`ProcessConfig` Dataclass** consolidates initialization parameters for `LLMProcess`.
- Encourages the `program.start()` pattern for validated process creation.

### Plugin-Based Hook System
- New hook framework allows modifying user input, tool calls, and results.

### File Descriptor Plugin
- File descriptor handling now implemented via `FileDescriptorPlugin`.
- Automatically converts large inputs and outputs using `hook_user_input` and `hook_tool_result` events.


## 🛠️ Improvements
- Deprecated `enabled_tools` attribute removed from `LLMProcess`.
- Runtime configuration consolidated with `ProcessConfig` validation.
- `RunResult.add_tool_call` now records tool arguments under `tool_args` to
  match callback APIs.
- `tool_start` callbacks must use the `tool_args` parameter. The deprecated
  `args` alias is removed.
- The method now accepts `tool_name` as its first parameter instead of `name`.
- Improved error messages and documentation examples.

## Breaking Changes
- `LLMProcess` constructor requires a `ProcessConfig` instance.
- File descriptor features provided exclusively via the `FileDescriptorPlugin`.

---
For detailed API documentation and more examples, visit the [documentation](https://github.com/cccntu/llmproc/tree/main/docs).

---
[← Back to Documentation Index](../index.md)
