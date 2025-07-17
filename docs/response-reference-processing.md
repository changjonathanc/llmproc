# Processing Response References

This document explains how LLMProc extracts reference IDs from model responses and how those references are shared with child processes.

## Response Hook Flow

After each call to `LLMProcess.run()`, the process triggers the `response` hook on all registered plugins. The `FileDescriptorPlugin` implements this hook to extract reference IDs from the assistant message. When the plugin is enabled and reference support is active, it scans the assistant message for `<ref id="...">` blocks and stores them in the file descriptor manager. No additional internal method processes references.

## File Descriptor Sharing in Child Processes

When a process forks or uses the `spawn` tool, the file descriptor system ensures reference IDs remain accessible:

- **Forking**: `ProcessForkingMixin._fork_process()` clones the parent plugin's `FileDescriptorManager` into the child. All existing references are copied so the child can read them immediately.
- **Spawning**: `spawn_tool()` retrieves the parent's `FileDescriptorPlugin` from the runtime context. If file descriptors are enabled, the child process inherits the parent's manager configuration and reference IDs before execution.

References created inside a child remain isolated from the parent, but inherited references allow the child to access marked-up content created earlier. This behavior enables collaborative workflows across multiple processes without manually passing file descriptor data.
