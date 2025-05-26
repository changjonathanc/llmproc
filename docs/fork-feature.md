# Fork System Call Feature

The fork system call allows an LLM process to create copies of itself to handle multiple tasks in parallel, similar to the Unix `fork()` system call but with advantages specific to LLM-based applications.

## Overview

The fork feature enables an LLM process to:

1. Create multiple copies of itself, each with the full conversation history
2. Process independent tasks in parallel without filling up the context window
3. Combine results from multiple forked processes into a single response

> ⚠️ **Access Control**: The fork tool is only available to processes with ADMIN access level. Child processes created by fork are given WRITE access level by default, which prevents them from calling fork again.

## Key Benefits

- **Shared Context**: Each forked process inherits the full conversation history, ensuring continuity and context preservation.
- **Parallel Processing**: Multiple tasks can be processed simultaneously, improving efficiency.
- **Prompt Caching**: Shared conversation history prefix can be cached for performance benefits.
- **Focus**: Each fork can concentrate on a specific subtask without distraction.

## Configuration

To enable the fork system call, add it to the `[tools]` section in your TOML configuration file:

```toml
[tools]
enabled = ["fork"]
```

You can also combine it with other system tools:

```toml
[tools]
enabled = ["fork", "spawn"]
```

## Usage

Once enabled, the fork tool is available to the LLM through the standard tool-calling interface.

### Tool Schema

```json
{
  "name": "fork",
  "description": "Create copies of the current process to handle multiple tasks in parallel. Each copy has the full conversation history.",
  "input_schema": {
    "type": "object",
    "properties": {
      "prompts": {
        "type": "array",
        "description": "List of prompts/instructions for each forked process",
        "items": {
          "type": "string",
          "description": "A specific task or query to be handled by a forked process"
        }
      }
    },
    "required": ["prompts"]
  }
}
```

### When to Use Fork

The fork system call is ideal for:

1. Breaking complex tasks into parallel subtasks
2. Performing multiple independent operations simultaneously
3. Processing data from multiple sources in parallel
4. Executing operations that would otherwise consume excessive context length

### Example Usage Scenarios

- **Research**: Fork to read and analyze multiple documents in parallel.
- **Code Analysis**: Fork to examine different parts of a codebase simultaneously.
- **Data Processing**: Fork to process different data segments independently.
- **Content Generation**: Fork to generate multiple variations of content in parallel.

## Implementation Details

The fork feature is implemented through:

1. A `fork_tool` function in `tools/builtin/fork.py` that handles forking requests
2. The `_fork_process` method in `LLMProcess` that manages the forking process
3. The standard `ToolManager` system for registering and calling the fork tool
4. Access level controls to manage which processes can use the fork tool

The fork tool is a first-class provider-agnostic tool executed via ToolManager just like any other tool, following the Unix-inspired model where tools are accessed through a consistent interface.

### Process Forking

When a process is forked:

1. A new process is created via `program.start()` to ensure proper initialization
2. The entire conversation state is deep-copied to the new process using the `ProcessSnapshot` mechanism
3. All preloaded content and system prompts are preserved
4. File descriptors are properly cloned to maintain independence between processes
5. The access level is set to WRITE for child processes, enforcing security boundaries
6. The forked process runs independently with its own query using the provider's appropriate executor
7. Results from all forked processes are collected and returned to the parent process

The implementation uses separate message buffers (`msg_prefix` and `tool_results_prefix`) to maintain proper causal ordering of messages and tool results, ensuring that tools executed later in a turn can see results from earlier tools.

## Differences from Unix Fork

While inspired by the Unix fork() system call, the LLMProc fork implementation has some key differences:

1. It creates multiple forks at once rather than a single child process.
2. Each fork is given a specific prompt/task rather than continuing execution from the fork point.
3. The parent immediately waits for all child processes and collects their results.

## Example

See `examples/fork.toml` for a complete example program configuration that demonstrates the fork system call.

## Current Capabilities and Future Work

- The implementation executes all forked processes in parallel with asyncio.gather(), with the parent waiting for all children to complete.
- The access level system (AccessLevel.ADMIN for parents, AccessLevel.WRITE for children) enforces security boundaries, preventing unauthorized fork operations.
- Each process has complete state isolation through deep copying and proper file descriptor cloning.
- The streaming implementation ensures proper causal ordering of messages and tool results.
- Future enhancements may include:
  - More sophisticated process management and job control features
  - Implementation of the Unix Fork-Exec pattern by combining fork with system prompt or tool modifications
  - Performance optimizations for large state handling
