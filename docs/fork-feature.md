# Fork System Call Feature

The fork system call allows an LLM process to create copies of itself to handle multiple tasks in parallel, similar to the Unix `fork()` system call but with advantages specific to LLM-based applications.

## Overview

The fork feature enables an LLM process to:

1. Create multiple copies of itself, each with the full conversation history
2. Process independent tasks in parallel without filling up the context window
3. Combine results from multiple forked processes into a single response

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

1. A `fork_tool` function in `tools/fork.py` that handles forking requests
2. A `fork_process` method in `LLMProcess` that creates deep copies of a process
3. Support in the LLMProcess initialization to register the fork tool when enabled

### Process Forking

When a process is forked:

1. A new LLMProcess instance is created with the same program configuration
2. The entire conversation state is deep-copied to the new process
3. All preloaded content and system prompts are preserved
4. The forked process runs independently with its own query
5. Results from all forked processes are collected and returned to the parent process

## Differences from Unix Fork

While inspired by the Unix fork() system call, the LLMProc fork implementation has some key differences:

1. It creates multiple forks at once rather than a single child process.
2. Each fork is given a specific prompt/task rather than continuing execution from the fork point.
3. The parent immediately waits for all child processes and collects their results.

## Example

See `examples/fork.toml` for a complete example program configuration that demonstrates the fork system call.

## Limitations and Future Work

- Current implementation executes all forked processes sequentially, though the API supports parallel execution.
- Future versions may add a more sophisticated process management system.
- The Unix Fork-Exec pattern could be implemented by combining fork with the ability to change the system prompt or tools.