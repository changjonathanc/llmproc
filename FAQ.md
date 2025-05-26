# LLMProc API Design FAQ

## Core Architecture

### Why use "process/program" terminology instead of "agent"?

The term "agent" has inconsistent meanings in the AI community. Some use it for LLMs with tool-calling capabilities, others for user-facing assistants, and still others for autonomous systems with their own goals. This ambiguity makes it problematic as a core architectural concept.

Instead, LLMProc adopts Unix-inspired process/program terminology. This provides clearer semantics that most developers already understand: programs define what to run, processes are running instances. This distinction maps cleanly to the different concerns in LLM applications.

### Why separate Program and Process in the API?

The separation between configuration (Program) and runtime state (Process) emerged from practical experience building LLM applications. As systems grew more complex, keeping these concerns separate became increasingly important.

Program objects are immutable configurations that define model parameters, tools, and system prompts. Process objects contain the mutable state - conversation history, current context, and runtime resources. This separation creates cleaner architecture with several benefits:

- Validation happens at a clear boundary when creating a process
- Configuration can be analyzed, serialized, and shared independently
- Runtime resources can be properly managed through process lifecycles
- Testing becomes simpler with clear separation of concerns

The typical usage pattern reflects this separation:

```python
program = LLMProgram(model_name="claude-3-7-sonnet", provider="anthropic")
process = await program.start()  # Validation happens here
```

### Why does run() not return a string?

The run() method returns a RunResult object rather than a string because execution involves more than just generating text. The method follows the semantics of a debugger's run command: execute until a natural stopping point.

This design accounts for several realities of LLM execution:
- An LLM might complete a turn without producing a text response
- Execution often involves multiple tool calls that need to be tracked
- Applications frequently need metadata about the execution process

For users who just want the text response, the API provides a clear path:

```python
result = await process.run("Hello")
message = process.get_last_message()  # Get text response if available
```

### Is `LLMProcess.run` safe to call concurrently?

No. `LLMProcess` instances are **not** thread-safe. Each process tracks
conversation history and other mutable state internally. Calling `run()` from
multiple tasks or threads at the same time can corrupt that state. The async
API is designed so you can run separate processes concurrently without
blocking the event loop—**not** for running the same process in parallel.
Always await one call before starting the next on a given instance.

## Configuration and Tools

### What configuration formats does LLMProc support?

LLMProc accepts configuration in multiple formats:

- **TOML** – the original format used by the project
- **YAML** – follows the same schema as the TOML examples
- **Python dictionaries** – create programs programmatically with `LLMProgram.from_dict()`

Both TOML and YAML files share identical fields, so choose whichever format best fits your workflow.

### What's the purpose of the file descriptor system?

The file descriptor approach solves several problems elegantly:
- It provides automatic paging for all tool calls, preventing context window issues
- It creates a unified paging interface without requiring each tool to implement its own
- The LLM can reference content by identifier without copying large text
- It follows Unix-like patterns familiar to developers

### Why implement program linking as a "spawn" tool?

We designed the spawn tool to follow the idea of program linking in Unix:
- It gives LLMs agency to decide when to delegate tasks
- Each process runs independently with its own configuration and state
- It creates clear API boundaries between processes
- The pattern maps to how programs link to other programs in computing systems

This has been particularly useful for creating expert systems where different models specialize in different tasks.

## Advanced Features

### When to use fork() and goto()?

Both of these tools are powerful and can sometimes be used to achieve similar goals. Here's when to use each:

- Fork is uniquely suited for performing multiple tasks in parallel. Use it when you need to explore different solution paths simultaneously. The challenge is managing potential race conditions across forked processes.

- Goto was designed specifically to address these race condition issues. It keeps a single task running, letting you reset the conversation to a previous point when needed.

- You can use goto() effectively after completing tasks that consumed a lot of context window. For example:
  - After code renaming across multiple files (with many tool calls), use goto() to compact the steps into a single summary.
  - When debugging goes down an incorrect path, use goto() to discard those steps, preserve the lessons learned, and continue with a better approach.

### How do I monitor what's happening during process execution?

We have callbacks! The callback system provides visibility into execution without blocking the main application flow:

```python
callbacks = {
    "on_tool_start": lambda tool_name, args: log_start(tool_name),
    "on_tool_end": lambda tool_name, result: log_result(result),
    "on_completion": lambda result: save_metrics(result)
}

result = await process.run("Your query", callbacks=callbacks)
```

This approach enables real-time monitoring, custom metrics collection, and integration with external logging systems. You can see examples of callback usage in the CLI implementation at `src/llmproc/cli.py`.

### How do I configure API retry behavior?

LLMProc supports configurable retry behavior for API calls through environment variables. You can control the number of retry attempts, initial wait time, and maximum wait time.

See the [Environment Variables documentation](docs/environment-variables.md#retry-configuration) for details on configuring:
- `LLMPROC_RETRY_MAX_ATTEMPTS`
- `LLMPROC_RETRY_INITIAL_WAIT`
- `LLMPROC_RETRY_MAX_WAIT`

## Configuration

### What's the recommended way to configure prompts?

LLMProc supports two naming styles for prompt configuration:

- **Recommended**: `system` and `user` (concise, modern)
- **Also supported**: `system_prompt` and `user_prompt` (explicit, legacy)

We recommend using the shorter forms for consistency with other configuration sections:

```toml
[prompt]
system = "You are a helpful assistant."
user = "Hello!"

[tools]
builtin = ["calculator", "read_file"]  # Not "builtin_tools"
```

The longer forms (`system_prompt`, `user_prompt`) remain fully supported for backward compatibility. Use them if you prefer more explicit field names or have existing configurations.

Both styles work identically in TOML and YAML configurations.

The retry mechanism uses exponential backoff to handle transient API errors gracefully.
