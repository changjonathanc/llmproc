# Program Linking Feature

## Overview

The Program Linking feature allows LLM processes to communicate with each other through the "spawn" system call. This enables primary processes to delegate queries to specialized processes that have different system prompts, preloaded files, and parameters.

Following Unix design principles, processes can spawn other processes to perform specialized tasks, creating a natural delegation pattern. For the design rationale behind this approach, see the [API Design FAQ](../FAQ.md#why-implement-program-linking-as-a-spawn-tool).

## Use Cases

1. **Knowledge Specialization**: Create specialized processes with domain-specific knowledge by preloading different files for each process.

2. **Task Distribution**: Split complex tasks among multiple processes, each optimized for a specific subtask.

3. **Context Separation**: Keep large reference documents in separate processes to maintain clean context spaces.

## Setup

Program linking is configured in two places:

1. The main TOML program's `[linked_programs]` section specifies which programs to link:

```toml
[linked_programs]
repo_expert = "./repo_expert.toml"
code_helper = "./code_helper.toml"
```

2. The `[tools]` section must include "spawn" in the enabled system calls:

```toml
[tools]
enabled = ["spawn"]
```

## Example Program

```toml
# main.toml - Primary LLM program
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"
display_name = "Claude Haiku"

[prompt]
system_prompt = """You are Claude, a helpful AI assistant.
You have access to the 'spawn' tool that lets you communicate with specialized experts."""

[parameters]
max_tokens = 1000

[tools]
enabled = ["spawn"]

[linked_programs]
repo_expert = "./repo_expert.toml"
```

```toml
# repo_expert.toml - Expert LLM program
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"

[prompt]
system_prompt = """You are a helpful assistant with specialized knowledge.
Use the preloaded project files to answer questions."""

[parameters]
max_tokens = 1000

[preload]
files = [
  "../../README.md",
  "../../pyproject.toml"
]
```

## Implementation Details

The spawn tool:
1. Takes a program name and query
2. Routes the query to the appropriate linked program
3. Returns the response along with metrics to the primary LLM

When the primary LLM uses the spawn tool, the query is executed by the linked program asynchronously, and the result is returned as part of the conversation.

### Spawning the Current Program

If no linked programs are configured, leave `program_name` blank. The spawn tool will create a fresh process from the current program and execute the provided prompt in that new context.

## Using Program Linking with the New API

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load the main program with linked programs
    main_program = LLMProgram.from_toml("path/to/main.toml")

    # Start the process with async initialization
    main_process = await main_program.start()

    # Run a query that will likely use the spawn tool
    run_result = await main_process.run(
        "Please use the repo expert to analyze the structure of this project."
    )

    # Get the assistant's response (which will include the expert's insights)
    response = main_process.get_last_message()
    print(f"Response (including expert knowledge): {response}")

    # Display run metrics
    print(f"API calls: {run_result.api_calls}")
    print(f"Duration: {run_result.duration_ms}ms")

# Run the async function
asyncio.run(main())
```

## Callback Support

You can monitor spawn tool usage with callbacks:

```python
import asyncio
from llmproc import LLMProgram

async def main():
    program = LLMProgram.from_toml("path/to/main.toml")
    process = await program.start()

    # Define callbacks for monitoring tool usage
    callbacks = {
        "on_tool_start": lambda tool_name, args: print(
            f"Starting {tool_name} with program={args.get('program_name')}"
        ),
        "on_tool_end": lambda tool_name, result: print(
            f"Expert response received from {result.get('program')}"
        )
    }

    # Run with callbacks
    run_result = await process.run(
        "Ask the code helper to explain how async/await works",
        callbacks=callbacks
    )

    # Get the final response
    response = process.get_last_message()
    print(f"Final response: {response}")

asyncio.run(main())
```

## Metrics and Diagnostics

The spawn tool now includes useful metrics in its results:

```python
# Example spawn tool result
{
    "program": "repo_expert",
    "query": "What is the structure of this project?",
    "response": "The project has a src/ directory with the main code...",
    "api_calls": 2  # Number of API calls made by the expert
}
```

These metrics are also aggregated in the main process's `RunResult` object.

## Best Practices

1. **Clear System Prompts**: Make sure the primary LLM's system prompt explains the available expert programs
2. **Specialized Knowledge**: Preload relevant files in each linked program to create true experts
3. **Appropriate Delegation**: Train the main LLM to delegate only relevant queries to the specialized programs
4. **Metrics Monitoring**: Use the RunResult object to track API usage across the entire program graph
5. **Real-time Monitoring**: Use callbacks to get insights into tool execution as it happens
