# Program Linking Feature

## Overview

The Program Linking feature allows LLM processes to communicate with each other through the "spawn" system call. This enables primary processes to delegate queries to specialized processes that have different system prompts, preloaded files, and parameters.

## Use Cases

1. **Knowledge Specialization**: Create specialized processes with domain-specific knowledge by preloading different files for each process.
   
2. **Task Distribution**: Split complex tasks among multiple processes, each optimized for a specific subtask.
   
3. **Context Separation**: Keep large reference documents in separate processes to maintain clean context spaces.

## Configuration

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

## Example

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
3. Returns the response to the primary LLM

When the primary LLM uses the spawn tool, the query is executed by the linked program asynchronously, and the result is returned as part of the conversation.

## Best Practices

1. **Clear System Prompts**: Make sure the primary LLM's system prompt explains the available expert programs
2. **Specialized Knowledge**: Preload relevant files in each linked program to create true experts
3. **Appropriate Delegation**: Train the main LLM to delegate only relevant queries to the specialized programs