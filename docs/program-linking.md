# Program Linking Feature

## Overview

The Program Linking feature allows LLMProcess instances to communicate with each other through a specialized "spawn" tool. This enables primary LLMs to delegate queries to specialized LLMs that have different configurations, system prompts, and preloaded content.

## Use Cases

1. **Knowledge Specialization**: Create expert models with specialized knowledge by preloading different files for each model.
   
2. **Task Distribution**: Split complex tasks among multiple LLMs, each optimized for a specific subtask.
   
3. **Context Separation**: Keep large reference documents in separate LLMs to maintain clean context spaces.

## Configuration

Program linking is configured in two places:

1. The main TOML configuration's `[linked_programs]` section specifies which programs to link:

```toml
[linked_programs]
repo_expert = "./repo_expert.toml"
code_helper = "./code_helper.toml"
```

2. The `[tools]` section must include "spawn" in the enabled tools:

```toml
[tools]
enabled = ["spawn"]
```

## Example

```toml
# main.toml - Primary LLM configuration
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
# repo_expert.toml - Expert LLM configuration
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