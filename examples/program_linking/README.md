# Program Linking Example

This example demonstrates how to use the program linking feature in LLMProc.

## Overview

Program linking allows a primary LLM to delegate queries to specialized LLMs through the "spawn" tool. This enables:

1. Creating specialized expert models with focused knowledge
2. Delegating specific tasks to models with different capabilities
3. Distributing complex tasks across multiple LLMs

## Files

- `main.toml`: Configuration for the primary LLM with spawn tool enabled
- `repo_expert.toml`: Configuration for a specialized LLM with knowledge of the LLMProc project

## Usage

Run the example:

```bash
llmproc-demo ./examples/program_linking/main.toml
```

Then ask questions that involve delegating to the repo expert:

```
User: What are the main features of LLMProc according to the repository?

Model: Let me check with the repository expert.
(The model will use the spawn tool to ask the repo_expert about LLMProc features)
```

## Implementation Details

The program linking feature consists of:

1. A `[linked_programs]` section in the TOML configuration
2. The `spawn` tool that allows the primary LLM to communicate with linked programs
3. Background initialization of all linked programs

## Configuration Example

```toml
# Primary LLM
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"
display_name = "Claude Haiku"

[prompt]
system_prompt = "You are Claude, a helpful AI assistant. Keep all responses under 3 sentences for brevity."

[tools]
enabled = ["spawn"]

[linked_programs]
repo_expert = "./repo_expert.toml"
```

```toml
# Expert LLM
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"

[prompt]
system_prompt = """You are a helpful assistant with knowledge of the LLMProc project.
Use the preloaded project files to answer questions about LLMProc functionality and usage."""

[preload]
# Preload key project files for reference
files = [
  "../../README.md",
  "../../pyproject.toml",
  "../../src/llmproc/llm_process.py",
]
```