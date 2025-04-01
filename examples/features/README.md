# LLMProc Features

This directory contains examples of LLMProc's core features, each demonstrated in a standalone configuration file.

## Basic Features

- **preload.toml**: Demonstrates file preloading for enhanced context
- **env-info.toml**: Shows how to add environment information to system prompts
- **prompt-caching.toml**: Explains prompt caching for token efficiency
- **mcp.toml**: Demonstrates basic Model Context Protocol tool usage
- **token-efficient-tools.toml**: Shows token-efficient tool use configuration
- **fork.toml**: Demonstrates the fork system call for process duplication
- **function_tools.py**: Shows how to register Python functions as LLM tools with the new fluent API

## Advanced Features

- **file-descriptor/**: Examples of file descriptor system for handling large outputs
- **program-linking/**: Examples of LLM-to-LLM communication via spawn tool

## Usage

Run any example with the CLI tool:

```bash
llmproc-demo ./examples/features/preload.toml
```

For Python script examples:

```bash
python ./examples/features/function_tools.py
```