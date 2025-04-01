# Claude Code Examples

This directory contains examples for using Claude Code capabilities within LLMProc.

## Examples

- **claude-code.toml**: Basic Claude Code configuration with tools and system prompt
- **dispatch-agent.toml**: Claude Code configuration with dispatch agent functionality

Claude Code provides code-related tools including file operations, search capabilities, and more.

## Features

- **Program Linking**: Uses the enhanced description system for linked programs
- **Token-Efficient Tool Use**: Reduces token usage and improves latency
- **Thinking Mode**: Optional thinking mode for complex coding tasks
- **MCP Integration**: Uses Model Context Protocol for tool functionality

## Usage

```bash
# Start a basic Claude Code session with dispatch agent available
llmproc-demo ./examples/claude-code/claude-code.toml
```

## Configuration Example

```toml
[tools]
enabled = ["spawn"]

[linked_programs]
dispatch_agent = {path="./dispatch-agent.toml", description="Specialized agent for searching and exploring codebases efficiently"}
```

The dispatch agent is made available through program linking with descriptive metadata, allowing Claude Code to better understand when to use it.