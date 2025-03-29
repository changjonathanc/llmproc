# Claude Code Examples

This directory contains examples for using Claude Code capabilities within LLMProc.

## Examples

- **claude-code.toml**: Basic Claude Code configuration with tools and system prompt
- **dispatch-agent.toml**: Claude Code configuration with dispatch agent functionality

Claude Code provides code-related tools including file operations, search capabilities, and more.

## Usage

```bash
# Start a basic Claude Code session
llmproc-demo ./examples/claude-code/claude-code.toml

# Use Claude Code with dispatch agent
llmproc-demo ./examples/claude-code/dispatch-agent.toml
```