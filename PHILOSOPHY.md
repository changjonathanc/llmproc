# LLMProc Design Philosophy

## Core Principles

### LLM Agent as Process

We view LLM Agents as processes in a computing environment:
- It's defined by a program (TOML configuration)
- It receives input, executes, and returns output
- It maintains state between executions
- It interacts with the system through defined interfaces

### LLMProc as Kernel

The LLMProc library functions as a kernel:
- Implements system calls for LLM agents
- Manages resources across agent processes
- Provides privileged operations agents cannot perform themselves
- Creates a standardized interface between agents and their environment

## Implementation Details

### Program Definition via TOML

- Enables agent self-modification (similar to Claude Code updating CLAUDE.md)
- Makes programs portable across implementations
- Human-readable and LLM-friendly

### System Calls

> **Note:** System calls are planned but not yet implemented in the current version.

Like Unix kernel system calls, LLMProc will implement:
- **Spawn**: Create new agent processes (analogous to exec())
- **Fork**: Duplicate an existing agent with its state (analogous to fork())

Reference: [forking-an-agent](https://github.com/cccntu/forking-an-agent)

### MCP Integration

- System calls are implemented in the LLMProc kernel
- "Userspace" tools are provided via MCP
- MCP provides a standard protocol for tools that's independent of the LLMProc implementation

## Roadmap

1. [ ] Implement Spawn System Call
2. [ ] Create a Claude Code Program 
3. [ ] Implement Fork System Call
4. [ ] Improve OpenAI integration (MCP support)