# LLMProc Architecture & Design Philosophy

This document outlines the core architectural decisions and design philosophy behind the LLMProc library. Unlike the README which focuses on usage and getting started, this document explains *why* specific design choices were made and the long-term vision for the project.

## Core Concepts

### LLM Agent as Process

We view LLM Agents as stateful processes, similar to operating system processes:

- **Definition via TOML**: Programs (agents) are defined declaratively using TOML configuration files
- **Input/Output Model**: You send input to the process, it may execute commands/tools, and returns a string response
- **Stateful by Default**: The process maintains conversation state across multiple interactions
- **Consistent API**: The interface abstracts away provider-specific details behind a unified API

This process-oriented approach enables predictable behavior, standardized configuration, and reusable patterns across different LLM providers and use cases.

### System Calls (Planned)

We're planning to implement "system calls" to enable more powerful agent capabilities:

- **Spawn**: Create new agent processes with specific configurations
- **Fork**: Duplicate an existing agent process (including its state) for parallel exploration
- **Inter-Process Communication**: Allow agents to communicate with each other
- **Resource Management**: Control compute/token usage and implement timeouts

These capabilities will allow for more complex agent architectures similar to how operating system processes enable complex software systems.

### MCP Integration & Portability

The Model Context Protocol (MCP) integration follows our core philosophy of portability:

- **Portable Program Definitions**: TOML configuration files define portable programs
- **Portable Tool Definitions**: MCP tools are defined in a standard way and work across providers
- **Reference Implementation**: The Python library is just one possible implementation
- **Language Agnostic**: Programs and tools can be implemented in any language

This approach allows LLMProc programs to be executed by runners implemented in various languages while maintaining consistent behavior.

## Implementation Principles

- **Abstraction with Reasonable Defaults**: Hide complexity while providing sensible defaults
- **Provider Agnostic**: Core functionality works consistently across different LLM providers
- **Unified API**: Same interface for both synchronous and asynchronous code
- **Progressive Enhancement**: Basic functionality works out of the box, advanced features available when needed
- **Explicit Configuration**: Configuration is explicit, clear, and documented

## Future Directions

- **Agent Orchestration**: Coordination of multiple agent processes
- **Agent Specialization**: Creating purpose-specific agent templates
- **Performance Optimization**: Improved caching and state management
- **Cross-Provider Tool Compatibility**: Ensuring tools work across different LLM providers