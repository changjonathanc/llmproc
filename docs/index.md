# LLMProc Documentation

Welcome to the LLMProc documentation. This guide will help you navigate the key concepts and features of LLMProc, a Unix-inspired framework for building powerful LLM applications.

## Documentation Roadmap

### For New Users

Start here if you're new to LLMProc:

1. **Core Architecture**
   - [Unix-Inspired Program/Process Model](unix-program-process-model.md) - Fundamental design pattern
   - [Python SDK](python-sdk.md) - Creating programs with the fluent API

2. **Getting Started**
   - [Function-Based Tools](function-based-tools.md) - Register Python functions as tools
   - [Preload Feature](preload-feature.md) - Include files in system prompts
   - [Environment Info](env_info.md) - Add runtime context

### Core Features

These features form the foundation of LLMProc's Unix-inspired approach:

1. **Large Content Handling**
   - [File Descriptor System](file-descriptor-system.md) - Unix-like pagination for large outputs
   - [Token-Efficient Tool Use](token-efficient-tool-use.md) - Optimize token usage for tools

2. **Process Management**
   - [Program Linking](program-linking.md) - Delegate tasks to specialized processes
   - [Fork Feature](fork-feature.md) - Create process copies with shared state
   - [GOTO Feature](goto-feature.md) - Reset conversations to previous points

3. **Tool System**
   - [Tool Aliases](tool-aliases.md) - Provide simpler names for tools
   - [Adding Built-in Tools](adding-builtin-tools.md) - Extend with custom tools
   - [MCP Feature](mcp-feature.md) - Model Context Protocol integration

### Provider-Specific Documentation

Documentation for specific model providers:

- [Anthropic Models](anthropic.md) - Claude models usage
- [Claude Thinking Models](claude-thinking-models.md) - Using Claude's thinking capabilities
- [OpenAI Reasoning Models](openai-reasoning-models.md) - Using OpenAI's reasoning capabilities
- [Gemini Models](gemini.md) - Google Gemini models usage

### Advanced Topics

For users looking to extend and optimize LLMProc:

- [Program Compilation](program-compilation.md) - How programs are compiled
- [RunResult Callbacks](runresult-callbacks.md) - Monitor execution with callbacks
- [Error Handling Strategy](error-handling-strategy.md) - How errors are managed
- [Testing](testing.md) - Testing approach and API testing

## Learning Paths

### For Application Developers

If you're building applications with LLM capabilities:

1. [Python SDK](python-sdk.md)
2. [Function-Based Tools](function-based-tools.md)
3. [File Descriptor System](file-descriptor-system.md)
4. [Program Linking](program-linking.md)

### For Tool Developers

If you're extending LLMProc with custom tools:

1. [Unix-Inspired Program/Process Model](unix-program-process-model.md)
2. [Adding Built-in Tools](adding-builtin-tools.md)
3. [Function-Based Tools](function-based-tools.md)
4. [MCP Feature](mcp-feature.md)

### For Advanced Users

If you're implementing complex agent architectures:

1. [Program Linking](program-linking.md)
2. [Program Linking Advantages](program-linking-advantages.md)
3. [Token-Efficient Tool Use](token-efficient-tool-use.md)
4. [Fork Feature](fork-feature.md) and [GOTO Feature](goto-feature.md)

## API Reference

- [API Patterns](api_patterns.md) - Recommended usage patterns
- [API Parameters](api_parameters.md) - Configuration parameters
- [API Testing](api_testing.md) - Testing with real APIs

For details on the full API, see:
- [Core API Architecture](api/core.md)
- [Class Reference](api/classes.md)
- [API Patterns and Best Practices](api/patterns.md)