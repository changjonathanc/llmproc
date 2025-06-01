# LLMProc Documentation

Welcome to the LLMProc documentation. This guide will help you navigate the key concepts and features of LLMProc, a Unix-inspired runtime for language models. LLMProc treats LLMs as computational processes with their own lifecycle, I/O channels, and system calls.

For design rationales and API decisions, see the [API Design FAQ](../FAQ.md).

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
   - [Environment Variables](environment-variables.md) - Configuration via environment variables
   - [Program Initialization](program-initialization.md) - Configure user prompts and demo mode

### Core Features

These features form the foundation of LLMProc's Unix-inspired approach:

1. **Large Content Handling**
   - [File Descriptor System](file-descriptor-system.md) - Unix-like pagination for large outputs
   - [Token-Efficient Tool Use](token-efficient-tool-use.md) - Optimize token usage for tools

2. **Process Management**
   - [Program Linking](program-linking.md) - Delegate tasks to specialized processes
   - [Fork Feature](fork-feature.md) - Create process copies with shared state
   - [GOTO Feature](goto-feature.md) - Reset conversations to previous points
   - [Persistent Event Loop](persistent-event-loop.md) - Dedicated loop for synchronous LLMProcess access
   - [Tool Access Control](tool-access-control.md) - Secure multi-process environments with permissions

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

- [Program Initialization](program-initialization.md) - How programs are initialized and validated
- [Callbacks System](callbacks.md) - Monitor execution events
- [Tool Registration Callback](tool-registration-callback.md) - Customize tool initialization during registration
- [Error Handling Strategy](error-handling-strategy.md) - How errors are managed
- [Testing](testing.md) - Testing approach and API testing

- [Program Compiler](program-compiler.md) - Compile and cache programs for reuse
- [Runtime Context Management](runtime-context.md) - Dependency injection for tools
- [System Prompt Examination Tool](system-prompt-tool.md) - Inspect enriched prompts
- [Tool Error Handling Guidelines](tool-error-handling.md) - Error handling patterns
- [Test Plan](test-plan.md) - Outline of planned tests
- [YAML Configuration Schema](yaml_config_schema.md) - Auto-generated configuration reference

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
4. [Fork Feature](fork-feature.md), [GOTO Feature](goto-feature.md), and [Tool Access Control](tool-access-control.md)

## API Reference

- [API Parameters](api_parameters.md) - Configuration parameters
- [API Testing](api_testing.md) - Testing with real APIs

For details on the full API, see:
- [Core API Architecture](api/core.md)
- [Class Reference](api/classes.md)
- [API Patterns and Best Practices](api/patterns.md)

## Release Notes

Detailed release notes for each version are available in the release_notes directory:

- [Release Notes 0.9.0](release_notes/RELEASE_NOTES_0.9.0.md)
- [Release Notes 0.8.0](release_notes/RELEASE_NOTES_0.8.0.md)

## External References

Additional provider guides are located in the [external-references](external-references/) directory.

- [Anthropic API Guide](external-references/anthropic-api.md)
