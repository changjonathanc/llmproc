# LLMProc - Miscellaneous Notes & Details

This document contains additional information supplementing the main README.md. For detailed documentation, see the `/docs` directory.

## Terminology & Core Concepts

**Note on "Agent" terminology**: We deliberately avoid using the term "agent" in LLMProc as it has varied definitions in the AI community. Some view agents as the underlying technology where LLMs have agency to decide what tools to call, while others view agents as user-facing programs that users interact with. To maintain clarity, we focus on the Unix-like "process" metaphor throughout LLMProc.

The LLMProc project follows a Unix-inspired program/process model:

- **Program**: Immutable configuration that describes what to run (similar to a Unix executable)
- **Process**: Runtime instance created from a Program (similar to a running process in Unix)
- **Runtime Context**: System for dependency injection to tools during execution

For more details on this architecture, see [unix-program-process-model.md](docs/unix-program-process-model.md).

## Installation Details

### Full Installation Options

```bash
# Install with uv (recommended)
# Basic installation
uv pip install llmproc

# Install with development dependencies
uv pip install "llmproc[dev]"

# Install with specific provider support
uv pip install "llmproc[openai]"     # For OpenAI models
uv pip install "llmproc[anthropic]"  # For Anthropic/Claude models
uv pip install "llmproc[vertex]"     # For Google Vertex AI models
uv pip install "llmproc[gemini]"     # For Google Gemini models

# Install with all provider support
uv pip install "llmproc[all]"

# Development installation
uv pip install -e ".[dev,all]"

# Or with pip
pip install llmproc               # Base package
pip install "llmproc[openai]"     # For OpenAI models
pip install "llmproc[anthropic]"  # For Anthropic/Claude models
pip install "llmproc[vertex]"     # For Google Vertex AI models
pip install "llmproc[gemini]"     # For Google Gemini models
pip install "llmproc[all]"        # All providers
```

### Environment Variables

LLMProc requires provider-specific API keys set as environment variables:

```bash
# Set API keys as environment variables
export OPENAI_API_KEY="your-key"            # For OpenAI models
export ANTHROPIC_API_KEY="your-key"         # For Claude models
export GOOGLE_API_KEY="your-key"            # For Gemini models
export ANTHROPIC_VERTEX_PROJECT_ID="id"     # For Claude on Vertex AI
export CLOUD_ML_REGION="us-central1"        # For Vertex AI (defaults to us-central1)
```

You can set these in your environment or include them in a `.env` file at the root of your project.

## Key Features Reference

### File Descriptor System
Handles large inputs/outputs by creating file-like references with paging support.

See [file-descriptor-system.md](docs/file-descriptor-system.md)

### Program Linking
Connects multiple LLMProcess instances for collaborative problem-solving.

See [program-linking.md](docs/program-linking.md)

### MCP Tool Support
Connect to external tool servers via Model Context Protocol.

See [mcp-feature.md](docs/mcp-feature.md)

### Tool Aliases
Provides shorter, more intuitive names for tools.

See [tool-aliases.md](docs/tool-aliases.md)

### Token Efficient Tool Use
Optimizes token usage for tool calls with Claude 3.7+.

See [token-efficient-tool-use.md](docs/token-efficient-tool-use.md)

## Design Philosophy

### Unix-Inspired Architecture
LLMProc treats LLMs as computing processes:
- Each model is a process defined by a program (TOML file)
- It maintains state between executions
- It interacts with the system through defined interfaces

The library functions as a kernel:
- Implements system calls for LLM processes
- Manages resources across processes
- Creates a standardized interface with the environment

### Performance Considerations

- **Resource Usage**: Each Process instance requires memory for its state
- **API Costs**: Using multiple processes results in multiple API calls
- **Linked Programs**: Program linking creates additional processes with separate API calls
- **Selective MCP Usage**: MCP tools now use selective initialization for better performance

## Common Patterns

```python
# Create a program (configuration)
program = LLMProgram(model_name="claude-3-7-sonnet", provider="anthropic")

# Create a process from the program (runtime instance)
process = await program.start()

# Use the process
await process.run("Hello, Claude!")
```

**Note on `compile()` method**: The public `compile()` method is intended to be used primarily when implementing program serialization/export functionality. For typical usage, the `start()` method handles necessary validation internally. Consider direct use of `program.start()` in most cases.

For more API patterns, see [api/patterns.md](docs/api/patterns.md).