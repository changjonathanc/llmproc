# LLMProc - Miscellaneous Notes & Details

This document contains additional information supplementing the main README.md. For detailed documentation, see the `/docs` directory. For design rationales and API decisions, see [FAQ.md](FAQ.md).

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
uv sync --all-extras --all-groups

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

## Performance Considerations

- **Resource Usage**: Each Process instance requires memory for its state
- **API Costs**: Using multiple processes results in multiple API calls
- **Linked Programs**: Program linking creates additional processes with separate API calls
- **Selective MCP Usage**: MCP tools now use selective initialization for better performance

**Note on `compile()` method**: The public `compile()` method is intended to be used primarily when implementing program serialization/export functionality. For typical usage, the `start()` method handles necessary validation internally. Consider direct use of `program.start()` in most cases.

For more API patterns, see [api/patterns.md](docs/api/patterns.md).
