# OpenAI Integration in LLMProc

LLMProc supports OpenAI models through multiple APIs, with automatic selection based on model type:

- **Chat Completions API**: For GPT-4, GPT-4o, GPT-3.5 models
- **Responses API**: For reasoning models (o1, o3, o4 series) - *coming soon*

## Provider Options

LLMProc offers three ways to specify OpenAI providers:

### Auto-Selection (Recommended)
```yaml
model:
  name: "gpt-4o-mini"  # Auto-selects Chat Completions API
  provider: "openai"

model:
  name: "o3-mini"      # Auto-selects Responses API (when implemented)
  provider: "openai"
```

### Explicit Provider Selection
```yaml
# Force Chat Completions API
model:
  name: "gpt-4o-mini"
  provider: "openai_chat"

# Force Responses API (when implemented)
model:
  name: "o3-mini"
  provider: "openai_response"
```

## Auto-Selection Logic

When using `provider = "openai"`, LLMProc automatically selects the appropriate API:

- **Models starting with 'o'** → `openai_response` (Responses API)
  - Examples: `o1-preview`, `o1-mini`, `o3-mini`, `o3`, `o4-mini`
- **All other models** → `openai_chat` (Chat Completions API)  
  - Examples: `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`

This ensures optimal API usage while maintaining simple configuration.

## Authentication

The OpenAI integration requires an OpenAI API key:

- Set the `OPENAI_API_KEY` environment variable with your key
- Obtain the key from the [OpenAI Platform](https://platform.openai.com/api-keys)

## Supported Models

LLMProc supports all OpenAI chat models, including:

- **GPT-4o models**: `gpt-4o`, `gpt-4o-mini`
- **GPT-4 models**: `gpt-4`, `gpt-4-turbo`
- **GPT-3.5**: `gpt-3.5-turbo`
- **Reasoning models**: `o1-preview`, `o1-mini`, `o3-mini` (see [OpenAI Reasoning Models](openai-reasoning-models.md))

## Tool Support

OpenAI models support the full range of tools available in LLMProc, including:

- Built-in system tools (calculator, spawn, fork, etc.)
- MCP (Model Context Protocol) tools
- Custom function-based tools
- Server-hosted tools for reasoning models like `web_search` (o3 and GPT-4o)

### Tool Error Handling

**Important**: OpenAI's tool calling API does not support the `is_error` field that Anthropic models use. To ensure clear error communication to the model, LLMProc automatically formats tool errors with an "ERROR:" prefix:

```python
# When a tool returns an error
result = ToolResult.from_error("File not found")

# OpenAI receives:
"ERROR: File not found"

# Anthropic receives:
{"content": "File not found", "is_error": true}
```

This formatting ensures that OpenAI models can properly recognize and handle tool failures.

## Configuration Example with Tools

```yaml
model:
  name: "gpt-4o-mini"
  provider: "openai"

prompt:
  system: "You are a helpful assistant with access to tools. Use them when appropriate."

parameters:
  max_tokens: 1000
  temperature: 0.1

tools:
  builtin:
    - calculator
    - spawn
  mcp:
    servers:
      - filesystem
```

## Limitations

- **Tool Error Format**: Unlike Anthropic models, OpenAI doesn't support the `is_error` field, so errors are prefixed with "ERROR:" in the content
- **No Explicit Caching**: OpenAI handles caching automatically; no manual cache control like Anthropic's `cache_control` parameter
- **Provider-Specific Features**: Some LLMProc features like token-efficient tool use are Anthropic-specific

## Dependencies

To use OpenAI models, install the optional OpenAI dependencies:

```bash
# Install with OpenAI support
uv sync --extra openai

# Or add to existing installation
uv add tiktoken
```

The `tiktoken` library is required for accurate token counting with OpenAI models.

## Reasoning Models

For information about using OpenAI's reasoning models (o1, o3 series), see the dedicated [OpenAI Reasoning Models](openai-reasoning-models.md) documentation.

---
[← Back to Documentation Index](index.md)
