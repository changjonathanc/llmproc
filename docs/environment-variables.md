# Environment Variables

This page documents all environment variables used by llmproc.

## API Keys

### Provider Authentication

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | Anthropic provider |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google API key for Gemini models | Gemini provider (either one works) |

### Google Cloud Configuration

| Variable | Description | Default | Used By |
|----------|-------------|---------|---------|
| `ANTHROPIC_VERTEX_PROJECT_ID` | Google Cloud project ID | None | Anthropic Vertex provider |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud project ID | None | Gemini Vertex provider |
| `CLOUD_ML_REGION` | Google Cloud region | `us-central1` | Both Vertex providers |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON | None | Google Cloud authentication |

## Retry Configuration

### API Retry Settings

| Variable | Description | Default | Type |
|----------|-------------|---------|------|
| `LLMPROC_RETRY_MAX_ATTEMPTS` | Maximum number of retry attempts | `6` | Integer |
| `LLMPROC_RETRY_INITIAL_WAIT` | Initial wait time in seconds | `1` | Integer |
| `LLMPROC_RETRY_MAX_WAIT` | Maximum wait time in seconds | `90` | Integer |

These variables control the exponential backoff retry mechanism for API calls.

## MCP Configuration

### External Tool Servers

MCP server configurations can reference environment variables using the `${VAR_NAME}` syntax:

```yaml
mcp:
  servers:
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
```

Common MCP environment variables:
- `GITHUB_TOKEN` or `GITHUB_PERSONAL_ACCESS_TOKEN` - GitHub API access
- `LLMPROC_MCP_TRANSIENT` - Set to `true` to disable persistent MCP connections
- `LLMPROC_TOOL_FETCH_TIMEOUT` - Maximum time in seconds to wait for MCP tool fetching (default: 30.0)
- `LLMPROC_TOOL_CALL_TIMEOUT` - Maximum time in seconds to wait for MCP tool calls (default: 30.0)
- `LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT` - Controls whether the process fails when MCP tool initialization timeouts occur (default: true, set to "false" to continue without tools)
- Any custom variables required by your MCP servers

## Testing and Development

### Test Configuration

| Variable | Description | Used For |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Running API tests |
| `ANTHROPIC_API_KEY` | Anthropic API key | Running API tests |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Vertex project | Running Vertex AI tests |

## Usage Examples

### Basic Setup

```bash
# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Gemini
export GEMINI_API_KEY="..."
```

### Vertex AI Setup

```bash
# For Anthropic on Vertex AI
export ANTHROPIC_VERTEX_PROJECT_ID="my-project"
export CLOUD_ML_REGION="us-east1"  # Optional, defaults to us-central1

# For Gemini on Vertex AI
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Retry Configuration

```bash
# Increase retry attempts for unstable networks
export LLMPROC_RETRY_MAX_ATTEMPTS=10
export LLMPROC_RETRY_INITIAL_WAIT=2
export LLMPROC_RETRY_MAX_WAIT=120
```

### MCP Tool Timeouts and Error Handling

```bash
# Increase timeout for MCP tool fetching (for very slow networks or many tools)
export LLMPROC_TOOL_FETCH_TIMEOUT=60

# Increase timeout for MCP tool calls (for long-running operations)
export LLMPROC_TOOL_CALL_TIMEOUT=60

# Allow the process to continue even if MCP tool initialization fails
export LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT=false
```

## Security Notes

- Never commit API keys to version control
- Use environment variables or secret management systems
- Consider using `.env` files for local development (not included in git)
- For production, use proper secret management (e.g., AWS Secrets Manager, Google Secret Manager)

## See Also

- [Anthropic Documentation](anthropic.md) - Anthropic-specific configuration
- [Gemini Documentation](gemini.md) - Gemini-specific configuration
- [MCP Feature](mcp-feature.md) - MCP server configuration
- [API Testing](api_testing.md) - Testing with API keys
