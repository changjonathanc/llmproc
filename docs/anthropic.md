# Anthropic Integration in LLMProc

LLMProc works with Anthropic's Claude models either through the direct Anthropic API or via Google Cloud Vertex AI. This guide explains how to configure both options.

## Direct Anthropic API

LLMProc supports direct integration with Anthropic's Claude models through the Anthropic API.

### Basic Configuration

```yaml
model:
  name: "claude-3-5-haiku-20241022"
  provider: "anthropic"
  display_name: "Claude Haiku"

prompt:
  system_prompt: "You are Claude, a helpful AI assistant."

parameters:
  temperature: 0.7
  max_tokens: 1000
  thinking:
    type: "enabled"
    budget_tokens: 4000
```

### Authentication

The direct integration requires an Anthropic API key:

- Set the `ANTHROPIC_API_KEY` environment variable with your key
- Obtain the key from the [Anthropic Console](https://console.anthropic.com/)

## Anthropic on Vertex AI Integration

You can also run Claude models via Google Cloud Vertex AI, which may offer different infrastructure, compliance options, or pricing.

### Basic Configuration

```yaml
model:
  name: "claude-3-5-haiku-20241022"  # Use appropriate Vertex model name
  provider: "anthropic_vertex"
  display_name: "Claude Haiku (Vertex AI)"

prompt:
  system_prompt: "You are Claude on Vertex AI, a helpful AI assistant."

parameters:
  temperature: 0.7
  max_tokens: 1000
  thinking:
    type: "enabled"
    budget_tokens: 4000
```

### Authentication and Setup

To use Anthropic models through Vertex AI:

1. **Google Cloud Project**:
   - Create or use an existing Google Cloud Project with Vertex AI API enabled
   - Ensure you have permissions to use the Vertex AI API and Claude models

2. **Environment Variables and Configuration**:
   - `ANTHROPIC_VERTEX_PROJECT_ID`: Your Google Cloud project ID
   - `CLOUD_ML_REGION`: Preferred region (defaults to `us-central1`)
   - In the `[model]` section you can also specify:
      ```yaml
      model:
        project_id: "your-project-id"
        region: "your-region"
      ```

3. **Google Cloud Authentication**:
   - Authenticate with Google Cloud using one of these methods:
     - `gcloud auth application-default login`
     - Service account credentials
     - Workload identity when running on Google Cloud

### Vertex AI Models

Vertex AI offers specific versions of Claude models. Check the [Google Cloud Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/models/claude) for the most up-to-date list of available models and their naming conventions.

### Parameter Differences

- Most parameters behave the same across providers
- Anthropic-specific settings may differ on Vertex AI
- See Google Cloud documentation for Vertex-specific limits

## Anthropic-Specific Features

Anthropic models expose features not found with other providers. LLMProc
supports two notable Claude-only capabilities:

- **Explicit Prompt Caching** – Use the `cache_control` parameter to cache
  system prompts and tool calls when interacting with Claude models. This
  reduces token usage on repeated requests. Other providers do not currently
  support explicit caching.
- **Token-Efficient Tool Use** – Claude 3.7 models offer a beta feature that
  lowers token consumption when using tools. Enable it via the
  `anthropic-beta` header or the `enable_token_efficient_tools()` method in the
  SDK. This optimization is specific to Anthropic models.

## Tool Support

Both Anthropic API and Anthropic on Vertex AI support the full range of tools available in LLMProc, including:

- System tools like fork and spawn
- MCP (Model Context Protocol) tools when properly configured

## Troubleshooting

### Common Issues with Vertex AI

- **Authentication Errors**: Ensure your Google Cloud credentials are properly set up
- **Project ID Issues**: Verify your `ANTHROPIC_VERTEX_PROJECT_ID` is correct
- **Region Availability**: Refer to the [Google Cloud documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/models/claude#model-versions) for the regions where Claude models are available
- **API Enablement**: Ensure Vertex AI API is enabled in your Google Cloud project
- **Permissions**: Confirm you have the required IAM permissions to use Vertex AI
- **Provider API Errors**: If you see errors like "Error calling tool: Error from provider API: PERMISSION_DENIED", check that:
  - You're using the correct model name format (`claude-3-5-haiku@20241022`, with the `@` symbol)
  - Your project has been approved to use Claude on Vertex AI
  - You have the proper service agent roles for the Vertex AI service accounts

## Further Reading

- [Anthropic API Guide](external-references/anthropic-api.md)

---
[← Back to Documentation Index](index.md)
