# Anthropic Integration in LLMProc

This document provides information about using Anthropic models with LLMProc, including both direct API access and Google Vertex AI integration.

## Direct Anthropic API

LLMProc supports direct integration with Anthropic's Claude models through the Anthropic API.

### Basic Configuration

```toml
[model]
name = "claude-3-5-haiku-20241022"
provider = "anthropic"
display_name = "Claude Haiku"

[prompt]
system_prompt = "You are Claude, a helpful AI assistant."

[parameters]
temperature = 0.7
max_tokens = 1000

# For Claude 3.7+ models, you can configure thinking capabilities
[parameters.thinking]
type = "enabled"
budget_tokens = 4000
```

### Authentication

The direct Anthropic API integration requires an Anthropic API key:

- Set the `ANTHROPIC_API_KEY` environment variable with your API key
- You can get an API key from the [Anthropic Console](https://console.anthropic.com/)

## Anthropic on Vertex AI Integration

LLMProc also supports using Anthropic models through Google Cloud's Vertex AI platform, which can provide better infrastructure, compliance features, and potentially different pricing.

### Basic Configuration

```toml
[model]
name = "claude-3-5-haiku-20241022" # Use appropriate Vertex model name
provider = "anthropic_vertex"
display_name = "Claude Haiku (Vertex AI)"

[prompt]
system_prompt = "You are Claude on Vertex AI, a helpful AI assistant."

[parameters]
temperature = 0.7
max_tokens = 1000

# For Claude 3.7+ models, you can configure thinking capabilities
[parameters.thinking]
type = "enabled"
budget_tokens = 4000
```

### Authentication and Setup

To use Anthropic models through Vertex AI:

1. **Google Cloud Project**:
   - Create or use an existing Google Cloud Project with Vertex AI API enabled
   - Ensure you have permissions to use the Vertex AI API and Claude models

2. **Environment Variables and Configuration**:
   - `ANTHROPIC_VERTEX_PROJECT_ID`: Set to your Google Cloud Project ID
   - `CLOUD_ML_REGION`: Set to your preferred Google Cloud region (defaults to us-central1)
   - In your TOML configuration, specify project and region:
     ```toml
     [model]
     project_id = "your-project-id"  # Optional, can also use environment variable
     region = "your-preferred-region"  # Refer to Google Cloud docs for available regions
     ```

3. **Google Cloud Authentication**:
   - Authenticate with Google Cloud using one of the following methods:
     - Run `gcloud auth application-default login` on your machine
     - Use service account credentials
     - Use workload identity when running on Google Cloud

### Vertex AI Models

Vertex AI offers specific versions of Claude models. Check the [Google Cloud Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/models/claude) for the most up-to-date list of available models and their naming conventions.

### Parameter Differences

- Most parameters work the same way across both providers
- Some Anthropic-specific parameters may have different behavior on Vertex AI
- Refer to the Google Cloud documentation for any Vertex-specific limitations

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
