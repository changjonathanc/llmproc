# Google Gemini Integration

This document describes how to use Google's Gemini models with LLMProc.

## Overview

LLMProc supports Google's Gemini models through both the direct Google AI Studio API and Vertex AI. The integration uses the official `google-genai` Python SDK to interact with the models.

## Available Models

- **Gemini 2.0 Flash**: Smaller, faster model for efficient processing
- **Gemini 2.5 Pro**: Larger, smarter model with advanced capabilities

> **Note**: Be careful with model names - only use officially released models. For LLMProc, we exclusively use gemini-2.0-flash and gemini-2.5-pro to standardize our implementation.

## Implementation Status

The current implementation provides:
- ✅ Basic text generation
- ✅ Conversation handling with proper state management
- ✅ Context window awareness
- ✅ Both direct API and Vertex AI support
- ✅ Token counting with proper fallback mechanism

Upcoming features:
- ❌ Tool calling (coming in next release)
- ❌ Multimodal inputs
- ❌ Streaming responses

## Setup

### Installation

First, install the `google-genai` SDK:

```bash
pip install google-genai
```

### Authentication

For security reasons, LLMProc requires all API keys to be configured via environment variables only. API keys cannot be passed directly in code.

#### Direct API (Google AI Studio)

1. Get an API key from [Google AI Studio](https://ai.google.dev/)
2. Set the API key in your environment:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   # Or alternatively:
   export GOOGLE_API_KEY=your_api_key_here
   ```

#### Vertex AI

1. Set up Google Cloud with the Vertex AI API enabled
2. Configure application default credentials:
   ```bash
   gcloud auth application-default login
   ```
3. Set environment variables:
   ```bash
   export GOOGLE_CLOUD_PROJECT=your_project_id
   export CLOUD_ML_REGION=us-central1  # or your preferred region
   ```

> **Security Note**: API keys and credentials should always be handled securely. Never hardcode API keys in your application code or check them into version control. LLMProc enforces this by requiring API keys to be provided through environment variables only.

## Configuration

### Direct API Configuration

```toml
[model]
name = "gemini-2.0-flash"  # Smaller, faster model
provider = "gemini"
display_name = "Gemini 2.0 Flash"

[parameters]
temperature = 0.7
max_tokens = 4096

[prompt]
system_prompt = "You are a helpful AI assistant."
```

### Vertex AI Configuration

```toml
[model]
name = "gemini-2.0-flash"  # Recommended for testing
provider = "gemini_vertex"
display_name = "Gemini 2.0 Flash (Vertex AI)"
# Optional: Override environment variables
# project_id = "your-project-id"
# region = "us-central1"

[parameters]
temperature = 0.7
max_tokens = 4096

[prompt]
system_prompt = "You are a helpful AI assistant."
```

## Parameter Mapping

LLMProc maps standard parameters to Gemini-specific parameters:

| LLMProc Parameter | Gemini Parameter |
|-------------------|------------------|
| temperature       | temperature      |
| max_tokens        | max_output_tokens|
| top_p             | top_p            |
| top_k             | top_k            |
| stop              | stop_sequences   |

## Model Selection Notes

For this implementation, we standardize on two models:

1. **Gemini 2.0 Flash** - Use for tasks requiring lower latency or when cost efficiency is important
2. **Gemini 2.5 Pro** - Use for tasks requiring advanced reasoning, complex understanding, or longer context

Both models support all features of our implementation, including token counting functionality.

> **Important**: For consistency across the codebase, we exclusively use these two models in LLMProc. This standardization simplifies testing, documentation, and user experience.

## Example Usage

### Python API

```python
from llmproc.program import LLMProgram

# Create a program with Gemini model
program = LLMProgram(
    model_name="gemini-2.0-flash",  # Recommended for testing
    provider="gemini",
    system_prompt="You are a helpful assistant",
    parameters={"temperature": 0.7, "max_tokens": 4096},
)

# Start the process
process = await program.start()

# Run with user input
result = await process.run("Hello, how are you?")

# Get the response
response = process.get_last_message()
print(response)

# Token counting
token_info = await process.count_tokens()
if "error" in token_info:
    print(f"Token counting error: {token_info['error']}")
elif "note" in token_info:
    print(f"Token counting note: {token_info['note']}")
    print(f"Estimated context usage: {token_info['percentage']:.1f}%")
else:
    print(f"Token count: {token_info['input_tokens']} tokens")
    print(f"Context window: {token_info['context_window']} tokens")
    print(f"Usage: {token_info['percentage']:.1f}% of context window")
    print(f"Remaining: {token_info['remaining_tokens']} tokens")
    if "cached_tokens" in token_info and token_info["cached_tokens"] > 0:
        print(f"Cached tokens: {token_info['cached_tokens']} tokens")
```

### Command Line

Using a TOML configuration file:

```bash
# Use the Gemini configuration
llmproc-demo ./examples/gemini.toml
```

## Troubleshooting

### API Errors

- **Authentication Errors**: Check that your API key is set correctly for direct API or that your Google Cloud authentication is configured correctly for Vertex AI.
- **Rate Limits**: If you encounter rate limit errors, either wait or request a quota increase.
- **Model Access**: Ensure you have access to the models in your region. Use our standardized models: `gemini-2.0-flash` or `gemini-2.5-pro`.

### SDK Installation Issues

If you have problems with the `google-genai` SDK:

1. Ensure you're using the latest version: `pip install --upgrade google-genai`
2. For Vertex AI, you might need additional dependencies: `pip install google-cloud-aiplatform`

### Version Compatibility

The implementation has been tested with `google-genai` SDK version 1.9.0. If you encounter issues, check your SDK version with `pip show google-genai`.

## Token Counting

Gemini integration supports token counting through the official Google SDK. The implementation:

1. Converts conversation history to the Gemini API format
2. Includes system instructions as part of the request configuration
3. Calculates context window usage based on token count and model's window size
4. Tracks cached tokens when available

### Token Counting Response Format

The `count_tokens()` method returns a dictionary with the following keys:

| Key               | Description                                               |
|-------------------|-----------------------------------------------------------|
| input_tokens      | Total number of tokens in the conversation                |
| context_window    | Maximum context window size for the model                 |
| percentage        | Percentage of the context window used                     |
| remaining_tokens  | Number of tokens remaining in the context window          |
| cached_tokens     | Number of tokens in cached content (if applicable)        |
| note              | Informational note (present when using estimation)        |
| error             | Error message if token counting failed                    |

### Fallback Mechanism

If token counting isn't available (e.g., API issues or client limitations), the implementation falls back to estimation:

- Returns `-1` for `input_tokens` to indicate estimation mode
- Provides reasonable estimates for other metrics based on model specifications
- Includes a note explaining the estimation is being used

This ensures robustness across different environments and configurations.
