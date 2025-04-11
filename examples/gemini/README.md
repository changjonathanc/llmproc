# Google Gemini Models for LLMProc

This directory contains example configurations for using Google's Gemini models with LLMProc.

## Models Available

For consistency across the codebase, we exclusively use these two standardized models:

- **Gemini 2.0 Flash**: Smaller, faster model for efficient processing
- **Gemini 2.5 Pro**: Larger, smarter model with advanced capabilities and longer context

## Setup Instructions

### Direct API Access (Google AI Studio)

1. Get an API key from [Google AI Studio](https://ai.google.dev/)
2. Set the API key in your environment:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   # Or alternatively:
   export GOOGLE_API_KEY=your_api_key_here
   ```
3. Run one of the direct API examples:
   ```bash
   llmproc-demo ./examples/gemini/gemini-2.5-pro-direct.toml
   ```

### Vertex AI Access

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
4. Run one of the Vertex AI examples:
   ```bash
   llmproc-demo ./examples/gemini/gemini-2.0-flash-vertex.toml  # Smaller, faster model
   llmproc-demo ./examples/gemini/gemini-2.5-pro-vertex.toml    # Larger, smarter model
   ```

## Development Status

The current Gemini integration supports basic text generation without advanced features like tool usage. Advanced features will be added in future releases.