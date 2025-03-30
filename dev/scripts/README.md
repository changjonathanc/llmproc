# Development Scripts

This directory contains scripts used for development, testing, and experimentation that aren't part of the main package.

## Vertex AI Test Scripts

These scripts demonstrate and test specific features with Anthropic Claude on Vertex AI:

- `test_vertex_claude_37_token_efficient_tools.py`: Tests the token-efficient tools beta feature with Claude 3.7 on Vertex AI
- `test_vertex_claude_37_token_efficient_tools_complex.py`: Tests token-efficient tools with more complex tool definitions
- `test_vertex_claude_37_with_cache_control.py`: Tests prompt caching with `cache_control` parameters on Vertex AI

### Usage

These scripts require:

1. Google Cloud project with Vertex AI enabled
2. Claude 3.7 access on Vertex AI
3. Environment variables:
   - `ANTHROPIC_VERTEX_PROJECT_ID`: Your GCP project ID
   - `CLOUD_ML_REGION`: The Vertex AI region (e.g., "us-central1")

Example:
```bash
export ANTHROPIC_VERTEX_PROJECT_ID="your-project-id"
export CLOUD_ML_REGION="us-central1"
python dev/scripts/test_vertex_claude_37_token_efficient_tools.py
```