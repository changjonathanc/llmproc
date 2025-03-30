"""Simple provider module for LLMProc to return appropriate API clients."""

import os
from typing import Any

# Import provider constants
from llmproc.providers.constants import (
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX,
    SUPPORTED_PROVIDERS
)

# Try importing providers, set to None if packages aren't installed
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None

try:
    from anthropic import AsyncAnthropicVertex
except ImportError:
    AsyncAnthropicVertex = None


def get_provider_client(
    provider: str,
    model_name: str,
    project_id: str | None = None,
    region: str | None = None,
) -> Any:
    """Get the appropriate provider client.

    Args:
        provider: The provider to use (openai, anthropic, or anthropic_vertex)
        model_name: The model name to use (used for logging)
        project_id: Google Cloud project ID for Vertex AI
        region: Google Cloud region for Vertex AI

    Returns:
        The initialized client for the specified provider

    Raises:
        NotImplementedError: If the provider is not supported
        ImportError: If the required package for a provider is not installed
    """
    # Normalize provider name
    provider = provider.lower()
    
    if provider == PROVIDER_OPENAI:
        if AsyncOpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAI provider. Install it with 'pip install openai'."
            )
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    elif provider == PROVIDER_ANTHROPIC:
        if AsyncAnthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic provider. Install it with 'pip install anthropic'."
            )
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    elif provider == PROVIDER_ANTHROPIC_VERTEX:
        if AsyncAnthropicVertex is None:
            raise ImportError(
                "The 'anthropic' package with vertex support is required. Install it with 'pip install \"anthropic[vertex]\"'."
            )
            
        # Use provided project_id/region or get from environment variables
        project = project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
        reg = region or os.getenv("CLOUD_ML_REGION", "us-central1")

        if not project:
            raise ValueError(
                "Project ID must be provided either as parameter or via ANTHROPIC_VERTEX_PROJECT_ID environment variable"
            )

        return AsyncAnthropicVertex(project_id=project, region=reg)

    else:
        raise NotImplementedError(
            f"Provider '{provider}' not implemented. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )
