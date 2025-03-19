"""Simple provider module for LLMProc to return appropriate API clients."""

import os
from typing import Any

# Import API clients
from openai import AsyncOpenAI as OpenAI  # Use async client for OpenAI

try:
    import anthropic
    from anthropic import AsyncAnthropic as Anthropic  # Use async client for Anthropic
    from anthropic import (
        AsyncAnthropicVertex as AnthropicVertex,
    )  # Use async client for Vertex
except ImportError:
    anthropic = None
    Anthropic = None
    AnthropicVertex = None


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
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    elif provider == "anthropic":
        if anthropic is None or Anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic provider. Install it with 'pip install anthropic'."
            )
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    elif provider == "anthropic_vertex":
        if anthropic is None or AnthropicVertex is None:
            raise ImportError(
                "The 'anthropic' package with vertex support is required. Install it with 'pip install \"anthropic[vertex]\"'."
            )

        # Use provided project_id/region or get from environment variables
        project = project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        reg = region or os.getenv("CLOUD_ML_REGION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project:
            raise ValueError(
                "Project ID must be provided either as parameter or via ANTHROPIC_VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable"
            )

        return AnthropicVertex(project_id=project, region=reg)

    else:
        raise NotImplementedError(f"Provider {provider} not implemented.")
