"""Simple provider module for LLMProc to return appropriate API clients."""

import os
from typing import Any

# Import provider constants
from llmproc.providers.constants import (
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_VERTEX,
    PROVIDER_OPENAI,
    SUPPORTED_PROVIDERS,
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

# Try importing Google GenAI SDK
try:
    from google import genai
except ImportError:
    genai = None


def get_provider_client(
    provider: str,
    model_name: str,
    project_id: str | None = None,
    region: str | None = None,
) -> Any:
    """Get the appropriate provider client.

    Args:
        provider: The provider to use (openai, anthropic, anthropic_vertex, gemini, or gemini_vertex)
        model_name: The model name to use (used for logging)
        project_id: Google Cloud project ID for Vertex AI providers
        region: Google Cloud region for Vertex AI providers

    Returns:
        The initialized client for the specified provider

    Raises:
        NotImplementedError: If the provider is not supported
        ImportError: If the required package for a provider is not installed
        ValueError: If required parameters or environment variables are missing
    """
    # Normalize provider name
    provider = provider.lower()

    if provider == PROVIDER_OPENAI:
        if AsyncOpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAI provider. Install it with 'pip install openai'."
            )
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided via OPENAI_API_KEY environment variable")
        return AsyncOpenAI(api_key=openai_api_key)

    elif provider == PROVIDER_ANTHROPIC:
        if AsyncAnthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic provider. Install it with 'pip install anthropic'."
            )
        # Get API key from environment
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key must be provided via ANTHROPIC_API_KEY environment variable")
        return AsyncAnthropic(api_key=anthropic_api_key)

    elif provider == PROVIDER_ANTHROPIC_VERTEX:
        if AsyncAnthropicVertex is None:
            raise ImportError(
                "The 'anthropic' package with vertex support is required. Install it with 'pip install \"anthropic[vertex]\"'."
            )

        # Use explicitly provided parameters first, fall back to environment variables
        project = project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
        reg = region or os.getenv("CLOUD_ML_REGION", "us-central1")

        if not project:
            raise ValueError(
                "Project ID must be provided via project_id parameter or ANTHROPIC_VERTEX_PROJECT_ID environment variable"
            )

        return AsyncAnthropicVertex(project_id=project, region=reg)

    elif provider == PROVIDER_GEMINI:
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required for Gemini provider. Install it with 'pip install google-genai'."
            )

        # Get API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not gemini_api_key:
            raise ValueError("API key must be provided via GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

        # Create the client for Google AI Studio (direct API)
        return genai.Client(api_key=gemini_api_key)

    elif provider == PROVIDER_GEMINI_VERTEX:
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required for Gemini on Vertex AI. Install it with 'pip install google-genai'."
            )

        # Use explicitly provided parameters first, fall back to environment variables
        project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        reg = region or os.getenv("CLOUD_ML_REGION", "us-central1")

        if not project:
            raise ValueError(
                "Project ID must be provided via project_id parameter or GOOGLE_CLOUD_PROJECT environment variable"
            )

        # Create the client for Vertex AI
        return genai.Client(vertexai=True, project=project, location=reg)

    else:
        raise NotImplementedError(
            f"Provider '{provider}' not implemented. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )
