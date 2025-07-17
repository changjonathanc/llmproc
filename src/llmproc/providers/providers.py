"""Simple provider module for LLMProc to return appropriate API clients."""

import os
import webbrowser
from collections.abc import Callable
from typing import Any

# Import provider constants
from llmproc.providers.constants import (
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX,
    PROVIDER_CLAUDE_CODE,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_VERTEX,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CHAT,
    PROVIDER_OPENAI_RESPONSE,
    SUPPORTED_PROVIDERS,
)

# Try importing providers, set to None if packages aren't installed
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from anthropic import AsyncAnthropicVertex
except ImportError:
    AsyncAnthropicVertex = None

from .claude_code_oauth import AnthropicOAuth

# Try importing Google GenAI SDK
try:
    from google import genai
except ImportError:
    genai = None


def _openai_client(*_: str, **__: str) -> Any:
    """Create OpenAI client."""
    if AsyncOpenAI is None:
        raise ImportError("The 'openai' package is required for OpenAI provider. Install it with 'pip install openai'.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via OPENAI_API_KEY environment variable")
    return AsyncOpenAI(api_key=api_key)


def _anthropic_client(*_: str, **__: str) -> Any:
    """Create Anthropic client."""
    if AsyncAnthropic is None:
        raise ImportError(
            "The 'anthropic' package is required for Anthropic provider. Install it with 'pip install anthropic'."
        )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key must be provided via ANTHROPIC_API_KEY environment variable")
    return AsyncAnthropic(api_key=api_key)


def _anthropic_vertex_client(project_id: str | None = None, region: str | None = None) -> Any:
    """Create Anthropic Vertex client."""
    if AsyncAnthropicVertex is None:
        raise ImportError(
            "The 'anthropic' package with vertex support is required. Install it with 'pip install \"anthropic[vertex]\"'."
        )
    project = project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
    reg = region or os.getenv("CLOUD_ML_REGION", "us-central1")
    if not project:
        raise ValueError(
            "Project ID must be provided via project_id parameter or ANTHROPIC_VERTEX_PROJECT_ID environment variable"
        )
    return AsyncAnthropicVertex(project_id=project, region=reg)


def _claude_code_client(*_: str, **__: str) -> Any:
    """Create Claude Code client using OAuth."""
    if AsyncAnthropic is None or AnthropicOAuth is None:
        raise ImportError("The 'anthropic' package and OAuth helper are required for Claude Code provider.")
    oauth = AnthropicOAuth()
    token = oauth.get_access_token()
    if not token:
        auth_data = oauth.authorize()
        print("Authorize Claude Code by visiting the following URL:\n" + auth_data["url"])
        try:
            webbrowser.open(auth_data["url"])
        except Exception:
            pass
        code = input("Enter authorization code: ").strip()
        oauth.exchange_code(code, auth_data["verifier"])
        token = oauth.get_access_token()
        if not token:
            raise ValueError("Authentication failed")
    headers = {"anthropic-version": "2023-06-01", "anthropic-beta": "oauth-2025-04-20"}
    return AsyncAnthropic(auth_token=token, default_headers=headers)


def _gemini_client(*_: str, **__: str) -> Any:
    """Create Gemini client."""
    if genai is None:
        raise ImportError(
            "The 'google-genai' package is required for Gemini provider. Install it with 'pip install google-genai'."
        )
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    return genai.Client(api_key=api_key)


def _gemini_vertex_client(project_id: str | None = None, region: str | None = None) -> Any:
    """Create Gemini Vertex client."""
    if genai is None:
        raise ImportError(
            "The 'google-genai' package is required for Gemini on Vertex AI. Install it with 'pip install google-genai'."
        )
    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    reg = region or os.getenv("CLOUD_ML_REGION", "us-central1")
    if not project:
        raise ValueError(
            "Project ID must be provided via project_id parameter or GOOGLE_CLOUD_PROJECT environment variable"
        )
    return genai.Client(vertexai=True, project=project, location=reg)


_CLIENT_CREATORS: dict[str, Callable[..., Any]] = {
    PROVIDER_OPENAI: _openai_client,
    PROVIDER_OPENAI_CHAT: _openai_client,
    PROVIDER_OPENAI_RESPONSE: _openai_client,
    PROVIDER_ANTHROPIC: _anthropic_client,
    PROVIDER_ANTHROPIC_VERTEX: _anthropic_vertex_client,
    PROVIDER_CLAUDE_CODE: _claude_code_client,
    PROVIDER_GEMINI: _gemini_client,
    PROVIDER_GEMINI_VERTEX: _gemini_vertex_client,
}


def get_provider_client(
    provider: str,
    project_id: str | None = None,
    region: str | None = None,
) -> Any:
    """Return the provider client for ``provider``.

    Args:
        provider: Provider identifier.
        project_id: Google Cloud project for Vertex providers.
        region: Google Cloud region for Vertex providers.

    Returns:
        Initialized provider client.

    Raises:
        NotImplementedError: If the provider is unsupported.
        ImportError: If the required package is missing.
        ValueError: If required configuration is missing.
    """
    provider = provider.lower()

    creator = _CLIENT_CREATORS.get(provider)
    if creator is None:
        raise NotImplementedError(
            f"Provider '{provider}' not implemented. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    return creator(project_id=project_id, region=region)
