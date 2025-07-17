"""Constants for LLMProc providers."""

# Provider identifiers
PROVIDER_OPENAI = "openai"
PROVIDER_OPENAI_CHAT = "openai_chat"
PROVIDER_OPENAI_RESPONSE = "openai_response"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_ANTHROPIC_VERTEX = "anthropic_vertex"
PROVIDER_CLAUDE_CODE = "claude_code"
PROVIDER_GEMINI = "gemini"
PROVIDER_GEMINI_VERTEX = "gemini_vertex"

# Set of all supported providers
SUPPORTED_PROVIDERS = {
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CHAT,
    PROVIDER_OPENAI_RESPONSE,
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX,
    PROVIDER_CLAUDE_CODE,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_VERTEX,
}

# Set of OpenAI providers (generic and specific)
OPENAI_PROVIDERS = {PROVIDER_OPENAI, PROVIDER_OPENAI_CHAT, PROVIDER_OPENAI_RESPONSE}

# Set of Anthropic providers (both direct API and Vertex AI)
ANTHROPIC_PROVIDERS = {PROVIDER_ANTHROPIC, PROVIDER_ANTHROPIC_VERTEX, PROVIDER_CLAUDE_CODE}

# Set of Gemini providers
GEMINI_PROVIDERS = {PROVIDER_GEMINI, PROVIDER_GEMINI_VERTEX}

# Set of Vertex AI providers
VERTEX_PROVIDERS = {PROVIDER_ANTHROPIC_VERTEX, PROVIDER_GEMINI_VERTEX}


def resolve_openai_provider(model_name: str, provider: str) -> str:
    """Resolve generic 'openai' provider to specific implementation based on model.

    Args:
        model_name: The model name to check
        provider: The provider string to resolve

    Returns:
        Resolved provider string. If provider is not 'openai', returns as-is.
        For 'openai' provider:
        - Models starting with 'o' → 'openai_response' (o1, o3, o4, etc.)
        - All other models → 'openai_chat' (gpt-4, gpt-4o, etc.)
    """
    if provider != PROVIDER_OPENAI:
        return provider  # Already specific, return as-is

    # Auto-select based on model name (case-insensitive)
    if model_name.lower().startswith("o"):
        return PROVIDER_OPENAI_RESPONSE
    else:
        return PROVIDER_OPENAI_CHAT
