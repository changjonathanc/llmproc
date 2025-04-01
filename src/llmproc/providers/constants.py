"""Constants for LLMProc providers."""

# Provider identifiers
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_ANTHROPIC_VERTEX = "anthropic_vertex"

# Set of all supported providers
SUPPORTED_PROVIDERS = {
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX
}

# Set of Anthropic providers (both direct API and Vertex AI)
ANTHROPIC_PROVIDERS = {
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX
}