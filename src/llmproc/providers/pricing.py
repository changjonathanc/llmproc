"""Pricing tables for provider models used by LLMProc."""

# USD cost per million tokens for Anthropic models (28 May 2025)
CLAUDE_PRICING_USD_PER_MTOK = {
    "claude-opus-4": {
        "input_tokens": 15.00,
        "output_tokens": 75.00,
        "cache_creation_input_tokens": 18.75,
        "cache_read_input_tokens": 1.50,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 18.75,
            "ephemeral_1h_input_tokens": 30.00,
        },
    },
    "claude-sonnet-4": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 3.75,
            "ephemeral_1h_input_tokens": 6.00,
        },
    },
    "claude-3-7-sonnet": {
        "input_tokens": 3.00,
        "output_tokens": 15.00,
        "cache_creation_input_tokens": 3.75,
        "cache_read_input_tokens": 0.30,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 3.75,
            "ephemeral_1h_input_tokens": 6.00,
        },
    },
}


def get_claude_pricing(model: str) -> dict | None:
    """Return pricing info for a versioned Anthropic model."""
    if model in CLAUDE_PRICING_USD_PER_MTOK:
        return CLAUDE_PRICING_USD_PER_MTOK[model]
    for prefix in ("claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet"):
        if model.startswith(prefix):
            return CLAUDE_PRICING_USD_PER_MTOK.get(prefix)
    return None
