"""Tests for the provider constants."""

import pytest
from llmproc.providers.constants import (
    ANTHROPIC_PROVIDERS,
    GEMINI_PROVIDERS,
    OPENAI_PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_VERTEX,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CHAT,
    PROVIDER_OPENAI_RESPONSE,
    SUPPORTED_PROVIDERS,
    VERTEX_PROVIDERS,
)


def test_provider_constants():
    """Test that the provider constants are defined properly."""
    # Check individual provider constants
    assert PROVIDER_OPENAI == "openai"
    assert PROVIDER_OPENAI_CHAT == "openai_chat"
    assert PROVIDER_OPENAI_RESPONSE == "openai_response"
    assert PROVIDER_ANTHROPIC == "anthropic"
    assert PROVIDER_GEMINI == "gemini"
    assert PROVIDER_GEMINI_VERTEX == "gemini_vertex"

    # Check that the sets contain the expected providers
    expected_supported = {
        PROVIDER_OPENAI,
        PROVIDER_OPENAI_CHAT,
        PROVIDER_OPENAI_RESPONSE,
        PROVIDER_ANTHROPIC,
        PROVIDER_GEMINI,
        PROVIDER_GEMINI_VERTEX,
    }
    assert expected_supported.issubset(SUPPORTED_PROVIDERS)

    assert OPENAI_PROVIDERS == {PROVIDER_OPENAI, PROVIDER_OPENAI_CHAT, PROVIDER_OPENAI_RESPONSE}
    assert {PROVIDER_ANTHROPIC}.issubset(ANTHROPIC_PROVIDERS)
    assert GEMINI_PROVIDERS == {PROVIDER_GEMINI, PROVIDER_GEMINI_VERTEX}
    assert {PROVIDER_GEMINI_VERTEX}.issubset(VERTEX_PROVIDERS)


def test_provider_set_membership():
    """Test provider set membership checks."""
    # Test SUPPORTED_PROVIDERS membership
    assert PROVIDER_OPENAI in SUPPORTED_PROVIDERS
    assert PROVIDER_OPENAI_CHAT in SUPPORTED_PROVIDERS
    assert PROVIDER_OPENAI_RESPONSE in SUPPORTED_PROVIDERS
    assert PROVIDER_ANTHROPIC in SUPPORTED_PROVIDERS
    assert PROVIDER_GEMINI in SUPPORTED_PROVIDERS
    assert PROVIDER_GEMINI_VERTEX in SUPPORTED_PROVIDERS
    assert "unsupported_provider" not in SUPPORTED_PROVIDERS

    # Test OPENAI_PROVIDERS membership
    assert PROVIDER_OPENAI in OPENAI_PROVIDERS
    assert PROVIDER_OPENAI_CHAT in OPENAI_PROVIDERS
    assert PROVIDER_OPENAI_RESPONSE in OPENAI_PROVIDERS
    assert PROVIDER_ANTHROPIC not in OPENAI_PROVIDERS
    assert PROVIDER_GEMINI not in OPENAI_PROVIDERS

    # Test ANTHROPIC_PROVIDERS membership
    assert PROVIDER_ANTHROPIC in ANTHROPIC_PROVIDERS
    assert PROVIDER_OPENAI not in ANTHROPIC_PROVIDERS
    assert PROVIDER_GEMINI not in ANTHROPIC_PROVIDERS

    # Test GEMINI_PROVIDERS membership
    assert PROVIDER_GEMINI in GEMINI_PROVIDERS
    assert PROVIDER_GEMINI_VERTEX in GEMINI_PROVIDERS
    assert PROVIDER_OPENAI not in GEMINI_PROVIDERS
    assert PROVIDER_ANTHROPIC not in GEMINI_PROVIDERS

    # Test VERTEX_PROVIDERS membership
    assert PROVIDER_GEMINI_VERTEX in VERTEX_PROVIDERS
    assert PROVIDER_OPENAI not in VERTEX_PROVIDERS
    assert PROVIDER_ANTHROPIC not in VERTEX_PROVIDERS
    assert PROVIDER_GEMINI not in VERTEX_PROVIDERS
