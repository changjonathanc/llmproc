"""Tests for the provider constants."""

import pytest

from llmproc.providers.constants import (
    ANTHROPIC_PROVIDERS,
    GEMINI_PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_VERTEX,
    PROVIDER_OPENAI,
    SUPPORTED_PROVIDERS,
    VERTEX_PROVIDERS,
)


def test_provider_constants():
    """Test that the provider constants are defined properly."""
    # Check individual provider constants
    assert PROVIDER_OPENAI == "openai"
    assert PROVIDER_ANTHROPIC == "anthropic"
    assert PROVIDER_ANTHROPIC_VERTEX == "anthropic_vertex"
    assert PROVIDER_GEMINI == "gemini"
    assert PROVIDER_GEMINI_VERTEX == "gemini_vertex"

    # Check that the sets contain the expected providers
    assert SUPPORTED_PROVIDERS == {
        PROVIDER_OPENAI,
        PROVIDER_ANTHROPIC,
        PROVIDER_ANTHROPIC_VERTEX,
        PROVIDER_GEMINI,
        PROVIDER_GEMINI_VERTEX,
    }

    assert ANTHROPIC_PROVIDERS == {PROVIDER_ANTHROPIC, PROVIDER_ANTHROPIC_VERTEX}
    assert GEMINI_PROVIDERS == {PROVIDER_GEMINI, PROVIDER_GEMINI_VERTEX}
    assert VERTEX_PROVIDERS == {PROVIDER_ANTHROPIC_VERTEX, PROVIDER_GEMINI_VERTEX}


def test_provider_set_membership():
    """Test provider set membership checks."""
    # Test SUPPORTED_PROVIDERS membership
    assert PROVIDER_OPENAI in SUPPORTED_PROVIDERS
    assert PROVIDER_ANTHROPIC in SUPPORTED_PROVIDERS
    assert PROVIDER_ANTHROPIC_VERTEX in SUPPORTED_PROVIDERS
    assert PROVIDER_GEMINI in SUPPORTED_PROVIDERS
    assert PROVIDER_GEMINI_VERTEX in SUPPORTED_PROVIDERS
    assert "unsupported_provider" not in SUPPORTED_PROVIDERS

    # Test ANTHROPIC_PROVIDERS membership
    assert PROVIDER_ANTHROPIC in ANTHROPIC_PROVIDERS
    assert PROVIDER_ANTHROPIC_VERTEX in ANTHROPIC_PROVIDERS
    assert PROVIDER_OPENAI not in ANTHROPIC_PROVIDERS
    assert PROVIDER_GEMINI not in ANTHROPIC_PROVIDERS

    # Test GEMINI_PROVIDERS membership
    assert PROVIDER_GEMINI in GEMINI_PROVIDERS
    assert PROVIDER_GEMINI_VERTEX in GEMINI_PROVIDERS
    assert PROVIDER_OPENAI not in GEMINI_PROVIDERS
    assert PROVIDER_ANTHROPIC not in GEMINI_PROVIDERS

    # Test VERTEX_PROVIDERS membership
    assert PROVIDER_ANTHROPIC_VERTEX in VERTEX_PROVIDERS
    assert PROVIDER_GEMINI_VERTEX in VERTEX_PROVIDERS
    assert PROVIDER_OPENAI not in VERTEX_PROVIDERS
    assert PROVIDER_ANTHROPIC not in VERTEX_PROVIDERS
    assert PROVIDER_GEMINI not in VERTEX_PROVIDERS
