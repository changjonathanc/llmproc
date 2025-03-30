"""Tests for the provider constants."""

import pytest

from llmproc.providers.constants import (
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_ANTHROPIC_VERTEX,
    SUPPORTED_PROVIDERS,
    ANTHROPIC_PROVIDERS
)


def test_provider_constants():
    """Test that the provider constants are defined properly."""
    # Check individual provider constants
    assert PROVIDER_OPENAI == "openai"
    assert PROVIDER_ANTHROPIC == "anthropic"
    assert PROVIDER_ANTHROPIC_VERTEX == "anthropic_vertex"
    
    # Check that the sets contain the expected providers
    assert SUPPORTED_PROVIDERS == {
        PROVIDER_OPENAI,
        PROVIDER_ANTHROPIC,
        PROVIDER_ANTHROPIC_VERTEX
    }
    
    assert ANTHROPIC_PROVIDERS == {
        PROVIDER_ANTHROPIC,
        PROVIDER_ANTHROPIC_VERTEX
    }


def test_provider_set_membership():
    """Test provider set membership checks."""
    # Test SUPPORTED_PROVIDERS membership
    assert PROVIDER_OPENAI in SUPPORTED_PROVIDERS
    assert PROVIDER_ANTHROPIC in SUPPORTED_PROVIDERS
    assert PROVIDER_ANTHROPIC_VERTEX in SUPPORTED_PROVIDERS
    assert "unsupported_provider" not in SUPPORTED_PROVIDERS
    
    # Test ANTHROPIC_PROVIDERS membership
    assert PROVIDER_ANTHROPIC in ANTHROPIC_PROVIDERS
    assert PROVIDER_ANTHROPIC_VERTEX in ANTHROPIC_PROVIDERS
    assert PROVIDER_OPENAI not in ANTHROPIC_PROVIDERS