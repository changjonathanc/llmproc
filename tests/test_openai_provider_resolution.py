"""Tests for OpenAI provider resolution logic."""

import pytest

from llmproc.providers.constants import (
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CHAT,
    PROVIDER_OPENAI_RESPONSE,
    resolve_openai_provider,
)


class TestOpenAIProviderResolution:
    """Tests for OpenAI provider auto-selection logic."""

    def test_resolve_openai_reasoning_models(self):
        """Test that reasoning models resolve to openai_response."""
        test_cases = [
            "o1-preview",
            "o1-mini",
            "o3-mini",
            "o3",
            "o4-mini",
            "o4",
        ]

        for model_name in test_cases:
            resolved = resolve_openai_provider(model_name, PROVIDER_OPENAI)
            assert resolved == PROVIDER_OPENAI_RESPONSE, f"Model {model_name} should resolve to openai_response"

    def test_resolve_openai_chat_models(self):
        """Test that non-reasoning models resolve to openai_chat."""
        test_cases = [
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]

        for model_name in test_cases:
            resolved = resolve_openai_provider(model_name, PROVIDER_OPENAI)
            assert resolved == PROVIDER_OPENAI_CHAT, f"Model {model_name} should resolve to openai_chat"

    def test_resolve_non_openai_provider_unchanged(self):
        """Test that non-openai providers are returned unchanged."""
        test_cases = [
            (PROVIDER_OPENAI_CHAT, "gpt-4o"),
            (PROVIDER_OPENAI_RESPONSE, "o3-mini"),
            ("anthropic", "claude-3-5-sonnet"),
            ("anthropic-vertex", "claude-3-haiku"),
            ("gemini", "gemini-pro"),
        ]

        for provider, model_name in test_cases:
            resolved = resolve_openai_provider(model_name, provider)
            assert resolved == provider, f"Provider {provider} should remain unchanged"

    def test_edge_cases(self):
        """Test edge cases for model name patterns."""
        # Model starting with 'o' but not a reasoning model should still use response
        # This is by design - the 'o' prefix is the heuristic
        assert resolve_openai_provider("openai-model", PROVIDER_OPENAI) == PROVIDER_OPENAI_RESPONSE

        # Empty model name with openai provider should default to chat
        assert resolve_openai_provider("", PROVIDER_OPENAI) == PROVIDER_OPENAI_CHAT

        # Model with mixed case should work
        assert resolve_openai_provider("GPT-4O", PROVIDER_OPENAI) == PROVIDER_OPENAI_CHAT
        assert resolve_openai_provider("O1-PREVIEW", PROVIDER_OPENAI) == PROVIDER_OPENAI_RESPONSE
