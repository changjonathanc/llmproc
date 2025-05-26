"""Tests for Anthropic retry logic using environment variables."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from llmproc.providers.anthropic_process_executor import RateLimitError, _call_with_retry


class DummyClient:
    """Simple dummy client to simulate Anthropic API calls."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.messages = self

    async def create(self, **_: str):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RateLimitError("retry")
        return "ok"


def test_call_with_retry_defaults(monkeypatch):
    """Ensure `_call_with_retry` succeeds using default retry settings."""
    client = DummyClient(fail_times=2)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    result = asyncio.run(_call_with_retry(client, {}))

    assert result == "ok"
    assert client.calls == 3


def test_call_with_retry_env_override(monkeypatch):
    """Verify environment variables control retry attempts."""
    client = DummyClient(fail_times=5)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    monkeypatch.setenv("LLMPROC_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("LLMPROC_RETRY_INITIAL_WAIT", "0")
    monkeypatch.setenv("LLMPROC_RETRY_MAX_WAIT", "0")

    with pytest.raises(RateLimitError):
        asyncio.run(_call_with_retry(client, {}))

    assert client.calls == 2
