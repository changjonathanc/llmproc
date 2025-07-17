"""Unit tests for the ProcessConfig dataclass."""

from unittest.mock import MagicMock

import pytest
from llmproc.common.access_control import AccessLevel
from llmproc.config.process_config import ProcessConfig


def test_defaults_and_basic_fields():
    """Defaults are applied when optional fields are omitted."""
    program = MagicMock()
    cfg = ProcessConfig(
        program=program,
        model_name="m",
        provider="openai",
        base_system_prompt="orig",
    )

    assert cfg.api_params == {}
    assert cfg.state == []
    assert cfg.access_level == AccessLevel.ADMIN


def test_basic_required_fields():
    """Test ProcessConfig with basic required fields."""
    program = MagicMock()
    cfg = ProcessConfig(
        program=program,
        model_name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        base_system_prompt="You are a helpful assistant",
    )

    assert cfg.program == program
    assert cfg.model_name == "claude-3-5-sonnet-20241022"
    assert cfg.provider == "anthropic"
    assert cfg.base_system_prompt == "You are a helpful assistant"


def test_optional_fields():
    """Test ProcessConfig with optional fields set."""
    program = MagicMock()
    tool_manager = MagicMock()
    cfg = ProcessConfig(
        program=program,
        model_name="gpt-4",
        provider="openai",
        base_system_prompt="Test prompt",
        tool_manager=tool_manager,
        access_level=AccessLevel.WRITE,
        plugins=[],
    )

    assert cfg.tool_manager == tool_manager
    assert cfg.access_level == AccessLevel.WRITE
