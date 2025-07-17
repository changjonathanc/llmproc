"""Tests for the instantiate_process function in program_exec.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llmproc.config.process_config import ProcessConfig
from llmproc.llm_process import LLMProcess
from llmproc.plugins.file_descriptor import FileDescriptorManager
from llmproc.program import LLMProgram
from llmproc.program_exec import instantiate_process


@pytest.fixture
def sample_config():
    """Create a sample ProcessConfig for testing."""
    tool_manager = MagicMock()
    fd_manager = MagicMock(spec=FileDescriptorManager)
    program = MagicMock(spec=LLMProgram)
    from llmproc.config.schema import FileDescriptorPluginConfig
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin

    plugin = FileDescriptorPlugin(FileDescriptorPluginConfig())
    plugin.fd_manager = fd_manager
    return ProcessConfig(
        program=program,
        model_name="test-model",
        provider="test-provider",
        base_system_prompt="Test system prompt",
        base_dir=Path("/test/base/dir"),
        api_params={"param1": "value1"},
        state=[],
        enriched_system_prompt=None,
        client=MagicMock(),
        tool_manager=tool_manager,
        mcp_config_path=None,
        mcp_servers=None,
        mcp_tools={},
        mcp_enabled=False,
        plugins=[plugin],
    )


def test_instantiate_process(sample_config):
    """instantiate_process should return an LLMProcess with matching attributes."""
    result = instantiate_process(sample_config)
    assert isinstance(result, LLMProcess)
    for field in ProcessConfig.__dataclass_fields__:
        if field == "loop":
            continue
        if field == "plugins":
            assert list(result.plugins) == sample_config.plugins
            continue
        if field in {"file_descriptor_enabled", "fd_manager", "references_enabled"}:
            continue
        assert getattr(result, field) == getattr(sample_config, field)


def test_instantiate_process_minimal_required():
    """instantiate_process works with minimal required parameters."""
    program = MagicMock(spec=LLMProgram)
    cfg = ProcessConfig(
        program=program,
        model_name="test-model",
        provider="test-provider",
        base_system_prompt="Test system prompt",
    )
    result = instantiate_process(cfg)
    assert isinstance(result, LLMProcess)
    assert result.program == program
    assert result.model_name == "test-model"
    assert result.provider == "test-provider"
