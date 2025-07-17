from unittest.mock import MagicMock

import pytest

from llmproc.config.process_config import ProcessConfig
from llmproc.llm_process import LLMProcess
from llmproc.plugins.stderr import StderrPlugin


def test_stderr_log_attribute_removed():
    """Accessing stderr_log now raises AttributeError."""
    plugin = StderrPlugin()
    cfg = ProcessConfig(
        program=MagicMock(),
        model_name="m",
        provider="p",
        base_system_prompt="x",
        plugins=[plugin],
    )
    process = LLMProcess(cfg)
    with pytest.raises(AttributeError):
        _ = process.stderr_log


def test_stderr_plugin_attribute_removed():
    """Accessing removed stderr_plugin raises AttributeError."""
    plugin = StderrPlugin()
    cfg = ProcessConfig(
        program=MagicMock(),
        model_name="m",
        provider="p",
        base_system_prompt="x",
        plugins=[plugin],
    )
    process = LLMProcess(cfg)
    with pytest.raises(AttributeError):
        _ = process.stderr_plugin


def test_get_stderr_log_removed():
    """Calling removed get_stderr_log raises AttributeError."""
    plugin = StderrPlugin()
    cfg = ProcessConfig(
        program=MagicMock(),
        model_name="m",
        provider="p",
        base_system_prompt="x",
        plugins=[plugin],
    )
    process = LLMProcess(cfg)
    with pytest.raises(AttributeError):
        process.get_stderr_log()
