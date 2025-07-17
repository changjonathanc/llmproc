"""Unit tests for FileDescriptorPluginConfig and loader parsing."""

from llmproc.config.program_loader import ProgramLoader
from llmproc.config.schema import (
    FileDescriptorPluginConfig,
    LLMProgramConfig,
    ModelConfig,
    PluginsConfig,
    PromptConfig,
)
from llmproc.plugins.file_descriptor import FileDescriptorPlugin
from llmproc.program_compiler import compile_program
from llmproc.config.tool import ToolConfig


def test_fd_plugin_config_defaults():
    """Defaults are applied correctly."""
    cfg = FileDescriptorPluginConfig()
    assert cfg.default_page_size == 4000
    assert cfg.max_direct_output_chars == 8000
    assert cfg.max_input_chars == 8000
    assert cfg.page_user_input is True
    assert cfg.enable_references is False
    assert cfg.tools == []


def test_loader_parses_fd_plugin_config(tmp_path):
    """ProgramLoader extracts file descriptor plugin settings."""
    config = LLMProgramConfig(
        model=ModelConfig(name="m", provider="anthropic"),
        prompt=PromptConfig(system_prompt=""),
        plugins=PluginsConfig(file_descriptor={"default_page_size": 1234}),
    )
    data = ProgramLoader._build_from_config(config, tmp_path)
    assert data.plugin_configs["file_descriptor"]["default_page_size"] == 1234
    assert any(isinstance(p, FileDescriptorPlugin) for p in data.plugins)

    compiled = compile_program(data)
    assert any(isinstance(p, FileDescriptorPlugin) for p in compiled.plugins)


def test_fd_plugin_tools_accept_toolconfig():
    """tools list can include ToolConfig objects."""
    cfg = FileDescriptorPluginConfig(
        tools=[ToolConfig(name="read_fd", description="read")]
    )
    assert isinstance(cfg.tools[0], ToolConfig)
