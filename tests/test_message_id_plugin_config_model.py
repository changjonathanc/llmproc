"""Unit tests for MessageIDPluginConfig and loader parsing."""

from llmproc.config.program_loader import ProgramLoader
from llmproc.config.schema import (
    MessageIDPluginConfig,
    LLMProgramConfig,
    ModelConfig,
    PluginsConfig,
    PromptConfig,
)
from llmproc.plugins.message_id import MessageIDPlugin
from llmproc.program_compiler import compile_program


def test_message_id_plugin_config_defaults():
    """Default values are applied correctly."""
    cfg = MessageIDPluginConfig()
    assert cfg.enable_goto is True
    assert cfg.tools == []


def test_loader_parses_message_id_plugin_config(tmp_path):
    """ProgramLoader extracts message ID plugin settings."""
    config = LLMProgramConfig(
        model=ModelConfig(name="m", provider="anthropic"),
        prompt=PromptConfig(system_prompt=""),
        plugins=PluginsConfig(message_id={"enable_goto": False, "tools": ["goto"]}),
    )
    data = ProgramLoader._build_from_config(config, tmp_path)
    assert data.plugin_configs["message_id"]["enable_goto"] is False
    assert data.plugin_configs["message_id"]["tools"] == ["goto"]
    assert any(isinstance(p, MessageIDPlugin) for p in data.plugins)

    compiled = compile_program(data)
    assert any(isinstance(p, MessageIDPlugin) for p in compiled.plugins)
