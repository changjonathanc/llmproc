"""Tests for program-level plugin registration."""

from pathlib import Path

import pytest

from llmproc.program import LLMProgram


class DummyPlugin:
    def hook_user_input(self, user_input, process):
        return user_input


class InvalidPlugin:
    pass


def test_add_plugins_records_plugins():
    """LLMProgram.add_plugins stores plugins in the program."""
    program = LLMProgram(model_name="m", provider="p")
    plugin = DummyPlugin()
    program.add_plugins(plugin)
    assert program.plugins == [plugin]


def test_compile_accepts_valid_plugin():
    """Program.compile works with valid plugins."""
    program = LLMProgram(model_name="m", provider="p")
    plugin = DummyPlugin()
    program.add_plugins(plugin)
    compiled = program.compile()
    assert compiled is program


def test_add_plugins_invalid_plugin():
    """add_plugins raises for objects without hook methods."""
    program = LLMProgram(model_name="m", provider="p")
    with pytest.raises(ValueError):
        program.add_plugins(InvalidPlugin())


def test_prepare_process_config_includes_plugins():
    """prepare_process_config copies program.plugins into ProcessConfig."""
    from unittest.mock import patch

    from llmproc import program_exec

    program = LLMProgram(model_name="m", provider="p")
    plugin = DummyPlugin()
    program.add_plugins(plugin)

    with patch("llmproc.program_exec.get_provider_client", return_value=None):
        cfg = program_exec.prepare_process_config(program)

    assert cfg.plugins == [plugin]


def test_instantiate_process_sets_immutable_plugins():
    """instantiate_process creates process with immutable plugin tuple."""
    from llmproc import program_exec
    from llmproc.config.process_config import ProcessConfig

    program = LLMProgram(model_name="m", provider="p")
    plugin = DummyPlugin()

    cfg = ProcessConfig(
        program=program,
        model_name="m",
        provider="p",
        base_system_prompt="",
        plugins=[plugin],
    )

    process = program_exec.instantiate_process(cfg)

    assert list(process.plugins) == [plugin]


def test_compile_registers_builtin_plugins(tmp_path: Path):
    """compile() adds built-in plugins based on configuration."""
    from llmproc.config import EnvInfoConfig
    from llmproc.config.schema import FileDescriptorPluginConfig
    from llmproc.plugins.env_info.plugin import EnvInfoPlugin
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin
    from llmproc.plugins.preload_files import PreloadFilesPlugin

    dummy = tmp_path / "foo.txt"
    dummy.write_text("bar")

    program = LLMProgram(model_name="m", provider="p")
    program.add_plugins(
        PreloadFilesPlugin(["foo.txt"], base_dir=tmp_path),
        EnvInfoPlugin(EnvInfoConfig(variables=["platform"])),
        FileDescriptorPlugin(FileDescriptorPluginConfig()),
    )

    program.compile()

    plugin_types = {type(p) for p in program.plugins}
    assert PreloadFilesPlugin in plugin_types
    assert EnvInfoPlugin in plugin_types
    assert FileDescriptorPlugin in plugin_types


def test_add_plugins_registers_file_descriptor_plugin():
    """add_plugins immediately registers FileDescriptorPlugin."""
    from llmproc.config.schema import FileDescriptorPluginConfig
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin

    program = LLMProgram(model_name="m", provider="p")
    program.add_plugins(FileDescriptorPlugin(FileDescriptorPluginConfig()))

    assert any(isinstance(p, FileDescriptorPlugin) for p in program.plugins)


def test_message_id_plugin_must_be_explicit():
    """Goto tool requires explicit MessageIDPlugin registration."""
    from llmproc.config.schema import MessageIDPluginConfig
    from llmproc.plugins.message_id import MessageIDPlugin

    program = LLMProgram(model_name="m", provider="p")
    # Auto-registration removed - must explicitly add plugin
    program.add_plugins(MessageIDPlugin(MessageIDPluginConfig(enable_goto=True)))
    program.compile()

    plugin_types = {type(p) for p in program.plugins}
    assert MessageIDPlugin in plugin_types
