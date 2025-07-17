"""LLMProc plugins package.

This package contains example plugins and utilities for the hook system.
"""

from llmproc.config.schema import (
    FileDescriptorPluginConfig,
    MessageIDPluginConfig,
    StderrPluginConfig,
)
from llmproc.extensions.examples import (
    CommandInfoPlugin,
    EnvVarInfoPlugin,
    FileMapPlugin,
    TimestampPlugin,
    ToolApprovalPlugin,
    ToolFilterPlugin,
)
from llmproc.plugins.file_descriptor import FileDescriptorPlugin
from llmproc.plugins.message_id import MessageIDPlugin
from llmproc.plugins.override_utils import apply_tool_overrides
from llmproc.plugins.registry import register_plugin
from llmproc.plugins.spawn import SpawnPlugin
from llmproc.plugins.stderr import StderrPlugin

register_plugin(
    "file_descriptor",
    lambda cfg: FileDescriptorPlugin(FileDescriptorPluginConfig(**cfg)),
)

register_plugin(
    "message_id",
    lambda cfg: MessageIDPlugin(MessageIDPluginConfig(**cfg)),
)

register_plugin(
    "spawn",
    lambda cfg: SpawnPlugin(cfg.get("linked_programs", {}), cfg.get("linked_program_descriptions", {})),
)

register_plugin(
    "stderr",
    lambda cfg: StderrPlugin(StderrPluginConfig(**cfg)),
)


def _load_preload_files_plugin():
    from llmproc.plugins.preload_files import PreloadFilesPlugin

    return PreloadFilesPlugin


# EnvInfoPlugin imported lazily to avoid circular imports during package init
def _load_env_info_plugin():
    from llmproc.plugins.env_info.plugin import EnvInfoPlugin

    return EnvInfoPlugin


__all__ = [
    "FileDescriptorPlugin",
    "MessageIDPlugin",
    "TimestampPlugin",
    "ToolApprovalPlugin",
    "ToolFilterPlugin",
    "EnvVarInfoPlugin",
    "CommandInfoPlugin",
    "FileMapPlugin",
    "apply_tool_overrides",
    "PreloadFilesPlugin",
    "SpawnPlugin",
    "StderrPlugin",
]


# Provide EnvInfoPlugin attribute lazily
def __getattr__(name):
    if name == "EnvInfoPlugin":
        return _load_env_info_plugin()
    if name == "PreloadFilesPlugin":
        return _load_preload_files_plugin()
    raise AttributeError(name)
