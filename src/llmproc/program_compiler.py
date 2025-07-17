"""Compilation utilities for :class:`LLMProgram`."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from llmproc._program_docs import COMPILE_SELF
from llmproc.config.program_data import ProgramConfigData
from llmproc.config.program_loader import ProgramLoader

# Plugin imports removed - plugins created via registry in program_loader.py

logger = logging.getLogger(__name__)


def _compile_linked_programs(cfg: ProgramConfigData) -> None:
    """Compile linked programs defined via :class:`SpawnPlugin`."""
    if not cfg.plugins:
        return

    from llmproc.plugins.spawn import SpawnPlugin
    from llmproc.program import LLMProgram  # local import to avoid circular dependency

    spawn_plugin = next((p for p in cfg.plugins if isinstance(p, SpawnPlugin)), None)
    if not spawn_plugin or not spawn_plugin.linked_programs:
        return

    compiled: dict[str, ProgramConfigData | LLMProgram] = {}

    for name, item in spawn_plugin.linked_programs.items():
        if isinstance(item, str):
            try:
                linked = ProgramLoader.from_file(Path(item))
                compiled[name] = LLMProgram._from_config_data(compile_program(linked))
            except FileNotFoundError:
                warnings.warn(f"Linked program not found: {item}", stacklevel=2)
        elif isinstance(item, LLMProgram):
            if not item.compiled:
                item.compile()
            compiled[name] = item
        elif isinstance(item, ProgramConfigData):
            compiled[name] = LLMProgram._from_config_data(compile_program(item))
        else:
            raise ValueError(f"Invalid linked program type for {name}: {type(item)}")

    spawn_plugin.linked_programs = compiled


# Removed redundant _register_builtin_plugins function
# Plugins are now created properly in program_loader.py via the registry system


def compile_program(cfg: ProgramConfigData, system_prompt_file: str | None = None) -> ProgramConfigData:
    """Validate and finalize a :class:`ProgramConfigData`."""
    if system_prompt_file and not cfg.system_prompt:
        try:
            cfg.system_prompt = Path(system_prompt_file).read_text()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"System prompt file not found: {system_prompt_file}") from e

    if cfg.system_prompt is None:
        cfg.system_prompt = ""

    if not cfg.model_name or not cfg.provider:
        missing = []
        if not cfg.model_name:
            missing.append("model_name")
        if not cfg.provider:
            missing.append("provider")
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    _compile_linked_programs(cfg)
    # Plugin registration removed - handled in program_loader.py

    return cfg


compile_program.__doc__ = COMPILE_SELF

__all__ = ["compile_program"]
