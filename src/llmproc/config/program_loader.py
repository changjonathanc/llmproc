"""Program loader for loading LLMProgram configurations from various sources."""

import logging
import tomllib
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError

from llmproc.config.program_data import ProgramConfigData
from llmproc.config.schema import (
    EnvInfoPluginConfig,
    LLMProgramConfig,
)
from llmproc.config.utils import resolve_path
from llmproc.plugins.env_info.plugin import EnvInfoPlugin
from llmproc.plugins.preload_files import PreloadFilesPlugin
from llmproc.plugins.registry import create_plugin, registered_plugins
from llmproc.plugins.spawn import SpawnPlugin

# Set up logger
logger = logging.getLogger(__name__)


# =========================================================================
# MODULE-LEVEL HELPER FUNCTIONS
# =========================================================================


def normalize_base_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Return ``base_dir`` as a :class:`Path`, defaulting to ``cwd``."""
    if base_dir is None:
        return Path.cwd()
    elif isinstance(base_dir, str):
        return Path(base_dir)
    return base_dir


def build_preload_plugin(config: LLMProgramConfig, base_dir: Path) -> PreloadFilesPlugin | None:
    """Return a :class:`PreloadFilesPlugin` for the config or ``None``."""
    if not config.plugins or not getattr(config.plugins, "preload_files", None):
        return None

    preload_cfg = config.plugins.preload_files
    if not preload_cfg.files:
        return None

    relative_to = getattr(preload_cfg, "relative_to", "program")

    return PreloadFilesPlugin(
        list(preload_cfg.files),
        base_dir,
        relative_to,
    )


def build_env_info_plugin(config: LLMProgramConfig) -> EnvInfoPlugin | None:
    """Return an :class:`EnvInfoPlugin` for the config or ``None``."""
    if not config.plugins or not getattr(config.plugins, "env_info", None):
        return None

    env_cfg = config.plugins.env_info
    if not env_cfg.variables:
        return None

    validated_cfg = EnvInfoPluginConfig.model_validate(env_cfg.model_dump())
    return EnvInfoPlugin(validated_cfg)


def build_spawn_plugin(config: LLMProgramConfig, base_dir: Path) -> SpawnPlugin | None:
    """Return a :class:`SpawnPlugin` for the config or ``None``."""
    if not config.plugins or not getattr(config.plugins, "spawn", None):
        return None

    spawn_cfg = config.plugins.spawn

    linked_programs = {}
    if spawn_cfg.linked_programs:
        for name, rel_path in spawn_cfg.linked_programs.items():
            try:
                path = resolve_path(
                    rel_path,
                    base_dir=base_dir,
                    must_exist=True,
                    error_prefix="Linked program file",
                )
                from llmproc.program import LLMProgram

                linked_programs[name] = LLMProgram.from_file(path, include_linked=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(str(e))

    return SpawnPlugin(linked_programs, spawn_cfg.linked_program_descriptions or {})


def resolve_mcp_config(config: LLMProgramConfig, base_dir: Path) -> str:
    """Return the MCP config path or ``None`` if not defined."""
    if not config.mcp or not config.mcp.config_path:
        return None

    try:
        return str(
            resolve_path(
                config.mcp.config_path,
                base_dir,
                must_exist=True,
                error_prefix="MCP config file",
            )
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))


def resolve_mcp_servers(config: LLMProgramConfig) -> dict | None:
    """Return embedded MCP servers dictionary or ``None``."""
    if not config.mcp or not config.mcp.servers:
        return None
    return config.mcp.servers


class ProgramLoader:
    """Load and build program configuration data from files or dictionaries."""

    # =========================================================================
    # CORE CONFIGURATION METHODS
    # =========================================================================

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> ProgramConfigData:
        """Create configuration data from a dictionary."""
        # Normalize base_dir
        base_dir = normalize_base_dir(base_dir)

        # Validate with Pydantic
        try:
            config = LLMProgramConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid program configuration dictionary:\n{str(e)}")

        # Build configuration data (linked programs remain as strings)
        return cls._build_from_config(config, base_dir)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        *,
        format: str = "auto",
    ) -> ProgramConfigData:
        """Load program configuration data from a TOML or YAML file."""
        path = resolve_path(file_path, must_exist=True, error_prefix="Program file")

        config_data = cls._compile_single_file(path, format=format)

        return config_data

    # =========================================================================
    # FILE PROCESSING METHODS
    # =========================================================================

    @classmethod
    def _compile_single_file(cls, path: Path, *, format: str = "auto") -> ProgramConfigData:
        """Compile program configuration data from a TOML or YAML file."""
        fmt = format.lower()

        if fmt not in {"auto", "yaml", "toml"}:
            raise ValueError(f"Unknown format: {format}")

        if fmt == "auto":
            suffix = path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                fmt = "yaml"
            elif suffix == ".toml":
                fmt = "toml"
            else:
                raise ValueError(f"Unsupported file format: {suffix} (expected .toml, .yaml, or .yml)")

        if fmt == "yaml":
            try:
                with path.open("r") as f:
                    config_data = yaml.safe_load(f)
            except Exception as e:
                raise ValueError(f"Error loading YAML file {path}: {str(e)}")
        else:  # fmt == "toml"
            try:
                with path.open("rb") as f:
                    config_data = tomllib.load(f)
            except Exception as e:
                raise ValueError(f"Error loading TOML file {path}: {str(e)}")

        # Use from_dict to handle the rest of the process
        return cls.from_dict(config_data, base_dir=path.parent)

    # =========================================================================
    # CONFIGURATION BUILDING METHODS
    # =========================================================================

    @classmethod
    def _build_from_config(cls, config: LLMProgramConfig, base_dir: Path) -> ProgramConfigData:
        """Construct :class:`ProgramConfigData` from a validated config."""
        # Resolve system prompt
        system_prompt = config.prompt.resolve(base_dir)

        # Process linked programs or spawn plugin configuration
        spawn_plugin = build_spawn_plugin(config, base_dir)

        # Extract tools from config
        tools_list = config.tools.builtin if config.tools else []

        # Incorporate MCP tool descriptors from [tools.mcp]
        if config.tools and config.tools.mcp:
            tools_list.extend(config.tools.mcp.build_mcp_tools())

        preload_plugin = build_preload_plugin(config, base_dir)
        env_plugin = build_env_info_plugin(config)
        plugins = []
        if preload_plugin:
            plugins.append(preload_plugin)
        if env_plugin:
            plugins.append(env_plugin)
        if spawn_plugin:
            plugins.append(spawn_plugin)

        plugin_configs: dict[str, dict] = {}
        if config.plugins:
            if config.plugins.preload_files and preload_plugin:
                plugin_configs["preload_files"] = config.plugins.preload_files.model_dump()
            if config.plugins.env_info and env_plugin:
                plugin_configs["env_info"] = config.plugins.env_info.model_dump()
            for name, cfg_dict in config.plugins.model_dump().items():
                if cfg_dict is None:
                    continue
                if name == "spawn" and spawn_plugin:
                    plugin_configs[name] = cfg_dict
                    continue
                if name in registered_plugins():
                    plugins.append(create_plugin(name, cfg_dict))
                    plugin_configs[name] = cfg_dict

        return ProgramConfigData(
            model_name=config.model.name,
            provider=config.model.provider,
            system_prompt=system_prompt,
            parameters=config.parameters.model_dump(exclude_none=True),
            plugins=plugins or None,
            plugin_configs=plugin_configs or None,
            mcp_config_path=resolve_mcp_config(config, base_dir),
            mcp_servers=resolve_mcp_servers(config),
            tools=tools_list,
            tools_config=config.tools,  # Store original ToolsConfig with official fields
            base_dir=base_dir,
            project_id=config.model.project_id,
            region=config.model.region,
            user_prompt=config.prompt.user if hasattr(config.prompt, "user") else None,
            max_iterations=config.model.max_iterations,
        )
