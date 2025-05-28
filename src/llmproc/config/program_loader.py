"""Program loader for loading LLMProgram configurations from various sources."""

import logging
import tomllib
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError

from llmproc.config.schema import EnvInfoConfig, LLMProgramConfig
from llmproc.config.tool import ToolConfig
from llmproc.config.utils import resolve_path
from llmproc.tools.mcp.constants import MCP_TOOL_SEPARATOR

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


def resolve_preload_files(config: LLMProgramConfig, base_dir: Path) -> tuple[list[str] | None, str]:
    """Return resolved preload file paths and relative mode."""
    if not config.preload or not config.preload.files:
        return None, "program"

    relative_to = getattr(config.preload, "relative_to", "program")

    if relative_to == "cwd":
        return list(config.preload.files), "cwd"

    preload_files = []
    for file_path in config.preload.files:
        try:
            resolved_path = resolve_path(file_path, base_dir, must_exist=False)
            if not resolved_path.exists():
                warnings.warn(f"Preload file not found: {resolved_path}", stacklevel=2)
            preload_files.append(str(resolved_path))
        except Exception as e:
            warnings.warn(f"Error resolving path '{file_path}': {str(e)}", stacklevel=2)
    return preload_files, "program"


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


def process_config_linked_programs(
    config: LLMProgramConfig,
) -> tuple[dict[str, str], dict[str, str]]:
    """Extract linked program paths and descriptions from ``config``."""
    if not config.linked_programs:
        return None, None

    linked_programs = {}
    linked_program_descriptions = {}

    for name, program_config in config.linked_programs.root.items():
        if isinstance(program_config, str):
            linked_programs[name] = program_config
            linked_program_descriptions[name] = ""
        else:
            linked_programs[name] = program_config.path
            linked_program_descriptions[name] = program_config.description

    return linked_programs, linked_program_descriptions


class ProgramLoader:
    """Load and build ``LLMProgram`` objects from files or dictionaries."""

    # =========================================================================
    # CORE CONFIGURATION METHODS
    # =========================================================================

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        base_dir: Optional[Union[str, Path]] = None,
        warn_linked_programs: bool = True,
    ) -> "LLMProgram":
        """Create a program from a configuration dictionary."""
        from llmproc.program import LLMProgram

        # Normalize base_dir
        base_dir = normalize_base_dir(base_dir)

        # Check for linked programs and warn only if requested
        # (don't warn when called from _compile_single_file since it will handle linking)
        if warn_linked_programs and "linked_programs" in config_dict and config_dict["linked_programs"]:
            warnings.warn(
                "Dictionary configuration with linked_programs is not automatically resolved. "
                "Use manual program linking with add_linked_program() after creation.",
                stacklevel=2,
            )

        # Validate with Pydantic
        try:
            config = LLMProgramConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid program configuration dictionary:\n{str(e)}")

        # Build program (Note: linked programs remain as strings)
        program = cls._build_from_config(config, base_dir)

        # Mark as compiled (even though linked programs aren't processed)
        # This prevents automatic linking in _compile_self
        program.compiled = True
        return program

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        include_linked: bool = True,
    ) -> "LLMProgram":
        """Load a program from a TOML or YAML file."""
        from llmproc.program import LLMProgram, ProgramRegistry

        # Resolve path and check registry
        path = resolve_path(file_path, must_exist=True, error_prefix="Program file")
        registry = ProgramRegistry()

        if registry.contains(path):
            return registry.get(path)

        # Create and register program
        program = cls._compile_single_file(path)
        registry.register(path, program)

        # Process linked programs if needed
        if include_linked and program.linked_programs:
            cls._process_linked_programs(program, path)

        program.compiled = True
        return program

    @classmethod
    def from_toml(cls, toml_path: Union[str, Path], include_linked: bool = True) -> "LLMProgram":
        """Convenience wrapper around :meth:`from_file` for TOML."""
        return cls.from_file(toml_path, include_linked=include_linked)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], include_linked: bool = True) -> "LLMProgram":
        """Convenience wrapper around :meth:`from_file` for YAML."""
        return cls.from_file(yaml_path, include_linked=include_linked)

    # =========================================================================
    # FILE PROCESSING METHODS
    # =========================================================================

    @classmethod
    def _compile_single_file(cls, path: Path) -> "LLMProgram":
        """Compile a program from a TOML or YAML file."""
        # Detect format from extension
        suffix = path.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            try:
                with path.open("r") as f:
                    config_data = yaml.safe_load(f)
            except Exception as e:
                raise ValueError(f"Error loading YAML file {path}: {str(e)}")
        elif suffix == ".toml":
            try:
                with path.open("rb") as f:
                    config_data = tomllib.load(f)
            except Exception as e:
                raise ValueError(f"Error loading TOML file {path}: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {suffix} (expected .toml, .yaml, or .yml)")

        # Use from_dict to handle the rest of the process
        # Use warn_linked_programs=False to prevent spurious warnings since
        # from_file will handle linked programs properly
        program = cls.from_dict(config_data, base_dir=path.parent, warn_linked_programs=False)
        program.source_path = path
        return program

    @classmethod
    def _process_linked_programs(cls, program: "LLMProgram", path: Path) -> None:
        """Resolve string references in ``linked_programs`` to ``LLMProgram`` objects."""
        from llmproc.program import LLMProgram

        base_dir = path.parent

        for name, program_or_path in list(program.linked_programs.items()):
            if not isinstance(program_or_path, str):
                continue

            try:
                linked_path = resolve_path(
                    program_or_path,
                    base_dir=base_dir,
                    must_exist=True,
                    error_prefix=f"Linked program file (from '{path}')",
                )

                # Use from_file to auto-detect format
                program.linked_programs[name] = cls.from_file(linked_path, include_linked=True)

            except FileNotFoundError as e:
                raise FileNotFoundError(str(e))

    # =========================================================================
    # CONFIGURATION BUILDING METHODS
    # =========================================================================

    @classmethod
    def _build_from_config(cls, config: LLMProgramConfig, base_dir: Path) -> "LLMProgram":
        """Construct an ``LLMProgram`` from a validated config."""
        from llmproc.program import LLMProgram

        # Resolve system prompt
        system_prompt = config.prompt.resolve(base_dir)

        # Process linked programs
        linked_programs, linked_program_descriptions = process_config_linked_programs(config)

        # Get display name with priority: demo > model > default
        # For backwards compatibility, we check both locations
        display_name = None
        if config.demo and config.demo.display_name:
            display_name = config.demo.display_name
        elif hasattr(config.model, "display_name") and config.model.display_name:
            # For backward compatibility with older TOML files
            display_name = config.model.display_name

        # Extract tools from config
        tools_list = config.tools.builtin if config.tools else []

        # Incorporate MCP tool descriptors from [tools.mcp]
        if config.tools and config.tools.mcp:
            tools_list.extend(config.tools.mcp.build_mcp_tools())

        # Create the program instance
        preload_files, preload_relative_to = resolve_preload_files(config, base_dir)
        program = LLMProgram(
            model_name=config.model.name,
            provider=config.model.provider,
            system_prompt=system_prompt,
            parameters=config.parameters,
            display_name=display_name,
            preload_files=preload_files,
            preload_relative_to=preload_relative_to,
            mcp_config_path=resolve_mcp_config(config, base_dir),
            mcp_servers=resolve_mcp_servers(config),
            tools=tools_list,
            linked_programs=linked_programs,
            linked_program_descriptions=linked_program_descriptions,
            env_info=config.env_info or EnvInfoConfig(),
            file_descriptor=config.file_descriptor.model_dump() if config.file_descriptor else None,
            base_dir=base_dir,
            disable_automatic_caching=config.model.disable_automatic_caching,
            project_id=config.model.project_id,
            region=config.model.region,
            user_prompt=config.prompt.user if hasattr(config.prompt, "user") else None,
            max_iterations=config.model.max_iterations,
        )

        return program
