"""Program loader for loading LLMProgram configurations from various sources."""

import tomllib
import warnings
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import ValidationError

from llmproc.config.schema import LLMProgramConfig
from llmproc.config.utils import resolve_path


class ProgramLoader:
    """Loads LLMProgram configurations from various sources.

    This class handles loading and parsing program configurations from TOML files,
    separating configuration concerns from the LLMProgram class itself.
    """

    @classmethod
    def from_toml(cls, toml_path: str | Path, include_linked: bool = True) -> "LLMProgram":
        """Load and compile a program from a TOML file.

        Args:
            toml_path: Path to the TOML file
            include_linked: Whether to include linked programs

        Returns:
            An initialized LLMProgram instance
        """
        from llmproc.program import LLMProgram, ProgramRegistry

        # Resolve path and check registry
        path = resolve_path(toml_path, must_exist=True, error_prefix="Program file")
        registry = ProgramRegistry()

        if registry.contains(path):
            return registry.get(path)

        # Create and register program
        program = cls._compile_single_program(path)
        registry.register(path, program)

        # Process linked programs if needed
        if include_linked and program.linked_programs:
            cls._process_toml_linked_programs(program, path)

        program.compiled = True
        return program

    @classmethod
    def _compile_single_program(cls, path: Path) -> "LLMProgram":
        """Compile a single program without recursively compiling linked programs."""
        from llmproc.program import LLMProgram

        # Load and validate the TOML file
        try:
            with path.open("rb") as f:
                config = LLMProgramConfig(**tomllib.load(f))
        except ValidationError as e:
            raise ValueError(f"Invalid program configuration in {path}:\n{str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading TOML file {path}: {str(e)}")

        # Build and return the program with source path
        program = cls._build_from_config(config, path.parent)
        program.source_path = path
        return program

    @classmethod
    def _build_from_config(cls, config: LLMProgramConfig, base_dir: Path) -> "LLMProgram":
        """Build an LLMProgram from a validated configuration."""
        from llmproc.program import LLMProgram

        # Resolve system prompt
        system_prompt = config.prompt.resolve(base_dir)

        # Process linked programs
        linked_programs, linked_program_descriptions = cls._process_config_linked_programs(config)

        # Create and return the program instance
        return LLMProgram(
            model_name=config.model.name,
            provider=config.model.provider,
            system_prompt=system_prompt,
            parameters=config.parameters,
            display_name=config.model.display_name,
            preload_files=cls._resolve_preload_files(config, base_dir),
            mcp_config_path=cls._resolve_mcp_config(config, base_dir),
            mcp_tools=config.mcp.tools.root if config.mcp and config.mcp.tools else None,
            tools=config.tools.model_dump() if config.tools else None,
            linked_programs=linked_programs,
            linked_program_descriptions=linked_program_descriptions,
            env_info=config.env_info.model_dump() if config.env_info else {"variables": []},
            file_descriptor=config.file_descriptor.model_dump() if config.file_descriptor else None,
            base_dir=base_dir,
            disable_automatic_caching=config.model.disable_automatic_caching,
            project_id=config.model.project_id,
            region=config.model.region,
        )

    @classmethod
    def _resolve_preload_files(cls, config: LLMProgramConfig, base_dir: Path) -> list[str]:
        """Resolve preload file paths from configuration."""
        if not config.preload or not config.preload.files:
            return None

        preload_files = []
        for file_path in config.preload.files:
            try:
                resolved_path = resolve_path(file_path, base_dir, must_exist=False)
                if not resolved_path.exists():
                    warnings.warn(f"Preload file not found: {resolved_path}", stacklevel=2)
                preload_files.append(str(resolved_path))
            except Exception as e:
                warnings.warn(f"Error resolving path '{file_path}': {str(e)}", stacklevel=2)
        return preload_files

    @classmethod
    def _resolve_mcp_config(cls, config: LLMProgramConfig, base_dir: Path) -> str:
        """Resolve MCP configuration path."""
        if not config.mcp or not config.mcp.config_path:
            return None

        try:
            return str(resolve_path(config.mcp.config_path, base_dir, must_exist=True, error_prefix="MCP config file"))
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))

    @classmethod
    def _process_config_linked_programs(cls, config: LLMProgramConfig) -> tuple:
        """Process linked programs from configuration."""
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

    @classmethod
    def _process_toml_linked_programs(cls, program: "LLMProgram", path: Path) -> None:
        """Process linked programs in a TOML-loaded program."""
        from llmproc.program import LLMProgram

        base_dir = path.parent

        for name, program_or_path in list(program.linked_programs.items()):
            if not isinstance(program_or_path, str):
                continue

            try:
                linked_path = resolve_path(program_or_path, base_dir=base_dir, must_exist=True, error_prefix=f"Linked program file (from '{path}')")
                program.linked_programs[name] = LLMProgram.from_toml(linked_path, include_linked=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(str(e))
