"""LLMProgram compiler for validating and loading LLM program configurations."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union

from llmproc._program_docs import (
    ADD_LINKED_PROGRAM,
    API_PARAMS,
    COMPILE,
    CONFIGURE_MCP,
    CONFIGURE_THINKING,
    ENABLE_TOKEN_EFFICIENT_TOOLS,
    FROM_DICT,
    INIT,
    LLMPROGRAM_CLASS,
    REGISTER_TOOLS,
    START,
    START_SYNC,
)
from llmproc.common.access_control import AccessLevel
from llmproc.config.program_data import ProgramConfigData
from llmproc.config.program_loader import ProgramLoader
from llmproc.config.utils import resolve_path
from llmproc.program_compiler import compile_program
from llmproc.program_config import ProgramConfigMixin
from llmproc.program_registry import ProgramRegistry

# Set up logger
logger = logging.getLogger(__name__)


class LLMProgram(ProgramConfigMixin):
    """Program definition for LLM processes."""

    def __init__(
        self,
        model_name: str,
        provider: str,
        system_prompt: str = None,
        system_prompt_file: str = None,
        parameters: dict[str, Any] = None,
        mcp_config_path: str | None = None,
        mcp_servers: dict[str, dict] | None = None,
        tools: list[Any] = None,
        linked_programs: dict[str, Union[str, "LLMProgram"]] | None = None,
        linked_program_descriptions: dict[str, str] | None = None,
        base_dir: Path | None = None,
        project_id: str | None = None,
        region: str | None = None,
        user_prompt: str = None,
        max_iterations: int = 10,
    ):
        """Initialize a program."""
        # Flag to track if this program has been fully compiled
        self.compiled = False
        self._system_prompt_file = system_prompt_file

        # Handle system prompt (either direct or from file)
        if system_prompt and system_prompt_file:
            raise ValueError("Cannot specify both system_prompt and system_prompt_file")

        # Single source of truth for program configuration
        self.config = ProgramConfigData(
            model_name=model_name,
            provider=provider,
            system_prompt=system_prompt,
            parameters=parameters or {},
            mcp_config_path=mcp_config_path,
            mcp_servers=mcp_servers,
            tools=[],
            plugins=[],
            base_dir=base_dir,
            project_id=project_id,
            region=region,
            user_prompt=user_prompt,
            max_iterations=max_iterations,
        )

        if linked_programs or linked_program_descriptions:
            from llmproc.plugins.spawn import SpawnPlugin

            spawn_plugin = SpawnPlugin(linked_programs or {}, linked_program_descriptions or {})
            self.config.plugins.append(spawn_plugin)

        # Process tools parameter: can include str names, callables, or
        # MCPServerTools descriptors
        if tools:
            # Normalize to list
            raw_tools = tools if isinstance(tools, list) else [tools]

            # Register all tools with the tool manager
            self.register_tools(raw_tools)

    def compile(self) -> "LLMProgram":
        """Validate and compile this program."""
        self.config = compile_program(self.config, self._system_prompt_file)
        self.compiled = True
        return self

    # ------------------------------------------------------------------
    # Attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple forward
        """Forward unknown attribute access to :attr:`config`."""
        cfg = object.__getattribute__(self, "config")
        try:
            value = getattr(cfg, name)
        except AttributeError as exc:  # pragma: no cover - explicit re-raise
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'") from exc

        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Write through to :attr:`config` if the field exists there."""
        if name in {"compiled", "config", "_system_prompt_file", "source_path"}:
            object.__setattr__(self, name, value)
            return
        cfg = self.__dict__.get("config")
        if cfg is not None and hasattr(cfg, name):
            setattr(cfg, name, value)
        else:
            object.__setattr__(self, name, value)

    @property
    def api_params(self) -> dict[str, Any]:
        """Get API parameters for LLM API calls."""
        return self.parameters.copy() if self.parameters else {}

    @classmethod
    def _from_config_data(cls, data: "ProgramConfigData") -> "LLMProgram":
        """Instantiate :class:`LLMProgram` from :class:`ProgramConfigData`."""
        program = cls(
            model_name=data.model_name,
            provider=data.provider,
            system_prompt=data.system_prompt,
            parameters=data.parameters,
            mcp_config_path=data.mcp_config_path,
            mcp_servers=data.mcp_servers,
            tools=None,
            base_dir=data.base_dir,
            project_id=data.project_id,
            region=data.region,
            user_prompt=data.user_prompt,
            max_iterations=data.max_iterations,
        )
        program.compiled = True
        program.config = data
        if program.config.plugins is None:
            program.config.plugins = []
        if data.tools:
            program.register_tools(list(data.tools))

        if data.plugins:
            program.plugins = list(data.plugins)
        return program

    @classmethod
    def _load_from_path(cls, path: Path, loader: Callable[[Path], ProgramConfigData]) -> "LLMProgram":
        """Load a program using ``loader`` and cache it via :class:`ProgramRegistry`."""
        registry = ProgramRegistry()
        if registry.contains(path):
            return registry.get(path)

        data = loader(path)
        program = cls._from_config_data(data)
        program.source_path = path
        registry.register(path, program)
        return program

    @classmethod
    def from_toml(cls, toml_file, include_linked: bool = True):
        """Create a program from a TOML file."""
        return cls.from_file(toml_file, format="toml")

    @classmethod
    def from_yaml(cls, yaml_file, include_linked: bool = True):
        """Create a program from a YAML file."""
        return cls.from_file(yaml_file, format="yaml")

    @classmethod
    def from_file(cls, file_path, *, format: str = "auto", include_linked: bool = True):
        """Create a program from a configuration file.

        Args:
            file_path: Path to the configuration file.
            format: ``"toml"``, ``"yaml"``, or ``"auto"`` to infer from extension.
            include_linked: Whether to load linked programs recursively.
        """
        path = resolve_path(file_path, must_exist=True, error_prefix="Program file")
        return cls._load_from_path(
            path,
            lambda p: ProgramLoader.from_file(p, format=format),
        )

    @classmethod
    def from_dict(cls, config: dict, base_dir: str | Path = None) -> "LLMProgram":
        data = ProgramLoader.from_dict(config, base_dir)
        return cls._from_config_data(data)

    def get_tool_configuration(self, linked_programs_instances: dict[str, Any] | None = None) -> dict:
        """Build the configuration used to initialize tools."""
        # Ensure the program is compiled
        if not self.compiled:
            self.compile()

        # Extract core configuration properties
        config = {
            "provider": self.provider,
            "mcp_config_path": getattr(self, "mcp_config_path", None),
            "mcp_servers": getattr(self, "mcp_servers", None),
            "mcp_enabled": (
                getattr(self, "mcp_config_path", None) is not None or getattr(self, "mcp_servers", None) is not None
            ),
        }

        # Provide linked program info from SpawnPlugin if available
        from llmproc.plugins.spawn import SpawnPlugin

        if linked_programs_instances is not None:
            config["linked_programs"] = linked_programs_instances
            config["has_linked_programs"] = bool(linked_programs_instances)
            for plugin in getattr(self, "plugins", []):
                if isinstance(plugin, SpawnPlugin):
                    config["linked_program_descriptions"] = plugin.linked_program_descriptions
                    break
        else:
            for plugin in getattr(self, "plugins", []):
                if isinstance(plugin, SpawnPlugin):
                    config["linked_programs"] = plugin.linked_programs
                    config["linked_program_descriptions"] = plugin.linked_program_descriptions
                    config["has_linked_programs"] = bool(plugin.linked_programs)
                    break
        if "linked_programs" not in config:
            config["linked_programs"] = {}
            config["linked_program_descriptions"] = {}
            config["has_linked_programs"] = False

        # File descriptor configuration is now handled by FileDescriptorPlugin

        # Add tools configuration including provider-specific tools (e.g., anthropic.web_search)
        if hasattr(self, "config") and hasattr(self.config, "tools_config") and self.config.tools_config:
            # Convert the ToolsConfig object to dict to get all official fields
            tools_dict = self.config.tools_config.model_dump()
            if tools_dict:
                config["tools"] = tools_dict

        logger.info("Created tool configuration for initialization")
        return config

    async def start(self, access_level: Optional[AccessLevel] = None) -> "LLMProcess":  # noqa: F821
        # Delegate to the modular implementation in program_exec.py
        from llmproc.program_exec import create_process

        return await create_process(self, access_level=access_level)

    def start_sync(self, access_level: Optional[AccessLevel] = None) -> "SyncLLMProcess":  # noqa: F821
        # Import here to avoid circular imports
        from llmproc.program_exec import create_sync_process

        # Delegate to the modular implementation in program_exec.py
        return create_sync_process(self, access_level=access_level)


# Apply full docstrings to class and methods
LLMProgram.__doc__ = LLMPROGRAM_CLASS
LLMProgram.__init__.__doc__ = INIT
LLMProgram.add_linked_program.__doc__ = ADD_LINKED_PROGRAM
LLMProgram.configure_thinking.__doc__ = CONFIGURE_THINKING
LLMProgram.enable_token_efficient_tools.__doc__ = ENABLE_TOKEN_EFFICIENT_TOOLS
LLMProgram.register_tools.__doc__ = REGISTER_TOOLS
LLMProgram.configure_mcp.__doc__ = CONFIGURE_MCP
LLMProgram.compile.__doc__ = COMPILE
LLMProgram.api_params.__doc__ = API_PARAMS
LLMProgram.from_dict.__func__.__doc__ = FROM_DICT
LLMProgram.start.__doc__ = START
LLMProgram.start_sync.__doc__ = START_SYNC
