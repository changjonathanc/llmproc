"""Program execution module for program-to-process transitions.

This module contains modular functions for transforming LLMProgram configurations
into LLMProcess instances, with each step isolated for better testing and maintenance.
"""

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypedDict, Union

from llmproc.common.access_control import AccessLevel
from llmproc.common.context import RuntimeContext
from llmproc.config import EnvInfoConfig
from llmproc.env_info.builder import EnvInfoBuilder
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess, SyncLLMProcess
from llmproc.program import LLMProgram
from llmproc.providers import get_provider_client
from llmproc.tools import ToolManager


class ProcessInitializationError(ValueError):
    """Custom exception for errors during LLMProcess initialization."""

    pass


logger = logging.getLogger(__name__)

# Parameters used only during initialization and removed before process creation
INITIALIZATION_ONLY_PARAMS = ("project_id", "region")


def _remove_init_params(params: dict[str, Any]) -> None:
    """Remove initialization-only parameters from the given mapping."""
    for param in INITIALIZATION_ONLY_PARAMS:
        if param in params:
            params.pop(param)


# --------------------------------------------------------
# Configuration Return Types & Helper Data Structures
# --------------------------------------------------------
class FileDescriptorSystemConfig(NamedTuple):
    """Configuration for the file descriptor system."""

    fd_manager: Optional[FileDescriptorManager]
    file_descriptor_enabled: bool
    references_enabled: bool


class LinkedProgramsConfig(NamedTuple):
    """Configuration for linked programs."""

    linked_programs: dict[str, LLMProgram]
    linked_program_descriptions: dict[str, str]
    has_linked_programs: bool


class CoreAttributes(TypedDict):
    """Core attributes extracted from an LLMProgram."""

    model_name: str
    provider: str
    original_system_prompt: Optional[str]
    display_name: Optional[str]
    base_dir: Optional[Path]
    api_params: dict[str, Any]
    tool_manager: ToolManager
    project_id: Optional[str]
    region: Optional[str]
    user_prompt: Optional[str]
    max_iterations: int


class RuntimeDefaults(TypedDict):
    """Default runtime state values for a new process."""

    state: list[dict[str, Any]]  # Empty conversation history
    enriched_system_prompt: Optional[str]  # Generated on first run
    system_prompt: Optional[str]  # For backward compatibility, usually original_prompt


class MCPConfig(TypedDict):
    """MCP (Model Context Protocol) configuration."""

    mcp_config_path: Optional[str]
    mcp_servers: Optional[dict]
    mcp_tools: dict[str, Any]
    mcp_enabled: bool


# --------------------------------------------------------
# Pure Initialization Functions
# --------------------------------------------------------
# These functions extract configuration from a program without side effects


def initialize_file_descriptor_system(
    program: LLMProgram,
) -> FileDescriptorSystemConfig:
    """Initialize file descriptor subsystem based on program configuration."""
    fd_config = getattr(program, "file_descriptor", {})
    if not fd_config.get("enabled", False):
        return FileDescriptorSystemConfig(None, False, False)

    references_enabled = fd_config.get("enable_references", False)
    fd_manager = FileDescriptorManager(
        default_page_size=fd_config.get("default_page_size", 4000),
        max_direct_output_chars=fd_config.get("max_direct_output_chars", 8000),
        max_input_chars=fd_config.get("max_input_chars", 8000),
        page_user_input=fd_config.get("page_user_input", True),
        enable_references=references_enabled,
    )
    logger.info(
        "File descriptor enabled: page_size=%s, references=%s",
        fd_manager.default_page_size,
        references_enabled,
    )

    if hasattr(program, "tools") and program.tools:
        enabled_tools = program.tools.get("enabled", []) if isinstance(program.tools, dict) else program.tools
        for tool_name in enabled_tools:
            if isinstance(tool_name, str) and tool_name in ("read_fd", "fd_to_file"):
                fd_manager.register_fd_tool(tool_name)

    return FileDescriptorSystemConfig(
        fd_manager=fd_manager,
        file_descriptor_enabled=True,
        references_enabled=references_enabled,
    )


def extract_linked_programs_config(program: LLMProgram) -> LinkedProgramsConfig:
    """Extract linked programs configuration from program."""
    linked_programs = getattr(program, "linked_programs", {})
    linked_program_descriptions = getattr(program, "linked_program_descriptions", {})
    has_linked_programs = bool(linked_programs)

    return LinkedProgramsConfig(
        linked_programs=linked_programs,
        linked_program_descriptions=linked_program_descriptions,
        has_linked_programs=has_linked_programs,
    )


def initialize_client(program: LLMProgram) -> Any:
    """Initialize provider client based on program configuration."""
    project_id = getattr(program, "project_id", None)
    region = getattr(program, "region", None)
    client = get_provider_client(program.provider, program.model_name, project_id, region)
    return client


def get_core_attributes(program: LLMProgram) -> CoreAttributes:
    """Extract core attributes from program."""
    return {
        "model_name": program.model_name,
        "provider": program.provider,
        "original_system_prompt": program.system_prompt,
        "display_name": program.display_name,
        "base_dir": program.base_dir,
        "api_params": program.api_params,
        "tool_manager": program.tool_manager,
        "project_id": getattr(program, "project_id", None),
        "region": getattr(program, "region", None),
        "user_prompt": getattr(program, "user_prompt", None),
        "max_iterations": getattr(program, "max_iterations", 10),
    }


def _initialize_runtime_defaults(original_prompt: Optional[str]) -> RuntimeDefaults:
    """Initialize default runtime state values."""
    return {
        "state": [],
        "enriched_system_prompt": None,
        "system_prompt": original_prompt,
    }


def _initialize_mcp_config(program: LLMProgram) -> MCPConfig:
    """Extract MCP configuration from the program."""
    mcp_config_path = getattr(program, "mcp_config_path", None)
    mcp_servers = getattr(program, "mcp_servers", None)
    return {
        "mcp_config_path": mcp_config_path,
        "mcp_servers": mcp_servers,
        "mcp_tools": getattr(program, "mcp_tools", {}),
        "mcp_enabled": (mcp_config_path is not None or mcp_servers is not None),
    }


# --------------------------------------------------------
# Process State Preparation
# --------------------------------------------------------
def prepare_process_state(
    program: LLMProgram,
    additional_preload_files: Optional[list[str]] = None,
    access_level: Optional[AccessLevel] = None,
) -> dict[str, Any]:
    """Prepare the complete initial state for LLMProcess."""
    state = {}

    state["program"] = program
    core_attrs = get_core_attributes(program)
    state.update(core_attrs)
    state.update(_initialize_runtime_defaults(core_attrs["original_system_prompt"]))
    state["client"] = initialize_client(program)
    fd_info = initialize_file_descriptor_system(program)
    state.update(fd_info._asdict())
    linked_info = extract_linked_programs_config(program)
    state.update(linked_info._asdict())

    preload_files = getattr(program, "preload_files", []).copy()
    if additional_preload_files:
        preload_files.extend(additional_preload_files)

    state.update(_initialize_mcp_config(program))
    state["access_level"] = access_level or AccessLevel.ADMIN

    env_config = getattr(program, "env_info", EnvInfoConfig())
    page_user_input = getattr(fd_info.fd_manager, "page_user_input", False) if fd_info.fd_manager else False

    preload_base = program.base_dir
    if getattr(program, "preload_relative_to", "program") == "cwd":
        preload_base = Path.cwd()

    state["enriched_system_prompt"] = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=state["original_system_prompt"],
        env_config=env_config,
        preload_files=preload_files,
        base_dir=preload_base,
        include_env=True,
        file_descriptor_enabled=state["file_descriptor_enabled"],
        references_enabled=state["references_enabled"],
        page_user_input=page_user_input,
    )
    return state


# --------------------------------------------------------
# Core Process Instantiation and Setup
# --------------------------------------------------------
def instantiate_process(process_state: dict[str, Any]) -> LLMProcess:
    """Create bare process instance from pre-computed state using introspection."""
    try:
        init_signature = inspect.signature(LLMProcess.__init__)
        init_params = init_signature.parameters
    except ValueError:
        logger.error("Could not retrieve signature for LLMProcess.__init__")
        raise ProcessInitializationError("Failed to inspect LLMProcess constructor")

    valid_param_names = {name for name in init_params if name != "self"}
    required_params = {
        name for name, param in init_params.items() if param.default is inspect.Parameter.empty and name != "self"
    }

    init_kwargs = {k: v for k, v in process_state.items() if k in valid_param_names}
    _remove_init_params(init_kwargs)

    missing_required_final = {p for p in required_params if p not in init_kwargs}
    if missing_required_final:
        sorted_missing = sorted(list(missing_required_final))
        sorted_provided_valid = sorted(list(init_kwargs.keys()))
        raise ProcessInitializationError(
            f"Missing required parameters for LLMProcess after filtering: {', '.join(sorted_missing)}. "
            f"Provided valid __init__ args: {', '.join(sorted_provided_valid)}"
        )

    none_values_in_required = {k for k in required_params if k in init_kwargs and init_kwargs[k] is None}
    if none_values_in_required:
        sorted_none_required = sorted(list(none_values_in_required))
        raise ProcessInitializationError(
            f"Required parameters for LLMProcess cannot be None: {', '.join(sorted_none_required)}"
        )

    try:
        init_kwargs.setdefault("loop", asyncio.get_running_loop())
    except RuntimeError:
        pass

    try:
        return LLMProcess(**init_kwargs)
    except TypeError as e:
        logger.error(f"TypeError during LLMProcess instantiation: {e}")
        raise ProcessInitializationError(f"Failed to instantiate LLMProcess: {e}") from e


def setup_runtime_context(
    process: LLMProcess, runtime_dependencies: Optional[dict[str, Any]] = None
) -> "RuntimeContext":
    """Set up runtime context for dependency injection."""
    if runtime_dependencies is not None:
        context = runtime_dependencies
    else:
        context: RuntimeContext = {"process": process}
        context["stderr"] = getattr(process, "stderr_log", [])
        if hasattr(process, "fd_manager"):
            context["fd_manager"] = process.fd_manager
        if hasattr(process, "linked_programs") and process.linked_programs:
            context["linked_programs"] = process.linked_programs
        if hasattr(process, "linked_program_descriptions") and process.linked_program_descriptions:
            context["linked_program_descriptions"] = process.linked_program_descriptions

    if process.tool_manager:
        process.tool_manager.set_runtime_context(context)
        if hasattr(process, "access_level"):
            process.tool_manager.set_process_access_level(process.access_level)
    else:
        logger.warning("Cannot set runtime context - process.tool_manager is None!")
    return context


def validate_process(process: LLMProcess) -> None:
    """Perform final validation and logging."""
    logger.info(f"Created process with model {process.model_name} ({process.provider})")
    logger.info(f"Tools enabled: {len(process.tool_manager.get_registered_tools())}")


# --------------------------------------------------------
# Generic Process Creation Logic
# --------------------------------------------------------
async def _create_process_generic(
    program: LLMProgram,
    process_class: type = LLMProcess,
    process_kwargs: Optional[dict[str, Any]] = None,
    additional_preload_files: Optional[list[str]] = None,
    access_level: Optional[AccessLevel] = None,
) -> Union[LLMProcess, SyncLLMProcess]:
    """Generic process creation for both async and sync modes."""
    display_name = getattr(program, "display_name", program.model_name)
    process_type = process_class.__name__
    logger.info(f"Starting {process_type} creation for program: {display_name}")

    if not program.compiled:
        program.compile()

    process_state = prepare_process_state(program, additional_preload_files, access_level)

    if process_kwargs:
        process_state.update(process_kwargs)

    config = program.get_tool_configuration()
    await program.tool_manager.initialize_tools(config)

    if process_class is LLMProcess:
        process = instantiate_process(process_state)
    else:
        init_kwargs_for_sync = process_state.copy()
        _remove_init_params(init_kwargs_for_sync)
        process = process_class(**init_kwargs_for_sync)

    setup_runtime_context(process)
    validate_process(process)

    logger.info(f"{process_type} created successfully for {process.model_name} ({process.provider})")
    return process


# --------------------------------------------------------
# Public Factory Functions
# --------------------------------------------------------
async def create_process(
    program: LLMProgram, additional_preload_files: Optional[list[str]] = None, access_level: Optional[Any] = None
) -> LLMProcess:
    """Create fully initialized async LLMProcess from program."""
    return await _create_process_generic(
        program=program,
        process_class=LLMProcess,
        additional_preload_files=additional_preload_files,
        access_level=access_level,
    )


def create_sync_process(
    program: LLMProgram,
    additional_preload_files: Optional[list[str]] = None,
    access_level: Optional[AccessLevel] = None,
) -> SyncLLMProcess:
    """Create a fully initialized SyncLLMProcess for synchronous API usage."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _create_process_generic(
                program=program,
                process_class=SyncLLMProcess,
                process_kwargs={"_loop": loop},
                additional_preload_files=additional_preload_files,
                access_level=access_level,
            )
        )
    except Exception as e:
        if not loop.is_closed():
            loop.close()
        raise e
