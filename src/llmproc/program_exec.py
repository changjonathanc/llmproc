"""Program execution module for program-to-process transitions.

This module contains modular functions for transforming LLMProgram configurations
into LLMProcess instances, with each step isolated for better testing and maintenance.
"""

import copy
import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union
from llmproc.env_info.builder import EnvInfoBuilder
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.providers import get_provider_client


class ProcessInitializationError(ValueError):
    """Custom exception for errors during LLMProcess initialization."""

    pass

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Configuration Return Types
# --------------------------------------------------------
# NamedTuples for initialization functions to provide structured returns
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


def instantiate_process(process_state: dict[str, Any]) -> LLMProcess:
    """Create bare process instance from pre-computed state using introspection.

    Args:
        process_state: Dictionary containing all required state for LLMProcess initialization

    Returns:
        Instantiated LLMProcess with pre-computed state

    Raises:
        ProcessInitializationError: If required parameters are missing or None
        TypeError: If unexpected parameters are passed and __init__ doesn't accept **kwargs
    """
    logger.debug(
        f"Instantiating LLMProcess with state dictionary containing {len(process_state)} entries"
    )

    # Use introspection to determine valid parameters for LLMProcess.__init__
    # This approach is used deliberately to make instantiate_process automatically
    # adapt to changes in LLMProcess.__init__ signature without requiring manual updates
    try:
        init_signature = inspect.signature(LLMProcess.__init__)
        init_params = init_signature.parameters
    except ValueError:
        logger.error("Could not retrieve signature for LLMProcess.__init__")
        raise ProcessInitializationError("Failed to inspect LLMProcess constructor")

    # Identify required parameters (excluding 'self')
    required_params = {
        name
        for name, param in init_params.items()
        if param.default is inspect.Parameter.empty and name != "self"
    }

    # --- Validation ---
    # Check for missing required parameters
    missing_required = required_params - process_state.keys()
    if missing_required:
        raise ProcessInitializationError(
            f"Missing required parameters for LLMProcess: {', '.join(sorted(missing_required))}. State provided keys: {', '.join(sorted(process_state.keys()))}"
        )

    # Check if any required parameters have None values
    none_required_values = {
        k for k in required_params if k in process_state and process_state[k] is None
    }
    if none_required_values:
        raise ProcessInitializationError(
            f"Required parameters cannot be None: {', '.join(sorted(none_required_values))}"
        )

    # --- Filtering ---
    # Filter the state to include only parameters accepted by __init__
    valid_param_names = {name for name in init_params if name != "self"}
    init_kwargs = {k: v for k, v in process_state.items() if k in valid_param_names}

    logger.debug(
        f"Filtered state for LLMProcess instantiation includes keys: {', '.join(sorted(init_kwargs.keys()))}"
    )

    # Instantiate the process
    try:
        return LLMProcess(**init_kwargs)
    except TypeError as e:
        logger.error(f"TypeError during LLMProcess instantiation: {e}")
        raise ProcessInitializationError(
            f"Failed to instantiate LLMProcess: {e}"
        ) from e


def setup_runtime_context(
    process: LLMProcess, runtime_dependencies: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Set up runtime context for dependency injection.

    Args:
        process: The LLMProcess to set up runtime context for
        runtime_dependencies: Optional pre-configured dependencies (useful for testing)

    Returns:
        The configured runtime context dictionary
    """
    # Use provided dependencies or extract from process
    # We use direct attribute access for required properties and
    # fallback defaults for optional ones for better explicitness
    if runtime_dependencies is not None:
        context = runtime_dependencies
    else:
        context = {
            "process": process,
            "fd_manager": process.fd_manager
            if hasattr(process, "fd_manager")
            else None,
            "linked_programs": process.linked_programs,
            "linked_program_descriptions": process.linked_program_descriptions,
        }

    # Apply context to the process's tool manager
    if process.tool_manager:
        process.tool_manager.set_runtime_context(context)

    return context


def validate_process(process: LLMProcess) -> None:
    """Perform final validation and logging.

    Args:
        process: The LLMProcess to validate
    """
    logger.info(f"Created process with model {process.model_name} ({process.provider})")
    logger.info(f"Tools enabled: {len(process.tool_manager.get_enabled_tools())}")


async def create_process(
    program: LLMProgram, additional_preload_files: Optional[list[str]] = None
) -> LLMProcess:
    """Create fully initialized process from program.

    This function handles the complete program-to-process transition by
    coordinating all the individual steps in sequence using a functional approach:

    1. Ensure program is compiled (done once at the beginning)
    2. Prepare all initial state using pure initialization functions
    3. Extract configuration from program
    4. Initialize tools with extracted configuration
    5. Create process instance with pre-computed state
    6. Set up runtime context for dependency injection
    7. Perform final validation

    Args:
        program: The LLMProgram to create a process from
        additional_preload_files: Optional list of additional files to preload

    Returns:
        A fully initialized LLMProcess
    """
    logger.info(
        f"Starting process creation for program: {getattr(program, 'display_name', program.model_name)}"
    )

    # 1. Ensure program is compiled (only done once here)
    if not program.compiled:
        logger.debug("Compiling program...")
        program.compile()
        logger.debug("Program compiled successfully")

    # 2. Prepare all initial state using pure initialization functions
    # Pass additional_preload_files to be included in enriched system prompt
    logger.debug("Preparing process state...")
    process_state = prepare_process_state(program, additional_preload_files)
    logger.debug(f"Process state prepared with {len(process_state)} entries")

    # 3. Extract configuration from program (program is already compiled)
    logger.debug("Extracting tool configuration...")
    config = program.get_tool_configuration()
    logger.debug("Tool configuration extracted")

    # 4. Initialize tools with configuration
    logger.debug("Initializing tools...")
    await program.tool_manager.initialize_tools(config)
    logger.debug("Tools initialized successfully")

    # 5. Create process instance with pre-computed state
    logger.debug("Instantiating LLMProcess...")
    process = instantiate_process(process_state)
    logger.debug(f"LLMProcess instantiated with ID: {id(process)}")

    # 6. Set up runtime context for tools
    logger.debug("Setting up runtime context...")
    setup_runtime_context(process)
    logger.debug("Runtime context set up successfully")

    # 7. Perform final validation
    logger.debug("Validating process...")
    validate_process(process)
    logger.debug("Process validated successfully")

    logger.info(
        f"Process created successfully for {process.model_name} ({process.provider})"
    )

    return process


# --------------------------------------------------------
# System Prompt Generation
# --------------------------------------------------------
# Function to generate the enriched system prompt at initialization time


# --------------------------------------------------------
# Pure Initialization Functions
# --------------------------------------------------------
# These functions extract configuration from a program without side effects


def initialize_file_descriptor_system(
    program: LLMProgram,
) -> FileDescriptorSystemConfig:
    """Initialize file descriptor subsystem based on program configuration.

    This is a pure function that takes program configuration and returns the
    file descriptor system configuration without modifying any state.

    Args:
        program: The LLMProgram containing file descriptor configuration

    Returns:
        FileDescriptorSystemConfig containing the manager and configuration flags
    """
    fd_manager = None
    file_descriptor_enabled = False
    references_enabled = False

    if hasattr(program, "file_descriptor"):
        fd_config = program.file_descriptor
        enabled = fd_config.get("enabled", False)

        if enabled:
            # Create file descriptor manager with configuration
            references_enabled = fd_config.get("enable_references", False)
            fd_manager = FileDescriptorManager(
                default_page_size=fd_config.get("default_page_size", 4000),
                max_direct_output_chars=fd_config.get("max_direct_output_chars", 8000),
                max_input_chars=fd_config.get("max_input_chars", 8000),
                page_user_input=fd_config.get("page_user_input", True),
                enable_references=references_enabled,
            )
            file_descriptor_enabled = True
            logger.info(
                f"File descriptor enabled: page_size={fd_manager.default_page_size}, references={references_enabled}"
            )

    return FileDescriptorSystemConfig(
        fd_manager=fd_manager,
        file_descriptor_enabled=file_descriptor_enabled,
        references_enabled=references_enabled,
    )


def extract_linked_programs_config(program: LLMProgram) -> LinkedProgramsConfig:
    """Extract linked programs configuration from program.

    This is a pure function that extracts the linked programs configuration
    from the program without modifying any state.

    Args:
        program: The LLMProgram containing linked programs configuration

    Returns:
        LinkedProgramsConfig with linked program references and descriptions
    """
    # Get linked programs from the program
    linked_programs = getattr(program, "linked_programs", {})
    linked_program_descriptions = getattr(program, "linked_program_descriptions", {})
    has_linked_programs = bool(linked_programs)

    logger.debug(f"Extracted {len(linked_programs)} linked program references")

    return LinkedProgramsConfig(
        linked_programs=linked_programs,
        linked_program_descriptions=linked_program_descriptions,
        has_linked_programs=has_linked_programs,
    )


def initialize_client(program: LLMProgram) -> Any:
    """Initialize provider client based on program configuration.

    This is a pure function that initializes a provider client without
    modifying any state.

    Args:
        program: The LLMProgram containing provider configuration

    Returns:
        Initialized provider client
    """
    project_id = getattr(program, "project_id", None)
    region = getattr(program, "region", None)

    client = get_provider_client(
        program.provider, program.model_name, project_id, region
    )
    logger.debug(f"Initialized client for provider {program.provider}")

    return client


def get_core_attributes(program: LLMProgram) -> dict[str, Any]:
    """Extract core attributes from program.

    This is a pure function that extracts basic attributes from the program
    without modifying any state.

    Args:
        program: The LLMProgram to extract attributes from

    Returns:
        Dictionary of core attributes
    """
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
    }


def _initialize_runtime_defaults(original_prompt: str) -> dict[str, Any]:
    """Initialize default runtime state values.

    Args:
        original_prompt: Original system prompt to use as default

    Returns:
        Dictionary of default runtime state values
    """
    return {
        "state": [],  # Empty conversation history
        "enriched_system_prompt": None,  # Generated on first run
        "allow_fork": True,  # Allow forking by default
        "system_prompt": original_prompt,  # For backward compatibility
    }


def _initialize_mcp_config(program: LLMProgram) -> dict[str, Any]:
    """Extract MCP configuration from the program.

    Args:
        program: The LLMProgram containing MCP configuration

    Returns:
        Dictionary of MCP configuration
    """
    mcp_config_path = getattr(program, "mcp_config_path", None)
    return {
        "mcp_config_path": mcp_config_path,
        "mcp_tools": getattr(program, "mcp_tools", {}),
        "mcp_enabled": mcp_config_path is not None,
    }


def prepare_process_state(
    program: LLMProgram, additional_preload_files: Optional[list[str]] = None
) -> dict[str, Any]:
    """Prepare the complete initial state for LLMProcess.

    This function aggregates the results of all initialization functions
    into a single dictionary that can be used to initialize an LLMProcess.
    It follows a process of incrementally building up the state from various
    sources and pure initialization functions.

    Args:
        program: The LLMProgram to prepare state for
        additional_preload_files: Optional list of additional file paths to preload

    Returns:
        Dictionary containing all state needed to initialize an LLMProcess
    """
    state = {}

    # --- Program Reference (needed for core operations) ---
    state["program"] = program

    # --- Core Model Attributes ---
    # Extract basic information about the model, provider, prompts, etc.
    core_attrs = get_core_attributes(program)
    state.update(core_attrs)

    # --- Initial Runtime State ---
    # Initialize empty conversation state and runtime configuration
    state.update(_initialize_runtime_defaults(core_attrs["original_system_prompt"]))

    # --- API Client Initialization ---
    # Create the provider-specific client for API calls
    state["client"] = initialize_client(program)

    # --- File Descriptor System ---
    # Initialize the file descriptor subsystem if enabled
    fd_info = initialize_file_descriptor_system(program)
    state.update(fd_info._asdict())

    # --- Linked Programs ---
    # Extract linked program configuration for spawn tool usage
    linked_info = extract_linked_programs_config(program)
    state.update(linked_info._asdict())

    # --- Preloaded Content Collection ---
    # Collect file paths from program and additional_preload_files
    preload_files = getattr(program, "preload_files", []).copy()
    if additional_preload_files:
        preload_files.extend(additional_preload_files)

    # We don't store preloaded content anymore
    # The effect of preloading is fully captured in the enriched_system_prompt

    # --- MCP Configuration ---
    # Set up Model Context Protocol configuration for external tools
    state.update(_initialize_mcp_config(program))

    # --- Generate Enriched System Prompt ---
    # Now that all necessary components are available, generate the enriched system prompt
    # This includes environment info, preloaded file content, and FD instructions
    # Generate it once at initialization time, making it immutable during process execution
    env_config = getattr(program, "env_info", {"variables": []})
    page_user_input = (
        getattr(fd_info.fd_manager, "page_user_input", False)
        if fd_info.fd_manager
        else False
    )

    # Generate the enriched system prompt directly
    # EnvInfoBuilder now handles all file loading
    logger.debug("Generating enriched system prompt with file preloading")
    state["enriched_system_prompt"] = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=state["original_system_prompt"],
        env_config=env_config or {"variables": []},
        preload_files=preload_files,
        base_dir=program.base_dir,
        include_env=True,
        file_descriptor_enabled=state["file_descriptor_enabled"],
        references_enabled=state["references_enabled"],
        page_user_input=page_user_input,
    )

    logger.debug("Enriched system prompt generated")

    logger.debug(f"Prepared process state dictionary with {len(state)} entries")
    return state
