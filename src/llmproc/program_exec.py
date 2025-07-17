"""Program execution module for program-to-process transitions.

This module contains modular functions for transforming LLMProgram configurations
into LLMProcess instances, with each step isolated for better testing and maintenance.
"""

import asyncio
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypedDict, Union

from llmproc.common.access_control import AccessLevel
from llmproc.common.context import RuntimeContext
from llmproc.config.process_config import ProcessConfig
from llmproc.llm_process import LLMProcess, SyncLLMProcess
from llmproc.program import LLMProgram
from llmproc.providers import get_provider_client
from llmproc.tools import ToolManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Configuration Return Types & Helper Data Structures
# --------------------------------------------------------


class CoreAttributes(TypedDict):
    """Core attributes extracted from an LLMProgram."""

    model_name: str
    provider: str
    base_system_prompt: Optional[str]
    base_dir: Optional[Path]
    api_params: dict[str, Any]
    project_id: Optional[str]
    region: Optional[str]
    user_prompt: Optional[str]
    max_iterations: int


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


def get_core_attributes(program: LLMProgram) -> CoreAttributes:
    """Extract core attributes from program."""
    return {
        "model_name": program.model_name,
        "provider": program.provider,
        "base_system_prompt": program.system_prompt,
        "base_dir": program.base_dir,
        "api_params": program.api_params,
        "project_id": program.project_id,
        "region": program.region,
        "user_prompt": getattr(program, "user_prompt", None),
        "max_iterations": getattr(program, "max_iterations", 10),
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
    access_level: Optional[AccessLevel] = None,
) -> dict[str, Any]:
    """Prepare the complete initial state for LLMProcess."""
    state = {}

    state["program"] = program
    core_attrs = get_core_attributes(program)
    state.update(core_attrs)
    state["state"] = []
    state["enriched_system_prompt"] = None
    state["client"] = get_provider_client(
        program.provider,
        project_id=program.project_id,
        region=program.region,
    )
    state.update(_initialize_mcp_config(program))
    state["access_level"] = access_level or AccessLevel.ADMIN
    state["enriched_system_prompt"] = state["base_system_prompt"]
    state["plugins"] = list(getattr(program, "plugins", []))
    return state


def prepare_process_config(
    program: LLMProgram,
    access_level: Optional[AccessLevel] = None,
) -> ProcessConfig:
    """Return a fully populated ``ProcessConfig`` for ``LLMProcess``."""
    state = prepare_process_state(program, access_level)
    cfg_fields = {f.name for f in fields(ProcessConfig)}
    cfg_kwargs = {k: v for k, v in state.items() if k in cfg_fields}
    try:
        cfg_kwargs.setdefault("loop", asyncio.get_running_loop())
    except RuntimeError:
        pass
    return ProcessConfig(**cfg_kwargs)


# --------------------------------------------------------
# Core Process Instantiation and Setup
# --------------------------------------------------------
def instantiate_process(cfg: ProcessConfig) -> LLMProcess:
    """Create bare process instance from a ProcessConfig."""
    return LLMProcess(cfg)


def setup_runtime_context(
    process: LLMProcess, runtime_dependencies: Optional[dict[str, Any]] = None
) -> "RuntimeContext":
    """Set up runtime context for dependency injection."""
    if runtime_dependencies is not None:
        context = runtime_dependencies
    else:
        context: RuntimeContext = {"process": process}

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
    logger.info(f"Tools enabled: {len(process.tool_manager.registered_tools)}")


# --------------------------------------------------------
# Generic Process Creation Logic
# --------------------------------------------------------
async def _create_process_generic(
    program: LLMProgram,
    process_class: type = LLMProcess,
    process_kwargs: Optional[dict[str, Any]] = None,
    access_level: Optional[AccessLevel] = None,
) -> Union[LLMProcess, SyncLLMProcess]:
    """Generic process creation for both async and sync modes."""
    process_type = process_class.__name__
    logger.info(f"Starting {process_type} creation for program: {program.model_name}")

    program.compile()

    cfg = prepare_process_config(program, access_level)

    if process_kwargs:
        for key, value in process_kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    config = program.get_tool_configuration()

    tool_manager = ToolManager()
    tools_attr = getattr(program, "tools", None)
    builtin_tools = list(tools_attr or [])

    # Collect tools from plugins
    plugin_tools = []
    if hasattr(program, "plugins") and program.plugins:
        from llmproc.plugin.plugin_event_runner import PluginEventRunner

        hooks = PluginEventRunner(lambda coro: coro, program.plugins)
        plugin_tools = hooks.provide_tools()

    # Register both builtin and plugin tools
    all_tools = builtin_tools + plugin_tools
    await tool_manager.register_tools(all_tools, config)

    cfg.tool_manager = tool_manager

    if process_class is LLMProcess:
        process = instantiate_process(cfg)
    else:
        loop_arg = process_kwargs.get("_loop") if process_kwargs else None
        process = process_class(cfg, _loop=loop_arg)

    setup_runtime_context(process)
    if process.plugins and hasattr(process, "enriched_system_prompt"):
        try:
            process.enriched_system_prompt = await process.plugins.system_prompt(
                process.enriched_system_prompt, process
            )
        except TypeError:
            pass
    validate_process(process)

    logger.info(f"{process_type} created successfully for {process.model_name} ({process.provider})")
    return process


# --------------------------------------------------------
# Public Factory Functions
# --------------------------------------------------------
async def create_process(program: LLMProgram, access_level: Optional[Any] = None) -> LLMProcess:
    """Create fully initialized async LLMProcess from program."""
    return await _create_process_generic(
        program=program,
        process_class=LLMProcess,
        access_level=access_level,
    )


def create_sync_process(
    program: LLMProgram,
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
                access_level=access_level,
            )
        )
    except Exception as e:
        if not loop.is_closed():
            loop.close()
        raise e
