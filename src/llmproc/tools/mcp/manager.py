"""MCPManager class implementation.

This module provides the MCPManager class for managing MCP tools and servers.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llmproc.common.access_control import AccessLevel
from llmproc.common.metadata import ToolMeta, attach_meta
from llmproc.common.results import ToolResult
from llmproc.tools.mcp import MCPServerTools
from llmproc.tools.mcp.constants import (
    MCP_DEFAULT_TOOL_CALL_TIMEOUT,
    MCP_DEFAULT_TOOL_FETCH_TIMEOUT,
    MCP_ERROR_INIT_FAILED,
    MCP_ERROR_NO_TOOLS_REGISTERED,
    MCP_ERROR_TOOL_FETCH_TIMEOUT,
    MCP_LOG_ENABLED_TOOLS,
    MCP_LOG_INITIALIZING_SERVERS,
    MCP_LOG_MCP_TOOL_NAMES,
    MCP_LOG_NO_SERVERS,
    MCP_LOG_NO_TOOLS_REGISTERED,
    MCP_LOG_REGISTERED_SERVER_TOOLS,
    MCP_LOG_RETRY_FETCH,
    MCP_LOG_TOTAL_REGISTERED,
    MCP_MAX_FETCH_RETRIES,
    MCP_TOOL_SEPARATOR,
)


# Utility function to create tool handlers with properly bound variables
def create_mcp_tool_handler(aggregator: Any, namespaced_tool_name: str) -> Callable:
    """Create a properly bound handler function for an MCP tool.

    Args:
        aggregator: The MCP aggregator instance
        namespaced_tool_name: The full tool name with namespace (e.g., "everything__add")
                             Used directly by MCPAggregator.call_tool
    """

    async def tool_handler(**kwargs) -> ToolResult:
        try:
            # MCPAggregator.call_tool expects the full namespaced tool name and kwargs
            result = await aggregator.call_tool(namespaced_tool_name, kwargs)
            if result.isError:
                return ToolResult(content=result.content, is_error=True)
            return ToolResult(content=result.content, is_error=False)
        except Exception as e:
            error_message = f"Error calling MCP tool {namespaced_tool_name}: {e}"
            logger.error(error_message)
            return ToolResult.from_error(error_message)

    return tool_handler


if TYPE_CHECKING:
    # Import MCP registry types only for type checking
    from llmproc import LLMProcess
    from llmproc.mcp_registry import MCPAggregator, ServerRegistry
    from llmproc.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages MCP tools and server connections."""

    def __init__(
        self,
        config_path: str | None = None,
        servers: dict[str, Any] | None = None,
        mcp_tools: list[MCPServerTools] | None = None,
        provider: str | None = None,
    ):
        """Initialize the MCP Manager.

        The MCP Manager follows the configuration-based approach which avoids
        circular dependencies between LLMProcess and tool initialization.

        Args:
            config_path: Path to the MCP configuration file
            servers: Embedded MCP servers dictionary
            mcp_tools: List of MCPServerTools descriptors specifying servers and tools
            provider: The provider name (e.g., "anthropic")
        """
        self.config_path = config_path
        self.servers = servers
        self.mcp_tools = mcp_tools or []
        self.aggregator = None
        self.initialized = False
        self.provider = provider

        # Validate provider (anthropic and anthropic_vertex are supported)
        if self.provider and self.provider not in ["anthropic", "anthropic_vertex"]:
            logger.warning(
                f"Provider {self.provider} is not supported for MCP. Only anthropic and anthropic_vertex are currently supported."
            )

    def is_enabled(self) -> bool:
        """Check if MCP is enabled and properly configured."""
        return bool(self.config_path or self.servers)

    def is_valid_configuration(self) -> bool:
        """Check if the MCP configuration is valid."""
        if not (self.config_path or self.servers):
            logger.warning("MCP configuration is not set")
            return False

        # Empty tools config is now valid - we just won't register any tools
        # but the manager should still initialize successfully

        # All checks passed
        return True

    async def initialize(self, tool_registry: "ToolRegistry") -> bool:
        """Initialize MCP registry and tools.

        This method initializes the MCP registry without applying any initial filtering.
        Tool selection is now handled through MCPServerTools descriptors via register_tools.

        Args:
            tool_registry: The ToolRegistry to register tools with

        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        # Check configuration validity
        if not self.is_valid_configuration():
            logger.warning("MCP configuration is not valid, skipping initialization")
            return False
        try:
            # Lazy import to avoid circular dependencies
            from llmproc.mcp_registry import MCPAggregator, ServerRegistry

            if self.servers is not None:
                full_registry = ServerRegistry.from_dict(self.servers)
            else:
                full_registry = ServerRegistry.from_config(self.config_path)
            server_names = [t.server for t in self.mcp_tools]
            registry = full_registry.filter_servers(server_names)

            # Create aggregator from the filtered registry
            self.aggregator = MCPAggregator(registry)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(MCP_ERROR_INIT_FAILED.format(error=str(e)))
            return False

    async def get_tool_registrations(self, tool_fetch_timeout: float = None) -> list[tuple[str, Callable, dict]]:
        """Return a list of (name, handler, schema) for all MCP tools.

        Args:
            tool_fetch_timeout: Maximum time in seconds to wait for tool fetching.
                               Defaults to value from LLMPROC_TOOL_FETCH_TIMEOUT env var or 30.0

        Returns:
            List of tuples containing (tool_name, handler_function, schema_dict)
        """
        # Get timeout from environment variable or use default
        if tool_fetch_timeout is None:
            tool_fetch_timeout = float(os.environ.get("LLMPROC_TOOL_FETCH_TIMEOUT", MCP_DEFAULT_TOOL_FETCH_TIMEOUT))
        if not self.initialized or not self.aggregator:
            return []
        # Lazy import to avoid circular deps
        from llmproc.tools.mcp.handlers import format_tool_for_anthropic

        regs: list[tuple[str, Callable, dict]] = []

        # Helper to fetch and cache tool list for a single server with retry logic
        async def _get_server_tools_with_retry(server_name: str):
            """Return the list of Tool objects for *server_name* with retry logic."""
            retry_count = 0
            max_retries = MCP_MAX_FETCH_RETRIES
            fail_on_init_timeout = os.environ.get("LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT", "true").lower() in (
                "true",
                "1",
                "yes",
            )

            while retry_count <= max_retries:
                try:
                    if getattr(self.aggregator, "transient", False):
                        async with self.aggregator.registry.get_client(server_name) as client:
                            result = await client.list_tools()
                            return result.tools or []
                    client = await self.aggregator._get_or_create_client(server_name)  # type: ignore[attr-defined]
                    result = await client.list_tools()
                    return result.tools or []
                except TimeoutError:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(
                            MCP_LOG_RETRY_FETCH.format(
                                server=server_name, attempt=retry_count, max_attempts=max_retries + 1
                            )
                        )
                        # Exponential backoff
                        await asyncio.sleep(1 * (2 ** (retry_count - 1)))
                    else:
                        error_msg = MCP_ERROR_TOOL_FETCH_TIMEOUT.format(server=server_name, timeout=tool_fetch_timeout)
                        logger.error(error_msg)

                        if fail_on_init_timeout:
                            raise RuntimeError(error_msg)
                        return []
                except Exception as exc:  # noqa: BLE001 – log and continue
                    error_msg = f"Unable to fetch tools from MCP server '{server_name}': {exc}"
                    logger.error(error_msg)

                    if fail_on_init_timeout:
                        raise RuntimeError(error_msg)
                    return []

        # Cache per-server to avoid redundant list_tools calls when multiple
        # descriptors reference the same server.
        # Fetch tool lists from all distinct servers concurrently to minimise
        # overall start-up latency.
        server_names = {d.server for d in self.mcp_tools}

        # Read environment variable to determine whether to fail on MCP initialization timeout
        fail_on_init_timeout = os.environ.get("LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        try:
            # Apply timeout to the concurrent tool fetching
            async with asyncio.timeout(tool_fetch_timeout):
                server_tool_cache: dict[str, list] = {
                    name: tools
                    for name, tools in zip(
                        server_names,
                        await asyncio.gather(*(_get_server_tools_with_retry(s) for s in server_names)),
                        strict=False,
                    )
                }
        except TimeoutError:
            # Provide more detailed error with server names
            server_list = ", ".join(server_names)
            error_msg = (
                f"Global timeout fetching tools from MCP servers ({server_list}) after {tool_fetch_timeout:.1f} seconds. "
                f"This typically happens when MCP servers are slow to respond or not running properly. "
                f"If you're using npx to run MCP servers, this may indicate the package doesn't exist or npm registry access issues. "
                f"Consider increasing LLMPROC_TOOL_FETCH_TIMEOUT environment variable "
                f"(current value: {tool_fetch_timeout:.1f} seconds) or check the MCP servers' status."
            )
            logger.error(error_msg)

            if fail_on_init_timeout:
                raise RuntimeError(error_msg)
            return []
        except Exception as exc:
            error_msg = f"Error fetching tools from MCP servers: {exc}"
            logger.error(error_msg)

            if fail_on_init_timeout:
                raise RuntimeError(error_msg)
            return []

        for descriptor in self.mcp_tools:
            server = descriptor.server
            server_tools = server_tool_cache.get(server, [])
            if not server_tools:
                # Check if we should fail when a server returns no tools
                if fail_on_init_timeout:
                    error_msg = (
                        f"MCP server '{server}' returned no tools. This may indicate a server configuration issue."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                continue

            for tool in server_tools:
                # Tool selection
                if descriptor.tools != "all":
                    # Support lists containing plain strings or ToolConfig objects
                    allowed = False
                    if isinstance(descriptor.tools, list):
                        for item in descriptor.tools:
                            if (isinstance(item, str) and item == tool.name) or (
                                not isinstance(item, str) and getattr(item, "name", None) == tool.name
                            ):
                                allowed = True
                                break
                    if not allowed:
                        continue

                access_level = descriptor.get_access_level(tool.name)

                override_desc = descriptor.get_description(tool.name)
                param_desc = descriptor.get_param_descriptions(tool.name)

                existing_desc: dict[str, str] = {}
                try:
                    for pname, prop in (tool.inputSchema or {}).get("properties", {}).items():
                        if isinstance(prop, dict) and "description" in prop:
                            existing_desc[pname] = prop["description"]
                except Exception:
                    pass
                if param_desc:
                    existing_desc.update(param_desc)

                namespaced_tool_name = f"{server}{MCP_TOOL_SEPARATOR}{tool.name}"
                handler = create_mcp_tool_handler(self.aggregator, namespaced_tool_name)

                meta = ToolMeta(
                    name=namespaced_tool_name,
                    access=access_level,
                    description=override_desc or tool.description,
                    param_descriptions=existing_desc or None,
                )
                attach_meta(handler, meta)

                schema = format_tool_for_anthropic(tool, server)
                if override_desc is not None:
                    schema["description"] = override_desc
                if param_desc:
                    for pname, pdesc in param_desc.items():
                        try:
                            schema["input_schema"]["properties"][pname]["description"] = pdesc
                        except Exception:
                            continue
                regs.append((namespaced_tool_name, handler, schema))

        return regs
