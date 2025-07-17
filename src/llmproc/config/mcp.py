"""MCP configuration models.

This module provides MCP tool configuration models with simplified interfaces.
"""

from typing import Any, Literal

from pydantic import BaseModel, RootModel, field_validator, model_validator

from llmproc.common.access_control import AccessLevel
from llmproc.config.tool import ToolConfig


class MCPServerTools(BaseModel):
    """Descriptor for selecting tools on an MCP server.

    Examples:
    --------
    >>> MCPServerTools(server="calc")                  # All tools with default WRITE
    >>> MCPServerTools(server="calc", tools=["add"])   # List of specific tools (WRITE)
    >>> MCPServerTools(                                # With custom access level
    ...     server="calc",
    ...     tools=["add"],
    ...     default_access=AccessLevel.READ
    ... )
    >>> MCPServerTools(server="calc", tools=[          # Per-tool access via objects
    ...     ToolConfig(name="add", access=AccessLevel.READ),
    ...     ToolConfig(name="subtract")
    ... ])
    """

    server: str
    # Can be "all" or a list of strings/ToolConfig objects
    tools: Literal["all"] | list[str | ToolConfig] = "all"
    # Default access applied to plain string tool entries
    default_access: AccessLevel | None = AccessLevel.WRITE

    # Support both positional and keyword arguments for backward compatibility
    def __init__(self, server=None, tools=None, default_access=None, **kwargs):
        # If positional arguments are provided, convert them to keyword arguments
        if server is not None and "server" not in kwargs:
            kwargs["server"] = server
        if tools is not None and "tools" not in kwargs:
            kwargs["tools"] = tools
        if default_access is not None and "default_access" not in kwargs:
            kwargs["default_access"] = default_access
        # Pass on to the standard initialization
        super().__init__(**kwargs)

    @field_validator("server")
    @classmethod
    def validate_server(cls, v: str):
        """Validate server name is not empty."""
        if not v:
            raise ValueError("Server name cannot be empty")
        return v

    @field_validator("tools", mode="before")
    @classmethod
    def parse_tools(cls, value: Any) -> Literal["all"] | list[str | ToolConfig]:
        """Normalize tools into the canonical representation.

        The canonical form is either "all" or a list of strings/ToolConfig objects.
        """
        if value in (None, "all"):
            return "all"

        if isinstance(value, str):
            return [value]

        if isinstance(value, dict):
            return [cls._parse_dict_item(name, val) for name, val in value.items()]

        if isinstance(value, list):
            return [cls._parse_list_item(item) for item in value]

        raise ValueError(f"Unsupported tools specification type: {type(value)}")

    def _find_tool(self, name: str) -> ToolConfig | None:
        """Return ToolConfig for *name* if present."""
        if self.tools == "all" or not isinstance(self.tools, list):
            return None
        for item in self.tools:
            if isinstance(item, ToolConfig) and item.name == name:
                return item
        return None

    @staticmethod
    def _normalize_access(value: AccessLevel | str | None) -> AccessLevel | None:
        """Convert a string or enum to an ``AccessLevel``."""
        if isinstance(value, str):
            return AccessLevel(value.lower())
        return value

    @classmethod
    def _parse_dict_item(cls, name: str, val: Any) -> ToolConfig:
        """Create ``ToolConfig`` from a dictionary entry."""
        if isinstance(val, dict):
            access = cls._normalize_access(val.get("access"))
            return ToolConfig(
                name=name,
                access=access,
                alias=val.get("alias"),
                description=val.get("description"),
                param_descriptions=val.get("param_descriptions"),
            )

        access = cls._normalize_access(val)
        return ToolConfig(name=name, access=access)

    @classmethod
    def _parse_list_item(cls, item: Any) -> str | ToolConfig:
        """Normalize a list entry to a string or ``ToolConfig``."""
        if isinstance(item, str):
            if not item:
                raise ValueError("Tool names cannot be empty")
            return item

        if isinstance(item, ToolConfig):
            if not item.name:
                raise ValueError("Tool name cannot be empty")
            return item

        if isinstance(item, dict):
            try:
                return ToolConfig(**item)
            except TypeError as exc:
                raise ValueError(f"Invalid tool definition: {item}") from exc

        raise ValueError(f"Unsupported tool specification type: {type(item)}")

    @model_validator(mode="after")
    def validate_combinations(self):
        """Validate that the configuration is consistent."""
        # If tools is a list of ToolConfig objects with explicit access levels,
        # conflicting with a non-default default_access
        has_per_tool_access = isinstance(self.tools, list) and any(isinstance(item, ToolConfig) for item in self.tools)
        if has_per_tool_access and self.default_access not in (None, AccessLevel.WRITE):
            raise ValueError("Cannot specify both per-tool access levels and a non-default default_access")

        # If all tools have explicit access levels, set default_access to None
        if isinstance(self.tools, list) and all(
            isinstance(item, ToolConfig) and item.access is not None for item in self.tools
        ):
            self.default_access = None

        return self

    def get_access_level(self, tool_name: str) -> AccessLevel:
        """Return the access level for the specified tool.

        Args:
            tool_name: Name of the tool to get access level for

        Returns:
            The appropriate AccessLevel for the tool
        """
        if self.tools == "all":
            return self.default_access or AccessLevel.WRITE

        if isinstance(self.tools, list):
            if tool_name in self.tools:
                return self.default_access or AccessLevel.WRITE
            cfg = self._find_tool(tool_name)
            if cfg is not None:
                return cfg.access or (self.default_access or AccessLevel.WRITE)

        return self.default_access or AccessLevel.WRITE

    def get_tool_names(self) -> list[str]:
        """Get a list of all tool names.

        Returns:
            A list of tool name strings
        """
        if self.tools == "all":
            return []  # Can't enumerate "all" tools until we connect to the server

        result = []
        for item in self.tools:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, ToolConfig):
                result.append(item.name)
        return result

    def __str__(self) -> str:
        """Return a user-friendly representation."""
        if self.tools == "all":
            access_str = (
                f" (access={self.default_access.value})"
                if self.default_access and self.default_access != AccessLevel.WRITE
                else ""
            )
            return f"<MCPServerTools {self.server}=ALL{access_str}>"

        # Build per-tool representation
        tool_parts = []
        for item in self.tools:
            if isinstance(item, str):
                tool_parts.append(item)
            else:
                if item.access and item.access != self.default_access:
                    tool_parts.append(f"{item.name}={item.access.value}")
                else:
                    tool_parts.append(item.name)

        tools_repr = ", ".join(tool_parts)
        access_str = (
            f" (access={self.default_access.value})"
            if self.default_access and self.default_access != AccessLevel.WRITE
            else ""
        )
        return f"<MCPServerTools {self.server}=[{tools_repr}]{access_str}>"

    __repr__ = __str__


class MCPToolsConfig(RootModel):
    """MCP tools configuration.

    This provides a simplified configuration model for MCP tools with four supported formats:

    1. "all" - Include all tools from a server
       example = "all"

    2. List of tool names - Include specific tools with default access
       example = ["tool1", "tool2"]

    3. List of ToolConfig objects - Include specific tools with custom access
       example = [
           ToolConfig(name="tool1", access=AccessLevel.READ),
           ToolConfig(name="tool2")
       ]
    4. Dictionary of tool names to access/description dictionaries
       example = {
           "tool1": {"access": "read", "description": "Custom desc"},
           "tool2": {"description": "Another desc"}
       }
    """

    root: dict[str, Literal["all"] | list[str | ToolConfig | dict[str, Any]]] = {}

    @field_validator("root")
    @classmethod
    def validate_tools(cls, v):
        """Validate that tool configurations follow the supported formats."""
        for server, tools in v.items():
            if not server:
                raise ValueError("Server name cannot be empty")

            if tools == "all":
                continue

            if isinstance(tools, dict):
                for name, item in tools.items():
                    if not name:
                        raise ValueError(f"Empty tool name in server '{server}'")
                    if isinstance(item, dict):
                        continue
                    elif isinstance(item, str):
                        continue
                    elif isinstance(item, AccessLevel):
                        continue
                    else:
                        raise ValueError(f"Unsupported tool specification in server '{server}'")
                continue

            if not isinstance(tools, list):
                raise ValueError(f"Tool configuration for server '{server}' must be 'all' or a list")

            # Validate each item in the list
            for item in tools:
                if isinstance(item, str):
                    if not item:
                        raise ValueError(f"Empty tool name in server '{server}'")
                elif isinstance(item, dict):
                    if "name" not in item:
                        raise ValueError(f"Missing 'name' field in tool definition for server '{server}'")
                    if not item["name"]:
                        raise ValueError(f"Empty tool name in server '{server}'")
                elif isinstance(item, ToolConfig):
                    if not item.name:
                        raise ValueError(f"Empty tool name in server '{server}'")
                else:
                    raise ValueError(f"Unsupported tool specification in server '{server}'")
        return v

    def build_mcp_tools(self) -> list[MCPServerTools]:
        """Convert configuration entries to MCPServerTools objects.

        Returns:
            List of MCPServerTools objects ready for use
        """
        result = []

        for server, entry in self.root.items():
            result.append(MCPServerTools(server=server, tools=entry))

        return result
