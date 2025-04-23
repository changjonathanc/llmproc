"""
MCP tool descriptor for selecting tools from MCP servers.
"""

from typing import Dict, List, Optional, Union

from llmproc.common.access_control import AccessLevel


class MCPTool:
    """
    Descriptor for selecting tools from an MCP server.

    Usage:
       MCPTool(server="calc")                          # all tools on "calc" with default WRITE access
       MCPTool(server="calc", names="add")             # just "add" with default WRITE access
       MCPTool(server="calc", names=["add", "sub"])    # multiple tools with default WRITE access
       MCPTool(server="calc", names=["add"], access=AccessLevel.READ)  # tools with READ access
       MCPTool(server="calc", names={"add": "write", "read": "read"})  # per-tool access levels
    """

    def __init__(
        self, 
        server: str, 
        names: Union[str, list[str], Dict[str, str]] = "all", 
        access: Optional[Union[AccessLevel, str]] = None
    ):
        if not server or not isinstance(server, str):
            raise ValueError("MCPTool requires a non-empty server name")
        self.server = server

        # Initialize names and access levels
        if isinstance(names, dict):
            # Dictionary form: {"tool1": "read", "tool2": "write"}
            if access is not None:
                raise ValueError("Cannot specify both names dictionary and access parameter")
                
            self.names_to_access = {}
            for tool_name, tool_access in names.items():
                if not isinstance(tool_name, str) or not tool_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")
                    
                if isinstance(tool_access, str):
                    self.names_to_access[tool_name] = AccessLevel.from_string(tool_access)
                else:
                    self.names_to_access[tool_name] = tool_access
                    
            self.names = list(names.keys())
            self.default_access = None
            
        else:
            # String or list form with uniform access
            # Convert names to proper format
            if names == "all":
                self.names = "all"
            elif isinstance(names, list):
                invalid = [n for n in names if not isinstance(n, str) or not n]
                if invalid:
                    raise ValueError(f"MCPTool invalid tool names: {invalid}")
                self.names = names
            elif isinstance(names, str):
                self.names = [names]
            else:
                raise ValueError(f"Invalid names type: {type(names)}")
                
            # Parse access level
            if access is None:
                self.default_access = AccessLevel.WRITE
            elif isinstance(access, str):
                self.default_access = AccessLevel.from_string(access)
            else:
                self.default_access = access
                
            self.names_to_access = None

    def __repr__(self) -> str:
        if isinstance(self.names, str) and self.names == "all":
            access_str = f", access={self.default_access.value}" if self.default_access != AccessLevel.WRITE else ""
            return f"<MCPTool {self.server}=ALL{access_str}>"
        elif self.names_to_access:
            levels = ", ".join(f"{name}={access.value}" for name, access in self.names_to_access.items())
            return f"<MCPTool {self.server}={{{levels}}}>"
        else:
            access_str = f", access={self.default_access.value}" if self.default_access != AccessLevel.WRITE else ""
            return f"<MCPTool {self.server}={self.names}{access_str}>"