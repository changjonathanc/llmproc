"""
MCP tool descriptor for selecting tools from MCP servers.
"""

from typing import List, Union


class MCPTool:
    """
    Descriptor for selecting tools from an MCP server.

    Usage:
      MCPTool("calc")                 # all tools on "calc"
      MCPTool("calc", "add")        # just "add"
      MCPTool("calc", "add", "sub")# "add" and "sub"
      MCPTool("calc", ["mul","div"])# list form
    """

    def __init__(self, server: str, *names: Union[str, list[str]]):
        if not server or not isinstance(server, str):
            raise ValueError("MCPTool requires a non-empty server name")
        self.server = server

        # Flatten single list arg
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = names[0]

        # No names => wildcard (all)
        if not names:
            self.names: Union[str, list[str]] = "all"
        else:
            invalid = [n for n in names if not isinstance(n, str) or not n]
            if invalid:
                raise ValueError(f"MCPTool invalid tool names: {invalid}")
            self.names = list(names)  # type: ignore[list-item]

    def __repr__(self) -> str:
        if self.names == "all":
            return f"<MCPTool {self.server}=ALL>"
        return f"<MCPTool {self.server}={self.names}>"
