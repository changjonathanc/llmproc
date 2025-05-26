"""Tests for the ToolConfig and MCPServerTools classes."""

import pytest

from llmproc.common.access_control import AccessLevel
from llmproc.tools.mcp import MCPServerTools
from llmproc.config.tool import ToolConfig


def test_tool_config_defaults():
    """Test that ToolConfig uses correct default values."""
    item = ToolConfig("add")
    assert item.name == "add"
    assert item.access == AccessLevel.WRITE


def test_tool_config_access_level():
    """Test that ToolConfig correctly sets the access level."""
    item1 = ToolConfig("add", AccessLevel.READ)
    assert item1.name == "add"
    assert item1.access == AccessLevel.READ

    item2 = ToolConfig("add", "read")
    assert item2.name == "add"
    assert item2.access == AccessLevel.READ


def test_tool_config_invalid_name():
    """Test that ToolConfig raises an error for invalid names."""
    with pytest.raises(ValueError):
        ToolConfig("")

    with pytest.raises(ValueError):
        ToolConfig(None)


def test_mcp_server_tools_all_tools():
    """Test MCPServerTools with 'all' tools specification."""
    tools = MCPServerTools("calc")
    assert tools.server == "calc"
    assert tools.tools == "all"
    assert tools.default_access == AccessLevel.WRITE


def test_mcp_server_tools_string_tool():
    """Test MCPServerTools with a single tool name as string."""
    tools = MCPServerTools("calc", "add")
    assert tools.server == "calc"
    assert tools.tools == ["add"]
    assert tools.default_access == AccessLevel.WRITE


def test_mcp_server_tools_list_tools():
    """Test MCPServerTools with a list of tool names."""
    tools = MCPServerTools("calc", ["add", "subtract"])
    assert tools.server == "calc"
    assert tools.tools == ["add", "subtract"]
    assert tools.default_access == AccessLevel.WRITE


def test_mcp_server_tools_with_access():
    """Test MCPServerTools with a specified access level."""
    tools = MCPServerTools("calc", ["add"], AccessLevel.READ)
    assert tools.server == "calc"
    assert tools.tools == ["add"]
    assert tools.default_access == AccessLevel.READ

    tools2 = MCPServerTools("calc", ["add"], "read")
    assert tools2.server == "calc"
    assert tools2.tools == ["add"]
    assert tools2.default_access == AccessLevel.READ


def test_mcp_server_tools_with_dict():
    """Test MCPServerTools with a dictionary of tool names to access levels."""
    tools = MCPServerTools("calc", {"add": "write", "subtract": "read"})
    assert tools.server == "calc"
    assert {t.name for t in tools.tools} == {"add", "subtract"}
    assert tools.default_access is None
    assert tools.get_access_level("add") == AccessLevel.WRITE
    assert tools.get_access_level("subtract") == AccessLevel.READ


def test_mcp_server_tools_description_override_dict():
    """Tools defined with dict values support description override."""
    tools = MCPServerTools(
        "calc",
        {"add": {"access": "read", "description": "Add numbers", "param_descriptions": {"a": "num"}}},
    )
    assert isinstance(tools.tools[0], ToolConfig)
    assert tools.get_description("add") == "Add numbers"
    assert tools.get_param_descriptions("add") == {"a": "num"}


def test_mcp_server_tools_with_tool_items():
    """Test MCPServerTools with a list of ToolConfig objects."""
    items = [ToolConfig("add", AccessLevel.WRITE), ToolConfig("subtract", AccessLevel.READ)]
    tools = MCPServerTools("calc", items)
    assert tools.server == "calc"
    assert {t.name for t in tools.tools} == {"add", "subtract"}
    assert tools.default_access is None
    assert tools.get_access_level("add") == AccessLevel.WRITE
    assert tools.get_access_level("subtract") == AccessLevel.READ


def test_mcp_server_tools_invalid_server():
    """Test that MCPServerTools raises an error for invalid server names."""
    with pytest.raises(ValueError):
        MCPServerTools("")

    with pytest.raises(ValueError):
        MCPServerTools(None)


def test_mcp_server_tools_invalid_tools():
    """Test that MCPServerTools raises an error for invalid tool names."""
    with pytest.raises(ValueError):
        MCPServerTools("calc", [None])

    with pytest.raises(ValueError):
        MCPServerTools("calc", [""])


def test_mcp_server_tools_invalid_combination():
    """Test that MCPServerTools raises an error for invalid parameter combinations."""
    with pytest.raises(ValueError):
        MCPServerTools("calc", {"add": "write"}, AccessLevel.READ)

    with pytest.raises(ValueError):
        items = [ToolConfig("add")]
        MCPServerTools("calc", items, AccessLevel.READ)
