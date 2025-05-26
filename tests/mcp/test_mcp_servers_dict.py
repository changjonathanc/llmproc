import pytest

from llmproc.config.program_loader import ProgramLoader
from llmproc.config.schema import (
    LLMProgramConfig,
    MCPConfig,
    MCPToolsConfig,
    ModelConfig,
    PromptConfig,
    ToolsConfig,
)
from llmproc.config.tool import ToolConfig


def test_mcp_servers_dict(tmp_path):
    """ProgramLoader supports embedded MCP servers without temp files."""
    config = LLMProgramConfig(
        model=ModelConfig(name="claude-3-5-sonnet", provider="anthropic"),
        prompt=PromptConfig(system_prompt="test"),
        mcp=MCPConfig(
            servers={
                "calc": {
                    "type": "stdio",
                    "command": "echo",
                    "args": ["calc"],
                }
            }
        ),
        tools=ToolsConfig(mcp=MCPToolsConfig(root={"calc": [ToolConfig(name="add")]})),
    )

    program = ProgramLoader._build_from_config(config, tmp_path)
    assert program.mcp_servers == {"calc": {"type": "stdio", "command": "echo", "args": ["calc"]}}
    cfg = program.get_tool_configuration()
    assert cfg["mcp_servers"] == program.mcp_servers
    assert cfg["mcp_enabled"] is True
