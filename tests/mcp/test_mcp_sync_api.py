import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from llmproc import LLMProgram
from llmproc.tools.mcp import MCPServerTools

from tests.patterns import assert_successful_response, timed_test


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
def test_sync_mcp_add_tool():
    """Verify sync API with a basic MCP add tool."""
    if os.environ.get("ANTHROPIC_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing ANTHROPIC_API_KEY environment variable")

    with TemporaryDirectory() as tmpdir:
        # Create a simple MCP server script
        server_script = Path(tmpdir) / "server.py"
        server_script.write_text(
            """from mcp.server.fastmcp import FastMCP

server = FastMCP()

@server.tool()
def add(a: float, b: float) -> dict:
    return {"result": a + b}

if __name__ == '__main__':
    server.run('stdio')
"""
        )

        config_path = Path(tmpdir) / "config.json"
        config = {
            "mcpServers": {
                "calc": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(server_script)],
                }
            }
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        program = LLMProgram(
            model_name="claude-3-5-haiku-20241022",
            provider="anthropic",
            system_prompt="You are a helpful assistant.",
            parameters={"max_tokens": 100},
            mcp_config_path=str(config_path),
            tools=[MCPServerTools(server="calc", names=["add"])],
        )

        with timed_test(timeout_seconds=8.0):
            process = program.start_sync()
            result = process.run("What is 2 plus 3? Use the calc__add tool.")

        assert_successful_response(result)
        response = process.get_last_message()
        assert "5" in response
