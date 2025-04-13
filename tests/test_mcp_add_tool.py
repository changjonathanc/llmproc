"""API integration test for MCP 'add' tool.

This test verifies that the MCP 'add' tool works correctly with real API calls.
It requires valid API keys to be set in the environment.
"""

import os
import re
from pathlib import Path

import pytest

from llmproc import LLMProcess
from llmproc.config.program_loader import ProgramLoader
from llmproc.program import LLMProgram


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Missing ANTHROPIC_API_KEY environment variable",
)
@pytest.mark.llm_api
@pytest.mark.essential_api  # Mark as essential for fast testing
@pytest.mark.asyncio
async def test_mcp_add_tool_integration():
    """
    NOTE: This test has been modified to accommodate the fix for the tool_calls tracking issue.

    The fix ensures that RunResult.add_tool_call() properly updates both tool_call_infos and tool_calls,
    which maintains backward compatibility with tests that check the tool_calls property.
    """
    """Test the 'add' MCP tool from the 'everything' server on a complex addition problem.
    
    Tests the specific case: 12321.124124 + 123124.1243254363 = 135445.2484494363
    """
    # Get path to the MCP example TOML file
    example_dir = Path(__file__).parent.parent / "examples" / "features"
    mcp_toml_path = example_dir / "mcp.toml"

    # Skip if the file doesn't exist
    if not mcp_toml_path.exists():
        pytest.skip(f"MCP example file not found: {mcp_toml_path}")

    # Prepare test query with our complex addition
    query = """Solve this precise addition problem: 12321.124124 + 123124.1243254363
    
    This requires high precision, so you should use the 'add' tool to get the exact answer."""

    # Compile the program (this will initialize the program, but not create the process)
    program = ProgramLoader.from_toml(mcp_toml_path)

    # Initialize the program and create a process
    process = await program.start()

    # Verify MCP tools are properly registered and enabled
    tool_schemas = process.tools
    tool_names = [schema.get("name", "") for schema in tool_schemas]

    # There should be at least one tool registered with the "everything" prefix and "add" suffix
    add_tools = [name for name in tool_names if "add" in name and "everything" in name]

    # Verify at least one add tool was registered
    assert len(add_tools) > 0, "No 'add' MCP tool was registered"

    # Add detailed diagnostic information
    print("\n-------- MCP DIAGNOSTIC INFORMATION --------")
    print(f"MCP configuration path: {program.mcp_config_path}")
    print(f"MCP tools configuration: {program.mcp_tools}")
    print(f"Available tool names: {tool_names}")

    # Check internal state of MCP manager
    if hasattr(process.tool_manager, "mcp_manager"):
        mcp_manager = process.tool_manager.mcp_manager
        print(f"MCP Manager initialized: {mcp_manager.initialized}")
        print(f"MCP Manager enabled: {mcp_manager.is_enabled()}")
        print(f"MCP Manager has aggregator: {mcp_manager.aggregator is not None}")

        if mcp_manager.aggregator:
            try:
                # Try to get tool list
                import asyncio

                servers = asyncio.run(mcp_manager.aggregator.list_servers())
                print(f"Available servers: {servers}")
            except Exception as e:
                print(f"Error listing servers: {e}")
    else:
        print("No MCP Manager found in tool_manager")

    # Check tool registry for MCP tools
    mcp_tools = [t for t in tool_names if "__" in t]
    print(f"All MCP-like tools: {mcp_tools}")
    print(f"Add tools: {add_tools}")
    print("-------- END DIAGNOSTIC INFORMATION --------")

    # Run the query
    result = await process.run(query)

    # Check the response
    response = process.get_last_message()

    # Expected result
    expected_result = "135445.2484494363"

    # Basic validation - should contain the exact answer
    assert expected_result in response, (
        f"Response doesn't contain the exact answer '{expected_result}': {response}"
    )

    # Should have used the tool - check the result
    # The tool_calls list should be populated by add_tool_call in RunResult
    assert len(result.tool_calls) > 0, "No tool calls were made"

    # Also verify tool_call_infos is populated
    assert len(result.tool_call_infos) > 0, "No tool call infos were recorded"

    # Log the response for verification
    print(f"\nResponse: {response}")
