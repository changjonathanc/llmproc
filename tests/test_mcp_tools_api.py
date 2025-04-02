"""API integration tests for MCP tools.

These tests verify that MCP tools work correctly with real API calls.
They require valid API keys to be set in the environment.
"""

import os
import re
import pytest
from pathlib import Path

from llmproc import LLMProcess
from llmproc.program import LLMProgram
from llmproc.config.program_loader import ProgramLoader


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), 
    reason="Missing ANTHROPIC_API_KEY environment variable"
)
@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_mcp_sequential_thinking_integration():
    """Test the sequential-thinking MCP tool on a real math problem using real API.
    
    This test mimics the exact example we used manually:
    python -m llmproc.cli ./examples/features/mcp.toml -p "Solve this math problem..."
    """
    # Get path to the MCP example TOML file
    example_dir = Path(__file__).parent.parent / "examples" / "features"
    mcp_toml_path = example_dir / "mcp.toml"
    
    # Skip if the file doesn't exist
    if not mcp_toml_path.exists():
        pytest.skip(f"MCP example file not found: {mcp_toml_path}")
    
    # Prepare test query
    query = "Solve this math problem using a step-by-step approach: what is the square root of 144?"
    
    # Compile the program (this will initialize the program, but not create the process)
    program = ProgramLoader.from_toml(mcp_toml_path)
    
    # Initialize the program and create a process
    process = await program.start()
    
    # Verify MCP tools are properly registered and enabled
    tool_schemas = process.tools
    tool_names = [schema.get("name", "") for schema in tool_schemas]
    
    # There should be at least one tool registered with the sequential-thinking prefix
    sequential_thinking_tools = [name for name in tool_names if name.startswith("sequential-thinking__")]
    assert len(sequential_thinking_tools) > 0, "No sequential-thinking MCP tools were registered"
    
    # Run the query
    result = await process.run(query)
    
    # Check the response
    response = process.get_last_message()
    
    # Basic validation - should mention the answer (12) and show some reasoning
    assert "12" in response, f"Response doesn't contain the answer '12': {response}"
    
    # Should have used the tool - check the result
    assert result.tool_calls > 0, "No tool calls were made"
    
    # Should have found the right answer showing some kind of step-by-step approach
    thinking_pattern = re.compile(r"(step|break down|approach|think|thought|verify|check|calculate|systematic)", re.IGNORECASE)
    assert thinking_pattern.search(response), f"Response doesn't show step-by-step thinking: {response}"