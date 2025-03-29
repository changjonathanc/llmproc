"""Test for the program linking descriptions feature with specific examples."""

import asyncio
import os
from pathlib import Path

import pytest

from llmproc.program import LLMProgram


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_program_linking_description_in_example():
    """Test program linking descriptions in the actual examples directory."""
    # Skip test if API keys aren't available
    for key_name in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
        if os.environ.get(key_name):
            break
    else:
        pytest.skip("No Anthropic API key available")
    
    # Path to the program linking example with descriptions
    program_path = Path(__file__).parent.parent / "examples" / "features" / "program-linking" / "main.toml"
    if not program_path.exists():
        pytest.skip(f"Example file not found: {program_path}")
    
    # Compile the program with linked programs
    try:
        program = LLMProgram.from_toml(program_path)
    except Exception as e:
        pytest.fail(f"Failed to compile program: {str(e)}")
    
    # Verify descriptions are parsed correctly
    assert hasattr(program, "linked_program_descriptions"), "Program missing linked_program_descriptions"
    
    # Check that we have at least two descriptions (repo_expert and thinking_expert)
    assert len(program.linked_program_descriptions) >= 2, "Expected at least two linked program descriptions"
    
    # Verify specific descriptions we expect
    assert "repo_expert" in program.linked_program_descriptions, "repo_expert missing from descriptions"
    assert "LLMProc" in program.linked_program_descriptions["repo_expert"], "repo_expert description incorrect"
    
    assert any(name.endswith("expert") for name in program.linked_program_descriptions), "No expert found in descriptions"
    
    # Check that all descriptions are non-empty
    for name, description in program.linked_program_descriptions.items():
        assert description, f"Empty description for {name}"
    
    # Start the process
    process = await program.start()
    
    # Verify process has descriptions
    assert hasattr(process, "linked_program_descriptions"), "Process missing linked_program_descriptions"
    
    # Check that spawn tool includes descriptions
    tools = process.tools
    spawn_tool = next((tool for tool in tools if tool["name"] == "spawn"), None)
    assert spawn_tool is not None, "Spawn tool missing"
    assert "description" in spawn_tool, "Spawn tool missing description"
    
    # Verify expert descriptions are included in the spawn tool
    for name, description in program.linked_program_descriptions.items():
        if description:  # Only check non-empty descriptions
            assert name in spawn_tool["description"], f"{name} not mentioned in spawn tool description"
    
    # Run a simple prompt to ensure the process works
    try:
        result = await process.run("Hello")
        response = process.get_last_message()
        assert response, "Empty response from process"
    except Exception as e:
        pytest.fail(f"Failed to run process: {str(e)}")
    finally:
        await process.close()


if __name__ == "__main__":
    asyncio.run(test_program_linking_description_in_example())