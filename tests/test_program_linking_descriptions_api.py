"""Tests for program linking descriptions with actual API calls."""

import asyncio
import os
from pathlib import Path
from textwrap import dedent

import pytest

from llmproc.program import LLMProgram

# Mark this test module as requiring API access
pytestmark = pytest.mark.llm_api


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp_dir = Path("tmp_test_descriptions_api")
    tmp_dir.mkdir(exist_ok=True)
    yield tmp_dir

    # Clean up test files after test
    for file in tmp_dir.glob("*.toml"):
        file.unlink()
    tmp_dir.rmdir()


@pytest.mark.asyncio
async def test_program_linking_descriptions_api(temp_dir):
    """Test program linking descriptions with actual API calls."""
    # Skip this test if no API key is available
    for key_name in ["VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION"]:
        if os.environ.get(key_name):
            break
    else:
        pytest.skip(
            "VERTEX_AI_PROJECT or VERTEX_AI_LOCATION environment variables not set"
        )

    # Create test files
    main_toml = temp_dir / "main.toml"
    expert_toml = temp_dir / "expert.toml"

    # Create the expert TOML
    expert_toml_content = dedent("""
    [model]
    name = "claude-3-5-haiku@20241022"
    provider = "anthropic_vertex"
    display_name = "Expert"

    [prompt]
    system_prompt = "You are an expert assistant. When asked about your role, explain that you are an expert with knowledge about program descriptions."

    [parameters]
    max_tokens = 500
    temperature = 0
    """)

    with open(expert_toml, "w") as f:
        f.write(expert_toml_content)

    # Create the main TOML with descriptions
    main_toml_content = dedent(f"""
    [model]
    name = "claude-3-5-haiku@20241022"
    provider = "anthropic_vertex"
    display_name = "Main"

    [prompt]
    system_prompt = "You are a helpful assistant with access to experts. For this test, when asked what experts you have access to, query the expert using the spawn tool."

    [parameters]
    max_tokens = 500
    temperature = 0

    [tools]
    enabled = ["spawn"]

    [linked_programs]
    expert = {{ path = "{expert_toml.name}", description = "Specialized expert with knowledge about program descriptions" }}
    """)

    with open(main_toml, "w") as f:
        f.write(main_toml_content)

    # Create and initialize the program with the API
    program = LLMProgram.from_toml(main_toml)
    process = await program.start()

    # Check that the descriptions were parsed correctly
    assert hasattr(program, "linked_program_descriptions")
    assert "expert" in program.linked_program_descriptions
    assert (
        program.linked_program_descriptions["expert"]
        == "Specialized expert with knowledge about program descriptions"
    )

    # Check that the spawn tool shows descriptions
    spawn_tool = next((tool for tool in process.tools if tool["name"] == "spawn"), None)
    assert spawn_tool is not None
    assert "expert" in spawn_tool["description"]
    assert "Specialized expert" in spawn_tool["description"]

    # Run the process with a prompt that will use the spawn tool
    test_prompt = "What experts do you have access to and what are they specialized in?"
    result = await process.run(test_prompt)

    # Verify the result
    final_message = process.get_last_message()
    assert "expert" in final_message.lower()
    assert "description" in final_message.lower()

    # Should have made spawn tool calls
    assert len(result.tool_calls) > 0
    assert any(call["name"] == "spawn" for call in result.tool_calls)

    # No cleanup needed in current version


if __name__ == "__main__":
    asyncio.run(
        test_program_linking_descriptions_api(Path("tmp_test_descriptions_api"))
    )
