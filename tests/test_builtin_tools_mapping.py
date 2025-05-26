"""Test for the builtin tools mapping and schema modifier functionality."""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from llmproc.program import LLMProgram
from llmproc.tools.builtin import BUILTIN_TOOLS
from llmproc.tools.builtin.integration import load_builtin_tools
from llmproc.tools.function_tools import register_tool
from llmproc.tools.tool_registry import ToolRegistry


def test_builtin_tools_mapping_exists():
    """Test that the BUILTIN_TOOLS mapping exists and contains expected tools."""
    # Verify mapping exists
    assert isinstance(BUILTIN_TOOLS, dict)

    # Check for some expected tools
    assert "calculator" in BUILTIN_TOOLS
    assert "read_file" in BUILTIN_TOOLS
    assert "spawn" in BUILTIN_TOOLS
    assert "goto" in BUILTIN_TOOLS

    # Verify all items are callable
    for name, func in BUILTIN_TOOLS.items():
        assert callable(func)


def test_schema_modifier_in_register_tool():
    """Test that register_tool decorator accepts and stores schema_modifier."""

    def test_modifier(schema, config):
        schema["description"] += " (modified)"
        return schema

    @register_tool(name="test_tool", description="Test tool", schema_modifier=test_modifier)
    async def test_function(arg1, arg2=None):
        """Test function docstring."""
        return f"Result: {arg1}, {arg2}"

    # Verify schema_modifier is stored in metadata
    from llmproc.common.metadata import get_tool_meta

    meta = get_tool_meta(test_function)
    assert meta.schema_modifier == test_modifier

    # Test that create_tool_from_function applies the modifier
    from llmproc.tools.function_tools import create_tool_from_function

    # Without config - should not apply modifier
    handler1, schema1 = create_tool_from_function(test_function)
    assert "modified" not in schema1["description"]

    # With config - should apply modifier
    test_config = {"some_data": "test"}
    handler2, schema2 = create_tool_from_function(test_function, test_config)
    assert "modified" in schema2["description"]


def test_load_builtin_tools_uses_mapping():
    """Test that load_builtin_tools uses the BUILTIN_TOOLS mapping."""
    # Create mock registry
    registry = ToolRegistry()

    # Call load_builtin_tools
    result = load_builtin_tools(registry)

    # Verify success
    assert result is True

    # Verify tools were registered
    tool_names = registry.get_tool_names()
    assert len(tool_names) == len(BUILTIN_TOOLS)

    # Verify some specific tools
    assert "calculator" in tool_names
    assert "read_file" in tool_names
    assert "spawn" in tool_names


def test_spawn_tool_schema_modifier():
    """Test that spawn tool's schema modifier is applied."""
    # Verify spawn tool has schema_modifier in metadata
    from llmproc.common.metadata import get_tool_meta
    from llmproc.tools.builtin.spawn import modify_spawn_schema, spawn_tool

    meta = get_tool_meta(spawn_tool)
    assert meta.schema_modifier == modify_spawn_schema

    # Test schema modifier function
    test_schema = {"description": "Original description"}
    test_config = {
        "linked_programs": {"prog1": {}, "prog2": {}},
        "linked_program_descriptions": {"prog1": "Program 1 desc"},
    }

    modified_schema = modify_spawn_schema(test_schema, test_config)

    # Verify schema was modified
    assert "Original description" in modified_schema["description"]
    assert "Available Programs" in modified_schema["description"]
    assert "prog1" in modified_schema["description"]
    assert "prog2" in modified_schema["description"]
    assert "Program 1 desc" in modified_schema["description"]

    # Test schema modification via create_tool_from_function
    from llmproc.tools.function_tools import create_tool_from_function

    _, schema = create_tool_from_function(spawn_tool, test_config)
    assert "Available Programs" in schema["description"]
    assert "prog1" in schema["description"]


def test_simplified_register_system_tools():
    """Test the simplified register_system_tools function."""
    from llmproc.tools.builtin import BUILTIN_TOOLS
    from llmproc.tools.builtin.integration import register_system_tools
    from llmproc.tools.tool_registry import ToolRegistry

    # Create source and target registries
    source_registry = ToolRegistry()
    target_registry = ToolRegistry()

    # Load builtin tools into source registry
    from llmproc.tools.builtin.integration import load_builtin_tools

    load_builtin_tools(source_registry)

    # Set up test config with dependencies
    test_config = {
        "fd_manager": MagicMock(),  # Mock fd_manager
        "has_linked_programs": True,
        "linked_programs": {"test_program": {}},
        "linked_program_descriptions": {"test_program": "Test program description"},
    }

    # Add register_fd_tool method to fd_manager mock
    test_config["fd_manager"].register_fd_tool = MagicMock(return_value=None)

    # Call register_system_tools for various tools
    enabled_tools = ["calculator", "read_file", "spawn", "read_fd", "fd_to_file"]
    registered_count = register_system_tools(source_registry, target_registry, enabled_tools, test_config)

    # Verify correct number of tools registered
    assert registered_count == len(enabled_tools)

    # Verify specific tools were registered
    assert "calculator" in target_registry.get_tool_names()
    assert "read_file" in target_registry.get_tool_names()
    assert "spawn" in target_registry.get_tool_names()
    assert "read_fd" in target_registry.get_tool_names()
    assert "fd_to_file" in target_registry.get_tool_names()

    # fd_manager.register_fd_tool is called during FD manager initialization in program_exec.py

    # Verify spawn tool schema includes program descriptions
    spawn_schema = next(schema for schema in target_registry.get_definitions() if schema.get("name") == "spawn")
    assert "Available Programs" in spawn_schema["description"]
    assert "test_program" in spawn_schema["description"]
    assert "Test program description" in spawn_schema["description"]


def test_register_tools_accepts_mixed_input():
    """Test that register_tools handles both string names and callables."""
    from llmproc.tools.builtin import calculator

    program = LLMProgram(model_name="test_model", provider="test_provider")

    # Test with string tool name - should work now
    program.register_tools(["calculator"])

    # Test with direct function reference - should work
    program.register_tools([calculator])

    # Test with dictionary access - should also work
    program.register_tools([BUILTIN_TOOLS["calculator"]])

    # Test with mixed input - should work
    program.register_tools(["calculator", calculator])


def test_toml_string_to_function_conversion():
    """Test that string tool names in TOML are converted to function references."""
    import os
    import tempfile

    # Create a temporary TOML file with string tool names
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as temp_file:
        temp_file.write(
            """
        [model]
        name = "test-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Test system prompt"

        [tools]
        builtin = ["calculator", "read_file"]
        """
        )
        temp_path = temp_file.name

    try:
        # Ignore the deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Load the program from the TOML file
            program = LLMProgram.from_toml(temp_path)

            # Find the function tools list to check conversion worked
            function_tools = program.tool_manager.function_tools

            # Verify the tools were converted to function references
            assert len(function_tools) >= 2, f"Expected at least 2 function tools, got {len(function_tools)}"

            # Verify at least calculator and read_file are in the function tools
            assert any(func is BUILTIN_TOOLS["calculator"] for func in function_tools), (
                "calculator function not in function_tools"
            )
            assert any(func is BUILTIN_TOOLS["read_file"] for func in function_tools), (
                "read_file function not in function_tools"
            )

            # Make sure the functions work properly by using process_function_tools()
            program.tool_manager.process_function_tools()

            # Check that runtime_registry has the tools registered
            assert "calculator" in program.tool_manager.runtime_registry.tool_handlers
            assert "read_file" in program.tool_manager.runtime_registry.tool_handlers

    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


def test_program_validates_tool_dependencies():
    """Test that program compilation validates tool dependencies."""
    from llmproc.tools.builtin import read_fd_tool, spawn_tool

    # Test spawn tool requires linked programs
    program = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test prompt",
        linked_programs={"test": "dummy_path"},  # Add linked programs to satisfy dependency
    )
    program.register_tools([spawn_tool])

    # Spawn tool compilation should complete with proper linked programs
    with pytest.warns(UserWarning):
        program.compile()
    assert program.compiled

    # FD tool compilation should complete with FD system configured
    program2 = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test prompt",
        file_descriptor={"enabled": True},  # Enable file descriptor system to satisfy dependency
    )
    program2.register_tools([read_fd_tool])
    program2.compile()
    assert program2.compiled

    # Test that proper configuration passes validation
    program3 = LLMProgram(model_name="test-model", provider="anthropic", system_prompt="Test prompt")

    # Configure FD system and add FD tool
    program3.configure_file_descriptor(enabled=True)
    program3.register_tools([read_fd_tool])

    # Should compile without errors
    program3.compile()
    assert program3.compiled


@pytest.mark.asyncio
async def test_direct_tool_registration():
    """Test that initialize_tools directly registers tools without using builtin registry."""
    from llmproc.tools import ToolManager
    from llmproc.tools.builtin import calculator, read_file

    manager = ToolManager()

    # Add two function tools
    manager.add_function_tool(calculator)
    manager.add_function_tool(read_file)

    # Register the tools
    manager.register_tools([calculator, read_file])

    # Create a minimal config and initialize tools
    config = {"provider": "anthropic"}
    await manager.initialize_tools(config)

    # Verify that tools were registered directly to the runtime registry
    assert "calculator" in manager.runtime_registry.tool_handlers
    assert "read_file" in manager.runtime_registry.tool_handlers
