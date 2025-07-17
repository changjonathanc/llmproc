"""Test for the builtin tools mapping and schema modifier functionality."""

import warnings
import asyncio

import pytest
from llmproc.program import LLMProgram
from llmproc.tools.builtin import BUILTIN_TOOLS
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
    # goto is now provided by MessageIDPlugin, not builtin tools
    # assert "goto" in BUILTIN_TOOLS

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

    # Test that create_schema_from_callable applies the modifier
    from llmproc.tools.function_tools import (
        create_handler_from_function,
        create_schema_from_callable,
    )

    # Without config - should not apply modifier
    handler1 = create_handler_from_function(test_function)
    schema1 = create_schema_from_callable(handler1)
    assert "modified" not in schema1["description"]

    # With config - should apply modifier
    test_config = {"some_data": "test"}
    handler2 = create_handler_from_function(test_function)
    schema2 = create_schema_from_callable(handler2, test_config)
    assert "modified" in schema2["description"]


def test_spawn_tool_schema_modifier():
    """Test that spawn tool's schema modifier is applied."""
    # Verify spawn tool has schema_modifier in metadata
    from llmproc.common.metadata import get_tool_meta
    from llmproc.plugins.spawn import modify_spawn_schema, spawn_tool

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

    # Test schema modification via create_schema_from_callable
    from llmproc.tools.function_tools import (
        create_handler_from_function,
        create_schema_from_callable,
    )

    handler = create_handler_from_function(spawn_tool)
    schema = create_schema_from_callable(handler, test_config)
    assert "Available Programs" in schema["description"]
    assert "prog1" in schema["description"]


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

            from llmproc.tools.tool_manager import ToolManager

            tm = ToolManager()
            asyncio.run(tm.register_tools(program.tools, {}))
            registered = tm.runtime_registry.get_tool_names()

            assert "calculator" in registered
            assert "read_file" in registered

            # Initialize tools directly
            config = {
                "fd_manager": None,
                "linked_programs": {},
                "linked_program_descriptions": {},
                "has_linked_programs": False,
                "provider": "test",
                "mcp_enabled": False,
            }
            asyncio.run(tm.register_tools(program.tools, config))
            assert "calculator" in tm.runtime_registry.get_tool_names()
            assert "read_file" in tm.runtime_registry.get_tool_names()

    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


def test_program_validates_tool_dependencies():
    """Test that program compilation validates tool dependencies."""
    from llmproc.tools.builtin import spawn_tool

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

    # Test FD tools are provided by plugin, not manual registration
    from llmproc.plugins.file_descriptor import FileDescriptorPlugin
    from llmproc.config.schema import FileDescriptorPluginConfig

    program2 = LLMProgram(
        model_name="test-model",
        provider="anthropic",
        system_prompt="Test prompt",
    )
    # FD tools now come from plugin only
    program2.add_plugins(FileDescriptorPlugin(FileDescriptorPluginConfig()))
    program2.compile()
    assert program2.compiled

    # Test that plugin provides tools automatically
    program3 = LLMProgram(model_name="test-model", provider="anthropic", system_prompt="Test prompt")

    # Configure FD plugin - tools are provided automatically
    program3.add_plugins(FileDescriptorPlugin(FileDescriptorPluginConfig()))
    # No need to manually register FD tools anymore

    # Should compile without errors
    program3.compile()
    assert program3.compiled


@pytest.mark.asyncio
async def test_direct_tool_registration():
    """Test that register_tools directly registers tools without using builtin registry."""
    from llmproc.tools import ToolManager
    from llmproc.tools.builtin import calculator, read_file

    manager = ToolManager()

    # Register the tools directly
    config = {"provider": "anthropic"}
    await manager.register_tools([calculator, read_file], config)

    # Verify that tools were registered directly to the runtime registry
    assert "calculator" in manager.runtime_registry.get_tool_names()
    assert "read_file" in manager.runtime_registry.get_tool_names()


def test_add_builtin_tool_and_lookup():
    """Libraries can extend BUILTIN_TOOLS and use names in configuration."""
    from llmproc.tools.builtin import add_builtin_tool, BUILTIN_TOOLS
    from llmproc.tools.utils import convert_to_callables
    from llmproc.tools.function_tools import register_tool

    @register_tool(name="custom_builtin")
    def custom_builtin() -> str:
        return "ok"

    add_builtin_tool("custom_builtin", custom_builtin)

    assert BUILTIN_TOOLS["custom_builtin"] is custom_builtin

    funcs = convert_to_callables(["custom_builtin"])
    assert funcs == [custom_builtin]

    with pytest.raises(ValueError):
        add_builtin_tool("custom_builtin", custom_builtin)
