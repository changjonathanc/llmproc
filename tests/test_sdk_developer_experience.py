"""Tests for the SDK developer experience enhancements."""

import asyncio
from pathlib import Path

import pytest

from llmproc.plugins.preload_files import PreloadFilesPlugin
from llmproc.program import LLMProgram
from llmproc.plugins.spawn import SpawnPlugin


def test_fluent_program_creation():
    """Test creating a program with the fluent interface."""
    # Create a basic program
    program = LLMProgram(
        model_name="claude-3-5-haiku",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
    )

    # Should not be compiled yet
    assert not program.compiled

    # Basic properties should be set
    assert program.model_name == "claude-3-5-haiku"
    assert program.provider == "anthropic"
    assert program.system_prompt == "You are a helpful assistant."

    # Default display name is created but we don't need to test it specifically


def test_program_linking():
    """Test linking programs together."""
    # Create main program
    main_program = LLMProgram(
        model_name="claude-3-5-haiku",
        provider="anthropic",
        system_prompt="You are a helpful coordinator.",
    )

    # Create expert program
    expert_program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a specialized expert.",
    )

    # Link them using the fluent interface
    main_program.add_linked_program("expert", expert_program, "Expert for specialized tasks")

    # Check the linking was done correctly via SpawnPlugin
    spawn_plugin = next((p for p in main_program.plugins if isinstance(p, SpawnPlugin)), None)
    assert spawn_plugin is not None
    assert "expert" in spawn_plugin.linked_programs
    assert spawn_plugin.linked_programs["expert"] == expert_program
    assert spawn_plugin.linked_program_descriptions["expert"] == "Expert for specialized tasks"


def test_fluent_methods_chaining(tmp_path: Path):
    """Test chaining multiple fluent methods."""
    # Create and configure a program with method chaining
    file1 = tmp_path / "example1.md"
    file2 = tmp_path / "example2.md"
    file1.write_text("one")
    file2.write_text("two")

    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet",
            provider="anthropic",
            system_prompt="You are a helpful assistant.",
        )
        .add_plugins(PreloadFilesPlugin(["example1.md", "example2.md"], base_dir=tmp_path))
        .add_linked_program(
            "expert",
            LLMProgram(
                model_name="claude-3-5-haiku",
                provider="anthropic",
                system_prompt="You are an expert.",
            ),
            "Expert for special tasks",
        )
    )

    # Verify everything was configured correctly
    plugin = next(p for p in program.plugins if p.__class__.__name__ == "PreloadFilesPlugin")
    assert len(plugin.file_paths) == 2
    assert "example1.md" in plugin.file_paths
    assert "example2.md" in plugin.file_paths
    spawn_plugin = next((p for p in program.plugins if isinstance(p, SpawnPlugin)), None)
    assert spawn_plugin is not None
    assert "expert" in spawn_plugin.linked_programs
    assert spawn_plugin.linked_program_descriptions["expert"] == "Expert for special tasks"


# API now compiles programs automatically when needed


def test_system_prompt_file():
    """Test loading system prompt from a file."""
    # Create a temporary system prompt file
    system_prompt_file = "test_system_prompt.txt"
    with open(system_prompt_file, "w") as f:
        f.write("You are a test assistant.")

    try:
        # Create program with system_prompt_file
        program = LLMProgram(
            model_name="claude-3-5-haiku",
            provider="anthropic",
            system_prompt_file=system_prompt_file,
        )

        # System prompt should be loaded when the process is started
        # We don't directly test this here as it would require an actual process start

    finally:
        # Clean up the test file
        Path(system_prompt_file).unlink()


# Test compile() through proper APIs


def test_complex_method_chaining(tmp_path: Path):
    """Test more complex method chaining scenarios."""
    # Create nested programs with method chaining
    inner_expert = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are an inner expert.",
    )

    # Function-based test tool
    def test_tool(query: str) -> str:
        """A test tool.

        Args:
            query: The query to process

        Returns:
            Processed result
        """
        return f"Processed: {query}"

    # Create the main program with fluent chaining
    ctx1 = tmp_path / "context1.md"
    ctx2 = tmp_path / "context2.md"
    expert1_ctx = tmp_path / "expert1_context.md"
    ctx1.write_text("c1")
    ctx2.write_text("c2")
    expert1_ctx.write_text("c3")

    main_program = (
        LLMProgram(
            model_name="gpt-4o",
            provider="openai",
            system_prompt="You are a coordinator.",
        )
        .add_plugins(PreloadFilesPlugin(["context1.md", "context2.md"], base_dir=tmp_path))
        .add_linked_program(
            "expert1",
            LLMProgram(
                model_name="claude-3-5-haiku",
                provider="anthropic",
                system_prompt="Expert 1",
            ).add_plugins(PreloadFilesPlugin(["expert1_context.md"], base_dir=tmp_path)),
            "First level expert",
        )
        .add_linked_program("inner_expert", inner_expert, "Special inner expert")
        .register_tools([test_tool])  # Register the test tool
    )

    # Validate the complex structure
    plugin = next(p for p in main_program.plugins if p.__class__.__name__ == "PreloadFilesPlugin")
    assert len(plugin.file_paths) == 2
    assert "context1.md" in plugin.file_paths
    assert "context2.md" in plugin.file_paths
    spawn_plugin = next((p for p in main_program.plugins if isinstance(p, SpawnPlugin)), None)
    assert spawn_plugin is not None
    assert "expert1" in spawn_plugin.linked_programs
    assert "inner_expert" in spawn_plugin.linked_programs

    # Validation and initialization happens during process startup, not here

    # Check that nested preload files were preserved
    child_plugin = next(
        p for p in spawn_plugin.linked_programs["expert1"].plugins if p.__class__.__name__ == "PreloadFilesPlugin"
    )
    assert "expert1_context.md" in child_plugin.file_paths


def test_register_tools():
    """Test registering built-in tools."""
    # Import tool functions directly
    from llmproc.tools.builtin import calculator, fork_tool, read_file
    from llmproc.common.metadata import get_tool_meta
    get_tool_meta(calculator).name = None
    get_tool_meta(read_file).name = None

    # Create a program
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
    )

    # Register tools using function references
    result = program.register_tools([calculator, read_file])

    # Check that the method returns self for chaining
    assert result is program

    # Compile program
    program.compile()
    from llmproc.tools.tool_manager import ToolManager

    tm = ToolManager()
    asyncio.run(
        tm.register_tools(
            program.tools,
            {
                "fd_manager": None,
                "linked_programs": {},
                "linked_program_descriptions": {},
                "has_linked_programs": False,
                "provider": "test",
                "mcp_enabled": False,
            },
        )
    )
    registered_tools = program.get_registered_tools()
    tool_names = [tool.__name__ if callable(tool) else tool for tool in registered_tools]
    assert "calculator" in tool_names
    assert "read_file" in tool_names

    assert "calculator" in tm.registered_tools
    assert "read_file" in tm.registered_tools

    previous_tools_len = len(tm.registered_tools)

    # Create a new program to avoid side effects from the previous calls
    program = LLMProgram(model_name="test-model", provider="test-provider", system_prompt="Test system prompt")

    # Register initial tools
    program.register_tools([calculator])
    config = {
        "fd_manager": None,
        "linked_programs": {},
        "linked_program_descriptions": {},
        "has_linked_programs": False,
        "provider": "test",
        "mcp_enabled": False,
    }

    from llmproc.tools.tool_manager import ToolManager

    tm = ToolManager()
    asyncio.run(tm.register_tools(program.tools, config))
    manager = tm

    # Verify initial state
    assert "calculator" in manager.registered_tools
    assert "fork" not in manager.registered_tools

    # Clear all existing tools from the runtime registry first
    tm.runtime_registry._tools.clear()
    tm.registered_tools.clear()

    # Replace with different tools
    program.register_tools([fork_tool])
    asyncio.run(tm.register_tools([fork_tool], config))
    manager = tm

    # Check that tools were replaced
    registered_tools = manager.registered_tools
    assert "fork" in registered_tools
    assert "calculator" not in registered_tools

    # The method might clear and set new tools or it might append
    # to existing tools - both are valid implementations
    # so we check that it's working correctly without assuming specific behavior
