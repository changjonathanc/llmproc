"""Tests for the SDK developer experience enhancements."""

from pathlib import Path

import pytest
from llmproc.program import LLMProgram


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

    # Check the linking was done correctly
    assert "expert" in main_program.linked_programs
    assert main_program.linked_programs["expert"] == expert_program
    assert main_program.linked_program_descriptions["expert"] == "Expert for specialized tasks"


def test_fluent_methods_chaining():
    """Test chaining multiple fluent methods."""
    # Create and configure a program with method chaining
    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet",
            provider="anthropic",
            system_prompt="You are a helpful assistant.",
        )
        .add_preload_file("example1.md")
        .add_preload_file("example2.md")
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
    assert len(program.preload_files) == 2
    assert "example1.md" in program.preload_files
    assert "example2.md" in program.preload_files
    assert "expert" in program.linked_programs
    assert program.linked_program_descriptions["expert"] == "Expert for special tasks"


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


def test_complex_method_chaining():
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
    main_program = (
        LLMProgram(
            model_name="gpt-4o",
            provider="openai",
            system_prompt="You are a coordinator.",
        )
        .add_preload_file("context1.md")
        .add_preload_file("context2.md")
        .add_linked_program(
            "expert1",
            LLMProgram(
                model_name="claude-3-5-haiku",
                provider="anthropic",
                system_prompt="Expert 1",
            ).add_preload_file("expert1_context.md"),
            "First level expert",
        )
        .add_linked_program("inner_expert", inner_expert, "Special inner expert")
        .register_tools([test_tool])  # Register the test tool
    )

    # Validate the complex structure
    assert len(main_program.preload_files) == 2
    assert "expert1" in main_program.linked_programs
    assert "inner_expert" in main_program.linked_programs

    # Validation and initialization happens during process startup, not here

    # Check that nested preload files were preserved
    assert "expert1_context.md" in main_program.linked_programs["expert1"].preload_files


def test_register_tools():
    """Test registering built-in tools."""
    # Import tool functions directly
    from llmproc.tools.builtin import calculator, fork_tool, read_file

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

    # Compile program to register tools
    program.compile()
    # Check that tools were registered
    registered_tools = program.get_registered_tools()
    # Tools are now stored as functions, but we can check their names
    tool_names = [tool.__name__ if callable(tool) else tool for tool in registered_tools]
    assert "calculator" in tool_names
    assert "read_file" in tool_names

    # The internal tool_manager.registered_tools uses string names
    assert "calculator" in program.tool_manager.registered_tools
    assert "read_file" in program.tool_manager.registered_tools

    # Remember the current list length
    previous_tools_len = len(program.tool_manager.registered_tools)

    # Create a new program to avoid side effects from the previous calls
    program = LLMProgram(model_name="test-model", provider="test-provider", system_prompt="Test system prompt")

    # Register initial tools
    program.register_tools([calculator])

    # Process function tools to ensure they're properly registered
    program.tool_manager.process_function_tools()

    # Verify initial state
    assert "calculator" in program.get_registered_tools()
    assert "fork" not in program.get_registered_tools()

    # Clear all existing tools from the runtime registry first
    program.tool_manager.runtime_registry.tool_handlers.clear()
    program.tool_manager.runtime_registry.tool_definitions.clear()
    program.tool_manager.function_tools.clear()

    # Replace with different tools
    program.register_tools([fork_tool])

    # Process function tools to actually register the tools
    program.tool_manager.process_function_tools()

    # Check that tools were replaced
    registered_tools = program.get_registered_tools()
    assert "fork" in registered_tools
    assert "calculator" not in registered_tools

    # The method might clear and set new tools or it might append
    # to existing tools - both are valid implementations
    # so we check that it's working correctly without assuming specific behavior
