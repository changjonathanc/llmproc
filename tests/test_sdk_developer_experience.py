"""Tests for the SDK developer experience enhancements."""

import pytest
from pathlib import Path
from llmproc.program import LLMProgram


def test_fluent_program_creation():
    """Test creating a program with the fluent interface."""
    # Create a basic program
    program = LLMProgram(
        model_name="claude-3-5-haiku",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    
    # Should not be compiled yet
    assert not program.compiled
    
    # Basic properties should be set
    assert program.model_name == "claude-3-5-haiku"
    assert program.provider == "anthropic"
    assert program.system_prompt == "You are a helpful assistant."
    
    # Default display name should be created
    assert program.display_name == "Anthropic claude-3-5-haiku"


def test_program_linking():
    """Test linking programs together."""
    # Create main program
    main_program = LLMProgram(
        model_name="claude-3-5-haiku",
        provider="anthropic",
        system_prompt="You are a helpful coordinator."
    )
    
    # Create expert program
    expert_program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a specialized expert."
    )
    
    # Link them using the fluent interface
    main_program.add_linked_program(
        "expert", 
        expert_program, 
        "Expert for specialized tasks"
    )
    
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
            system_prompt="You are a helpful assistant."
        )
        .add_preload_file("example1.md")
        .add_preload_file("example2.md")
        .add_linked_program(
            "expert",
            LLMProgram(
                model_name="claude-3-5-haiku",
                provider="anthropic",
                system_prompt="You are an expert."
            ),
            "Expert for special tasks"
        )
    )
    
    # Verify everything was configured correctly
    assert len(program.preload_files) == 2
    assert "example1.md" in program.preload_files
    assert "example2.md" in program.preload_files
    assert "expert" in program.linked_programs
    assert program.linked_program_descriptions["expert"] == "Expert for special tasks"


def test_compile_method():
    """Test the instance-level validate method."""
    # Create a program
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    
    # Compile it and check the result is self (for chaining)
    result = program.compile()
    assert result is program
    assert program.compiled
    
    # Compiling again should be a no-op
    program.compile()
    assert program.compiled


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
            system_prompt_file=system_prompt_file
        )
        
        # System prompt should not be loaded yet
        assert program.system_prompt is None
        
        # After compilation, system prompt should be loaded
        program.compile()
        assert program.system_prompt == "You are a test assistant."
        
    finally:
        # Clean up the test file
        Path(system_prompt_file).unlink()


def test_recursive_program_compilation():
    """Test that linked programs are recursively compiled."""
    # Create a main program with a linked program
    expert = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are an expert."
    )
    
    main = LLMProgram(
        model_name="claude-3-5-haiku",
        provider="anthropic",
        system_prompt="You are a coordinator.",
        linked_programs={"expert": expert}
    )
    
    # Compile the main program
    main.compile()
    
    # Both main and expert should be compiled
    assert main.compiled
    assert expert.compiled


def test_complex_method_chaining():
    """Test more complex method chaining scenarios."""
    # Create nested programs with method chaining
    inner_expert = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are an inner expert."
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
            system_prompt="You are a coordinator."
        )
        .add_preload_file("context1.md")
        .add_preload_file("context2.md")
        .add_linked_program(
            "expert1", 
            LLMProgram(
                model_name="claude-3-5-haiku",
                provider="anthropic",
                system_prompt="Expert 1"
            ).add_preload_file("expert1_context.md"),
            "First level expert"
        )
        .add_linked_program("inner_expert", inner_expert, "Special inner expert")
        .add_tool(test_tool)
    )
    
    # Validate the complex structure
    assert len(main_program.preload_files) == 2
    assert "expert1" in main_program.linked_programs
    assert "inner_expert" in main_program.linked_programs
    
    # Compile the program
    compiled = main_program.compile()
    
    # Should return self for chaining
    assert compiled is main_program
    
    # Check that linked programs were also compiled
    assert main_program.linked_programs["expert1"].compiled
    assert main_program.linked_programs["inner_expert"].compiled
    
    # Check that nested preload files were preserved
    assert "expert1_context.md" in main_program.linked_programs["expert1"].preload_files


def test_set_enabled_tools():
    """Test setting enabled built-in tools."""
    # Create a program
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    
    # Set enabled tools
    result = program.set_enabled_tools(["calculator", "read_file"])
    
    # Check that the method returns self for chaining
    assert result is program
    
    # Check that tools were enabled
    assert "calculator" in program.tools["enabled"]
    assert "read_file" in program.tools["enabled"]
    assert "calculator" in program.tool_manager.enabled_tools
    assert "read_file" in program.tool_manager.enabled_tools
    
    # Try replacing with different tools
    program.set_enabled_tools(["calculator", "fork"])
    
    # Check that tools list was replaced (not appended to)
    assert len(program.tools["enabled"]) == 2
    assert "calculator" in program.tools["enabled"]
    assert "fork" in program.tools["enabled"]
    assert "read_file" not in program.tools["enabled"]


def test_error_handling_in_fluent_api():
    """Test error handling in the fluent API."""
    # Test missing system prompt
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        # No system_prompt provided
    )
    
    # Should raise ValueError during compilation
    with pytest.raises(ValueError) as excinfo:
        program.compile()
    
    # Check error message
    assert "system_prompt" in str(excinfo.value)
    
    # Test system prompt file that doesn't exist
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt_file="non_existent_file.md"
    )
    
    # Should raise FileNotFoundError during compilation
    with pytest.raises(FileNotFoundError) as excinfo:
        program.compile()
    
    # Check error message
    assert "System prompt file not found" in str(excinfo.value)