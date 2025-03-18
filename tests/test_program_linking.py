"""Tests for program linking functionality."""

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc.llm_process import LLMProcess
from llmproc.tools.spawn import spawn_tool


class TestProgramLinking:
    """Test program linking functionality."""
    
    def test_program_linking_compilation(self):
        """Test compilation of linked programs using LLMProgram.compile."""
        # Create a temporary directory for test files
        tmp_dir = Path("tmp_test_linked")
        tmp_dir.mkdir(exist_ok=True)
        
        try:
            # Create a test file path
            expert_toml = tmp_dir / "expert.toml"
            with open(expert_toml, "w") as f:
                f.write("""
                [model]
                name = "expert-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "Expert prompt"
                """)
                
            # Create a main toml that links to the expert
            main_toml = tmp_dir / "main.toml"
            with open(main_toml, "w") as f:
                f.write(f"""
                [model]
                name = "main-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "Main prompt"
                
                [linked_programs]
                expert = "{expert_toml.name}"
                """)
                
            # Mock the client creation to avoid API calls
            with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                
                # Test with direct compilation
                from llmproc.program import LLMProgram
                
                with patch("llmproc.program.LLMProgram.compile", wraps=LLMProgram.compile) as mock_compile:
                    # Compile the main program with linked programs
                    main_program = LLMProgram.compile(main_toml, include_linked=True)
                    
                    # Verify the compilation worked - now linked_programs contains Program objects
                    assert hasattr(main_program, 'linked_programs')
                    assert "expert" in main_program.linked_programs
                    
                    # Create a process from the program
                    process = LLMProcess(program=main_program)
                    
                    # Verify the process has the linked program
                    assert process.has_linked_programs
                    assert "expert" in process.linked_programs
                    
                    # Verify compile was called
                    mock_compile.assert_called_with(main_toml, include_linked=True)
        
        finally:
            # Clean up test files
            for file_path in [expert_toml, main_toml]:
                if file_path.exists():
                    file_path.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()
    
    def test_register_spawn_tool(self):
        """Test registration of spawn tool."""
        # Mock the client creation to avoid API calls
        with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            # Create a process with linked programs
            from llmproc.program import LLMProgram
            from llmproc.tools import register_spawn_tool
            
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
                tools={"enabled": ["spawn"]}
            )
            process = LLMProcess(
                program=program,
                linked_programs_instances={"expert": MagicMock()}
            )
            
            # Registry should already contain the spawn tool from initialization
            # but we'll register it directly for testing
            register_spawn_tool(process.tool_registry, process)
        
        # Check that the tool was registered
        assert len(process.tools) >= 1
        assert any(tool["name"] == "spawn" for tool in process.tools)
        assert "spawn" in process.tool_handlers
        assert process.tools[0]["name"] == "spawn"
        assert "input_schema" in process.tools[0]
        # Handler is stored separately in tool_handlers
        assert "spawn" in process.tool_handlers
    
    @pytest.mark.asyncio
    async def test_spawn_tool_functionality(self):
        """Test the functionality of the spawn tool."""
        # Create mock linked program
        mock_expert = MagicMock()
        mock_expert.run = AsyncMock(return_value="Expert response")
        
        # Mock the client creation to avoid API calls
        with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            # Create a process with linked programs
            from llmproc.program import LLMProgram
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt"
            )
            process = LLMProcess(
                program=program,
                linked_programs_instances={"expert": mock_expert}
            )
        
        # Test the spawn tool
        result = await spawn_tool(
            program_name="expert",
            query="Test query",
            llm_process=process
        )
        
        # Check the result
        assert result["program"] == "expert"
        assert result["query"] == "Test query"
        assert result["response"] == "Expert response"
        mock_expert.run.assert_called_once_with("Test query")
    
    @pytest.mark.asyncio
    async def test_spawn_tool_error_handling(self):
        """Test error handling in the spawn tool."""
        # Mock the client creation to avoid API calls
        with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            
            # Create a process without linked programs
            from llmproc.program import LLMProgram
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt"
            )
            process = LLMProcess(program=program)
        
        # Test with missing linked program
        result = await spawn_tool(
            program_name="nonexistent",
            query="Test query",
            llm_process=process
        )
        
        # Check that an error was returned
        assert "error" in result
        assert result["is_error"] is True
        assert "not found" in result["error"]
        
        # Test with exception in linked program
        mock_expert = MagicMock()
        mock_expert.run = AsyncMock(side_effect=Exception("Test error"))
        process.linked_programs = {"expert": mock_expert}
        process.has_linked_programs = True
        
        result = await spawn_tool(
            program_name="expert",
            query="Test query",
            llm_process=process
        )
        
        # Check that an error was returned
        assert "error" in result
        assert result["is_error"] is True
        assert "Test error" in result["error"]