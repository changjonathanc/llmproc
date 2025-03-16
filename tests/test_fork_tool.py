"""Tests for the fork system call."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.fork import fork_tool


class TestForkTool:
    """Test the fork system call."""

    def test_fork_registration(self):
        """Test that the fork tool is properly registered."""
        # Create a minimal program with fork tool enabled
        program = LLMProgram(
            model_name="test-model",
            provider="anthropic",
            system_prompt="Test system prompt",
            tools={"enabled": ["fork"]}
        )
        
        # Create a process
        process = LLMProcess(program=program)
        
        # Check that fork tool is registered
        assert any(tool["name"] == "fork" for tool in process.tools)
        assert "fork" in process.tool_handlers
        
    @pytest.mark.asyncio
    async def test_fork_process_method(self):
        """Test the fork_process method creates a proper copy."""
        # Create a minimal program
        program = LLMProgram(
            model_name="test-model",
            provider="anthropic",
            system_prompt="Test system prompt"
        )
        
        # Create a process with some state
        process = LLMProcess(program=program)
        process.state = [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        process.preloaded_content = {"test.txt": "Test content"}
        process.enriched_system_prompt = "Enriched prompt with content"
        
        # Fork the process
        forked = await process.fork_process()
        
        # Check that it's a new instance
        assert forked is not process
        
        # Check that state was copied
        assert forked.state == process.state
        assert id(forked.state) != id(process.state)  # Different objects
        
        # Check that preloaded content was copied
        assert forked.preloaded_content == process.preloaded_content
        assert id(forked.preloaded_content) != id(process.preloaded_content)  # Different objects
        
        # Check that enriched system prompt was copied
        assert forked.enriched_system_prompt == process.enriched_system_prompt
        
        # Modify the original to confirm they're independent
        process.state.append({"role": "user", "content": "New message"})
        assert len(forked.state) == 3  # Still has original length
        
    @pytest.mark.asyncio
    async def test_fork_tool_function(self):
        """Test the fork_tool function itself."""
        # Create a mock process
        mock_process = MagicMock()
        mock_process.fork_process = AsyncMock()
        mock_forked_process = MagicMock()
        mock_forked_process.run = AsyncMock(return_value="Forked response")
        mock_process.fork_process.return_value = mock_forked_process
        
        # Call the fork tool
        result = await fork_tool(
            prompts=["Task 1", "Task 2"],
            llm_process=mock_process
        )
        
        # Check that fork_process was called twice
        assert mock_process.fork_process.call_count == 2
        
        # Check that run was called on each forked process
        assert mock_forked_process.run.call_count == 2
        
        # Check the result format
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == 0
        assert result["results"][0]["message"] == "Forked response"
        assert result["results"][1]["id"] == 1
        assert result["results"][1]["message"] == "Forked response"

    @pytest.mark.asyncio
    async def test_fork_tool_error_handling(self):
        """Test error handling in the fork tool."""
        # Call without a process
        result = await fork_tool(prompts=["Test"], llm_process=None)
        assert "error" in result
        assert result["is_error"] is True
        
        # Call with a process that raises an exception
        mock_process = MagicMock()
        mock_process.fork_process = AsyncMock(side_effect=Exception("Test error"))
        
        result = await fork_tool(prompts=["Test"], llm_process=mock_process)
        assert "error" in result
        assert result["is_error"] is True
        assert "Test error" in result["error"]


# API tests that require real API keys
@pytest.mark.llm_api
class TestForkToolWithAPI:
    """Test the fork system call with real API calls."""
    
    @pytest.mark.asyncio
    async def test_fork_with_real_api(self):
        """Test the fork tool with actual API calls."""
        # Only run this test if we have the API key
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not available")
            
        # Create a program from the example file
        from pathlib import Path
        example_path = Path(__file__).parents[1] / "examples" / "fork.toml"
        
        program = LLMProgram.compile(example_path)
        process = LLMProcess(program=program)
        
        # Run a test query
        response = await process.run(
            "Fork yourself to perform these two tasks in parallel: "
            "1. Count from 1 to 5. "
            "2. List the first 5 letters of the alphabet."
        )
        
        # Check that the response includes both tasks' results
        assert any(word in response.lower() for word in ["1", "2", "3", "4", "5"])
        assert any(letter in response.lower() for letter in ["a", "b", "c", "d", "e"])