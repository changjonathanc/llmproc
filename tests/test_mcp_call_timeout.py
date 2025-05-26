"""Tests for MCP tool call timeout handling.

This module contains tests to verify that MCP tool call timeouts are handled correctly
and proper error messages are returned.
"""

import asyncio
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from llmproc.mcp_registry.compound import MCPAggregator, ServerRegistry, MCPServerSettings
from llmproc.tools.mcp.constants import MCP_DEFAULT_TOOL_CALL_TIMEOUT, MCP_ERROR_TOOL_CALL_TIMEOUT


@pytest.mark.unit
class TestMCPCallTimeout:
    """Test suite for MCP tool call timeout handling."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry with test servers."""
        servers = {
            "stdio_server": MCPServerSettings(
                type="stdio",
                command="test_command",
                args=["arg1", "arg2"],
                description="Test stdio server"
            ),
            "sse_server": MCPServerSettings(
                type="sse",
                url="https://test-server.example",
                description="Test SSE server"
            )
        }
        registry = MagicMock(spec=ServerRegistry)
        registry.registry = servers
        registry.list_servers.return_value = list(servers.keys())
        return registry

    @pytest.fixture
    def aggregator(self, mock_registry):
        """Create an MCPAggregator with a mock registry."""
        return MCPAggregator(mock_registry)

    @pytest.mark.asyncio
    async def test_call_tool_timeout_default_message(self, aggregator):
        """Test timeout error message for tool call."""
        # Setup mock client that raises TimeoutError
        mock_client = AsyncMock()
        with patch.object(aggregator, '_get_or_create_client', return_value=mock_client), \
             patch('asyncio.timeout') as mock_timeout, \
             patch('logging.Logger.error') as mock_error:
            
            # Make asyncio.timeout context manager raise TimeoutError
            mock_cm = MagicMock()
            mock_cm.__aenter__.return_value = None
            mock_cm.__aexit__.side_effect = asyncio.TimeoutError()
            mock_timeout.return_value = mock_cm
            
            # Call tool that will timeout
            result = await aggregator.call_tool("stdio_server__test_tool", {"arg": "value"})
            
            # Verify error message
            assert result.isError is True
            assert "timeout" in result.message.lower()
            assert "stdio_server" in result.message
            assert "test_tool" in result.message
            assert str(MCP_DEFAULT_TOOL_CALL_TIMEOUT) in result.message
            
            # Since there are multiple error log calls, check for our specific one
            found_timeout_log = False
            for call in mock_error.call_args_list:
                if len(call[0]) > 0 and isinstance(call[0][0], str):
                    error_msg = call[0][0]
                    if "timeout" in error_msg.lower() and "stdio_server" in error_msg and "test_tool" in error_msg:
                        found_timeout_log = True
                        break
            
            assert found_timeout_log, "Did not find expected timeout error log message"

    @pytest.mark.asyncio
    async def test_call_tool_timeout_custom_env(self, aggregator):
        """Test timeout error message with custom environment variable."""
        custom_timeout = 45.0
        
        # Setup mock client that raises TimeoutError
        mock_client = AsyncMock()
        with patch.dict(os.environ, {'LLMPROC_TOOL_CALL_TIMEOUT': str(custom_timeout)}), \
             patch.object(aggregator, '_get_or_create_client', return_value=mock_client), \
             patch('asyncio.timeout') as mock_timeout:
            
            # Make asyncio.timeout context manager raise TimeoutError
            mock_cm = MagicMock()
            mock_cm.__aenter__.return_value = None
            mock_cm.__aexit__.side_effect = asyncio.TimeoutError()
            mock_timeout.return_value = mock_cm
            
            # Call tool that will timeout
            result = await aggregator.call_tool("stdio_server__test_tool", {"arg": "value"})
            
            # Verify timeout is set from environment variable
            mock_timeout.assert_called_with(custom_timeout)
            
            # Verify error message includes custom timeout
            assert str(custom_timeout) in result.message

    @pytest.mark.asyncio
    async def test_call_tool_timeout_with_server_info(self, aggregator):
        """Test that timeout error includes server information."""
        # Test with stdio server
        with patch.object(aggregator, '_get_or_create_client'), \
             patch('asyncio.timeout') as mock_timeout:
            
            # Make asyncio.timeout context manager raise TimeoutError
            mock_cm = MagicMock()
            mock_cm.__aenter__.return_value = None
            mock_cm.__aexit__.side_effect = asyncio.TimeoutError()
            mock_timeout.return_value = mock_cm
            
            # Call tool on stdio server
            result = await aggregator.call_tool("stdio_server__test_tool")
            
            # Verify error includes server info
            assert "Server type: stdio" in result.message
            assert "Command: test_command" in result.message
        
        # Test with SSE server
        with patch.object(aggregator, '_get_or_create_client'), \
             patch('asyncio.timeout') as mock_timeout:
            
            # Make asyncio.timeout context manager raise TimeoutError
            mock_cm = MagicMock()
            mock_cm.__aenter__.return_value = None
            mock_cm.__aexit__.side_effect = asyncio.TimeoutError()
            mock_timeout.return_value = mock_cm
            
            # Call tool on SSE server
            result = await aggregator.call_tool("sse_server__test_tool")
            
            # Verify error includes server info
            assert "Server type: sse" in result.message
            assert "URL: https://test-server.example" in result.message