"""Tests for MCP timeout handling.

This module contains tests to verify that MCP timeouts are handled correctly,
including both tool call timeouts and resource cleanup timeouts.
"""

import asyncio
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from llmproc.llm_process import LLMProcess
from llmproc.mcp_registry.compound import MCPAggregator, _PersistentClient
from llmproc.tools.mcp.manager import MCPManager
from llmproc.tools.mcp.constants import (
    MCP_DEFAULT_TOOL_FETCH_TIMEOUT,
    MCP_ERROR_TOOL_FETCH_TIMEOUT,
    MCP_MAX_FETCH_RETRIES
)


@pytest.mark.unit
class TestMCPTimeoutHandling:
    """Test suite for MCP timeout handling."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry for testing."""
        mock = MagicMock()
        mock.list_servers.return_value = ["test_server"]
        return mock

    @pytest.fixture
    def mock_aggregator(self, mock_registry):
        """Create a mock aggregator for testing."""
        mock = AsyncMock()
        mock.registry = mock_registry
        mock._get_or_create_client = AsyncMock()
        mock.transient = False
        return mock

    @pytest.fixture
    def mcp_manager(self, mock_aggregator):
        """Create an MCPManager with a mock aggregator."""
        manager = MCPManager(servers={"test_server": {}})
        manager.aggregator = mock_aggregator
        manager.initialized = True
        manager.mcp_tools = [MagicMock(server="test_server", tools="all")]
        return manager

    @pytest.mark.asyncio
    async def test_get_tool_registrations_timeout_retry(self, mcp_manager, mock_aggregator):
        """Test that get_tool_registrations retries on timeout."""
        # Mock gathering
        with patch.dict(os.environ, {"LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT": "false"}), \
             patch('asyncio.gather', new=AsyncMock(return_value=[])) as mock_gather, \
             patch('logging.Logger.warning') as mock_warning:
            
            # Call with shorter timeout for faster test
            result = await mcp_manager.get_tool_registrations(tool_fetch_timeout=0.1)
            
            # Verify the function was called
            assert mock_gather.called
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_tool_registrations_timeout_exhausted(self, mcp_manager, mock_aggregator):
        """Test behavior when all retries are exhausted."""
        # Mock timeout error in asyncio.gather
        with patch.dict(os.environ, {"LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT": "false"}), \
             patch('asyncio.gather', new=AsyncMock(side_effect=asyncio.TimeoutError())) as mock_gather, \
             patch('logging.Logger.error') as mock_error:
            
            # Call with shorter timeout for faster test
            result = await mcp_manager.get_tool_registrations(tool_fetch_timeout=0.1)
            
            # Verify error was logged
            mock_error.assert_called()
            
            # Verify empty result
            assert isinstance(result, list)
            assert len(result) == 0  # No tools returned due to timeout

    @pytest.mark.asyncio
    async def test_get_tool_registrations_custom_timeout(self, mcp_manager):
        """Test that custom timeout is used."""
        custom_timeout = 45.0
        
        with patch.dict(os.environ, {"LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT": "false"}), \
             patch('asyncio.timeout') as mock_timeout, \
             patch('asyncio.gather', new=AsyncMock(return_value=[])):
            await mcp_manager.get_tool_registrations(tool_fetch_timeout=custom_timeout)
            
            # Verify timeout was created with custom value
            mock_timeout.assert_called_with(custom_timeout)

    @pytest.mark.asyncio
    async def test_get_tool_registrations_env_var_timeout(self, mcp_manager):
        """Test that environment variable timeout is used."""
        env_timeout = 50.0
        
        with patch.dict(os.environ, {
            'LLMPROC_TOOL_FETCH_TIMEOUT': str(env_timeout),
            'LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT': 'false'
        }), \
             patch('asyncio.timeout') as mock_timeout, \
             patch('asyncio.gather', new=AsyncMock(return_value=[])):
            await mcp_manager.get_tool_registrations()
            
            # Verify timeout was created with env var value
            mock_timeout.assert_called_with(env_timeout)

    @pytest.mark.asyncio
    async def test_global_timeout_error_message(self, mcp_manager):
        """Test that global timeout error produces appropriate message."""
        with patch.dict(os.environ, {"LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT": "false"}), \
             patch('asyncio.timeout') as mock_timeout, \
             patch('logging.Logger.error') as mock_error:
            
            # Make asyncio.timeout context manager raise TimeoutError
            mock_cm = MagicMock()
            mock_cm.__aenter__.return_value = None
            mock_cm.__aexit__.side_effect = asyncio.TimeoutError()
            mock_timeout.return_value = mock_cm
            
            # Set mcp_tools to have a server to avoid an early return
            mcp_manager.mcp_tools = [MagicMock(server="test_server")]
            
            await mcp_manager.get_tool_registrations(tool_fetch_timeout=30.0)
            
            # Verify error message contains timeout value and env var name
            called = False
            for call in mock_error.call_args_list:
                if len(call[0]) > 0 and isinstance(call[0][0], str):
                    error_msg = call[0][0]
                    if "30.0 seconds" in error_msg and "LLMPROC_TOOL_FETCH_TIMEOUT" in error_msg:
                        called = True
                        break
            
            assert called, "Did not find expected timeout error message with timeout value and env var name"


@pytest.mark.unit
class TestMCPCleanupTimeoutHandling:
    """Test suite for MCP cleanup timeout handling."""
    
    @pytest.mark.asyncio
    async def test_llmprocess_aclose_with_timeout(self):
        """Test that LLMProcess.aclose applies a timeout when closing MCP clients."""
        # Mock aggregator that hangs on close_clients
        mock_aggregator = AsyncMock()
        mock_aggregator.close_clients = AsyncMock(side_effect=asyncio.sleep(10))
        
        # Mock tool_manager with mcp_manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.mcp_manager = MagicMock()
        mock_tool_manager.mcp_manager.aggregator = mock_aggregator
        
        # Create LLMProcess with mocked components
        process = LLMProcess(
            program=MagicMock(),
            model_name="test-model",
            provider="test-provider",
            original_system_prompt="test",
            system_prompt="test",
            tool_manager=mock_tool_manager
        )
        
        # Call aclose with a short timeout - should not hang
        start_time = asyncio.get_event_loop().time()
        await process.aclose(timeout=0.5)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # The close_clients should have been called
        mock_aggregator.close_clients.assert_called_once()
        
        # The function should return quickly due to timeout
        assert elapsed < 1.0, "aclose should timeout and return quickly"
    
    @pytest.mark.asyncio
    async def test_mcp_aggregator_close_clients_timeout(self):
        """Test that MCPAggregator.close_clients applies a timeout per client."""
        # Create a mock persistent client that hangs on close
        mock_client = AsyncMock()
        mock_client.close = AsyncMock(side_effect=asyncio.sleep(10))
        
        # Create MCPAggregator with mocked client
        aggregator = MCPAggregator(MagicMock())
        aggregator.transient = False
        aggregator._client_cms = {"test-server": mock_client}
        
        # Call close_clients with a short timeout - should not hang
        start_time = asyncio.get_event_loop().time()
        await aggregator.close_clients(client_timeout=0.5)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # The client.close should have been called
        mock_client.close.assert_called_once()
        
        # The function should return quickly due to timeout
        assert elapsed < 1.0, "close_clients should timeout and return quickly"
    
    @pytest.mark.asyncio
    async def test_persistent_client_close_timeout(self):
        """Test that _PersistentClient.close applies a timeout."""
        # Create a mock task that never completes
        mock_task = asyncio.create_task(asyncio.sleep(10))
        
        # Create _PersistentClient with mocked task
        client = _PersistentClient(MagicMock())
        client._task = mock_task
        client._stop = asyncio.Event()
        
        # Call close with a short timeout - should not hang
        start_time = asyncio.get_event_loop().time()
        await client.close(timeout=0.5)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # The function should return quickly due to timeout
        assert elapsed < 1.0, "client.close should timeout and return quickly"
        
        # Clean up
        mock_task.cancel()
        try:
            await asyncio.wait_for(mock_task, timeout=0.1)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
