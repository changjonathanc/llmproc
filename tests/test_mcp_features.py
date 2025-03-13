"""Tests for the MCP (Model Context Protocol) feature."""

import os
import json
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from llmproc import LLMProcess


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    original_env = os.environ.copy()
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    os.environ["GITHUB_TOKEN"] = "test-github-token"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_mcp_registry():
    """Mock the MCP registry and related components."""
    # Create MCP registry module mock
    mock_mcp_registry = MagicMock()
    
    # Setup mocks for MCP components
    mock_server_registry = MagicMock()
    mock_server_registry_class = MagicMock()
    mock_server_registry_class.from_config.return_value = mock_server_registry
    
    mock_aggregator = MagicMock()
    mock_aggregator_class = MagicMock()
    mock_aggregator_class.return_value = mock_aggregator
    
    # Create mock tools
    mock_tool1 = MagicMock()
    mock_tool1.name = "github.search_repositories"
    mock_tool1.description = "Search for GitHub repositories"
    mock_tool1.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}
    
    mock_tool2 = MagicMock()
    mock_tool2.name = "github.get_file_contents"
    mock_tool2.description = "Get file contents from a GitHub repository"
    mock_tool2.inputSchema = {
        "type": "object", 
        "properties": {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "path": {"type": "string"}
        }
    }
    
    mock_tool3 = MagicMock()
    mock_tool3.name = "codemcp.ReadFile"
    mock_tool3.description = "Read a file from the filesystem"
    mock_tool3.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}
    
    # Setup tool calls
    mock_tools_result = MagicMock()
    mock_tools_result.tools = [mock_tool1, mock_tool2, mock_tool3]
    mock_aggregator.list_tools = AsyncMock(return_value=mock_tools_result)
    
    # Setup tool results
    mock_tool_result = MagicMock()
    mock_tool_result.content = "Tool call result"
    mock_tool_result.isError = False
    mock_aggregator.call_tool = AsyncMock(return_value=mock_tool_result)
    
    # Create patches for the mcp_registry module
    with patch.dict('sys.modules', {
        'mcp_registry': mock_mcp_registry,
    }):
        # Set attributes on the mock module
        mock_mcp_registry.ServerRegistry = mock_server_registry_class
        mock_mcp_registry.MCPAggregator = mock_aggregator_class
        mock_mcp_registry.get_config_path = MagicMock(return_value='/mock/config/path')
        
        yield


@pytest.fixture
def mcp_config_file():
    """Create a temporary MCP config file."""
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump({
            "mcpServers": {
                "github": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}
                },
                "codemcp": {
                    "type": "stdio",
                    "command": "/bin/zsh",
                    "args": ["-c", "uvx --from git+https://github.com/cccntu/codemcp@main codemcp "]
                }
            }
        }, temp_file)
        temp_path = temp_file.name
    
    yield temp_path
    os.unlink(temp_path)


@patch("llmproc.llm_process.HAS_MCP", True)
@patch("llmproc.llm_process.asyncio.run")
@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.Anthropic")
def test_mcp_initialization(mock_anthropic, mock_asyncio_run, mock_mcp_registry, mock_env, mcp_config_file):
    """Test that LLMProcess initializes correctly with MCP configuration."""
    # Setup mock client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # Create LLMProcess with MCP configuration
    process = LLMProcess(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        mcp_config_path=mcp_config_file,
        mcp_tools={"github": ["search_repositories"], "codemcp": ["ReadFile"]}
    )
    
    # Check that MCP was initialized
    assert process.mcp_enabled is True
    assert process.mcp_config_path == mcp_config_file
    assert process.mcp_tools == {"github": ["search_repositories"], "codemcp": ["ReadFile"]}
    
    # Verify asyncio.run was called to initialize MCP tools
    mock_asyncio_run.assert_called_once()


@patch("llmproc.llm_process.HAS_MCP", True)
def test_from_toml_with_mcp(mock_mcp_registry, mock_env, mcp_config_file):
    """Test loading from a TOML configuration with MCP settings."""
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create a config directory and copy the MCP config file
        config_dir = temp_dir_path / "config"
        config_dir.mkdir()
        
        # Read the content of the mcp_config_file
        with open(mcp_config_file, "r") as src_file:
            mcp_config_content = src_file.read()
        
        # Write the content to the new file
        mcp_config_dest = config_dir / "mcp_servers.json"
        mcp_config_dest.write_text(mcp_config_content)
        
        # Create a TOML config file
        config_file = temp_dir_path / "config.toml"
        config_file.write_text(f"""
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"
display_name = "Test MCP Assistant"

[prompt]
system_prompt = "You are a test assistant with tool access."

[parameters]
temperature = 0.7
max_tokens = 300

[mcp]
config_path = "config/mcp_servers.json"

[mcp.tools]
github = ["search_repositories", "get_file_contents"]
codemcp = ["ReadFile"]
""")
        
        # Create and patch the instance
        with patch("llmproc.providers.anthropic", MagicMock()):
            with patch("llmproc.providers.Anthropic"):
                with patch("llmproc.llm_process.asyncio.run"):
                    process = LLMProcess.from_toml(config_file)
                    
                    # Check that MCP was initialized correctly
                    assert process.mcp_enabled is True
                    assert process.mcp_tools == {
                        "github": ["search_repositories", "get_file_contents"],
                        "codemcp": ["ReadFile"]
                    }
                    assert process.model_name == "claude-3-haiku-20240307"
                    assert process.provider == "anthropic"
                    assert process.display_name == "Test MCP Assistant"


@patch("llmproc.llm_process.HAS_MCP", True)
@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.Anthropic")
def test_mcp_with_no_tools(mock_anthropic, mock_mcp_registry, mock_env, mcp_config_file):
    """Test behavior when MCP is enabled but no tools are specified."""
    # Mock asyncio.run to do nothing
    with patch("llmproc.llm_process.asyncio.run"):
        # Create empty tools dictionary - not an empty dict
        empty_tools = {"github": []}
        process = LLMProcess(
            model_name="claude-3-haiku-20240307",
            provider="anthropic",
            system_prompt="You are a test assistant.",
            mcp_config_path=mcp_config_file,
            mcp_tools=empty_tools
        )
        
        assert process.mcp_enabled is True
        assert len(process.tools) == 0


@patch("llmproc.llm_process.HAS_MCP", True)
@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.Anthropic")
def test_mcp_with_all_tools(mock_anthropic, mock_mcp_registry, mock_env, mcp_config_file):
    """Test behavior when all tools from a server are requested."""
    # Mock asyncio.run to actually call the _initialize_mcp_tools method
    with patch("llmproc.llm_process.asyncio.run"):
        process = LLMProcess(
            model_name="claude-3-haiku-20240307",
            provider="anthropic",
            system_prompt="You are a test assistant.",
            mcp_config_path=mcp_config_file,
            mcp_tools={"github": "all", "codemcp": ["ReadFile"]}
        )
        
        assert process.mcp_enabled is True


@patch("llmproc.llm_process.HAS_MCP", True)
def test_invalid_mcp_tools_config(mock_mcp_registry, mock_env, mcp_config_file):
    """Test that an invalid MCP tools configuration raises a ValueError."""
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create a config directory and copy the MCP config file
        config_dir = temp_dir_path / "config"
        config_dir.mkdir()
        
        # Read the content of the mcp_config_file
        with open(mcp_config_file, "r") as src_file:
            mcp_config_content = src_file.read()
        
        # Write the content to the new file
        mcp_config_dest = config_dir / "mcp_servers.json"
        mcp_config_dest.write_text(mcp_config_content)
        
        # Create a TOML config file with invalid tools configuration
        config_file = temp_dir_path / "config.toml"
        config_file.write_text(f"""
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"

[prompt]
system_prompt = "You are a test assistant with tool access."

[mcp]
config_path = "config/mcp_servers.json"

[mcp.tools]
github = 123  # This is invalid, should be a list or "all"
""")
        
        # Test that it raises a ValueError
        with patch("llmproc.providers.anthropic", MagicMock()):
            with patch("llmproc.providers.Anthropic"):
                with pytest.raises(ValueError):
                    LLMProcess.from_toml(config_file)


@patch("llmproc.llm_process.HAS_MCP", True)
@patch("llmproc.providers.anthropic", MagicMock())
@patch("llmproc.providers.Anthropic")
def test_run_with_tools(mock_anthropic, mock_mcp_registry, mock_env, mcp_config_file):
    """Test the run method with tool support."""
    # Setup mock client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # Mock the async create method
    mock_response = MagicMock()
    mock_content = [
        MagicMock(type="text", text="This is a test response"),
        MagicMock(type="tool_use", id="tool1", name="test.tool", input={"arg": "value"})
    ]
    mock_response.content = mock_content
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    
    # Also mock _run_anthropic_with_tools to avoid actual execution
    async def mock_run_with_tools(*args, **kwargs):
        return "Mocked tool execution response"
    
    # Create LLMProcess with MCP configuration
    process = LLMProcess(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a test assistant.",
        mcp_config_path=mcp_config_file,
        mcp_tools={"github": ["search_repositories"]}
    )
    
    # Patch the internal method
    with patch.object(process, '_run_anthropic_with_tools', new=mock_run_with_tools):
        # Run the test using asyncio.run to handle the async method
        result = asyncio.run(process.run("Test input"))
        
        # Check the result
        assert result == "Mocked tool execution response"
        
        # Verify the appropriate methods were called
        assert len(process.state) == 2  # system prompt + user input
        
    # Test with a mocked async method
    # This tests that the _async_run method is properly called
    with patch.object(process, '_async_run', new=AsyncMock(return_value="Mocked _async_run result")):
        # Run in async context
        result = asyncio.run(process.run("Test input in async context"))
        assert result == "Mocked _async_run result"


@patch("llmproc.llm_process.HAS_MCP", True)
def test_openai_with_mcp_raises_error(mock_mcp_registry, mock_env, mcp_config_file):
    """Test that using OpenAI with MCP raises an error (not yet supported)."""
    with patch("llmproc.providers.OpenAI", MagicMock()):
        with pytest.raises(ValueError, match="MCP features are currently only supported with the Anthropic provider"):
            LLMProcess(
                model_name="gpt-4o",
                provider="openai",
                system_prompt="You are a test assistant.",
                mcp_config_path=mcp_config_file,
                mcp_tools={"github": ["search_repositories"]}
            )


@patch("llmproc.llm_process.HAS_MCP", False)
def test_mcp_import_error(mock_env, mcp_config_file):
    """Test that trying to use MCP when the package is not installed raises an ImportError."""
    with patch("llmproc.providers.anthropic", MagicMock()):
        with patch("llmproc.providers.Anthropic", MagicMock()):
            with pytest.raises(ImportError, match="MCP features require the mcp-registry package"):
                LLMProcess(
                    model_name="claude-3-haiku-20240307",
                    provider="anthropic",
                    system_prompt="You are a test assistant.",
                    mcp_config_path=mcp_config_file,
                    mcp_tools={"github": ["search_repositories"]}
                )