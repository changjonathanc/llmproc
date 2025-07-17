"""Tests for provider-hosted server tools registration."""

import pytest

from llmproc import LLMProgram
from llmproc.tools.tool_manager import ToolManager


@pytest.mark.asyncio
async def test_anthropic_web_search_registration():
    """Anthropic WebSearchTool registers when enabled."""
    manager = ToolManager()
    config = {
        "mcp_enabled": False,
        "tools": {"anthropic": {"web_search": {"enabled": True}}},
    }

    await manager.register_tools([], config)

    assert "web_search" in manager.runtime_registry.get_tool_names()


@pytest.mark.asyncio
async def test_openai_web_search_registration():
    """OpenAI WebSearchTool registers when enabled."""
    manager = ToolManager()
    config = {
        "mcp_enabled": False,
        "tools": {"openai": {"web_search": {"enabled": True}}},
    }

    await manager.register_tools([], config)

    assert "web_search" in manager.runtime_registry.get_tool_names()


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_anthropic_web_search_api_integration():
    """Test Anthropic web search with actual API integration."""
    program = LLMProgram.from_yaml("examples/web_search.yaml")
    process = await program.start()

    # Verify web search tool is registered
    tool_names = process.tool_manager.runtime_registry.get_tool_names()
    assert "web_search" in tool_names

    # Verify it's the Anthropic tool
    web_search_tool = process.tool_manager.runtime_registry.get_tool("web_search")
    assert web_search_tool.provider == "anthropic"
    assert web_search_tool.schema["type"] == "web_search_20250305"

    # Test that tool schemas are properly formatted for API
    schemas = process.tool_manager.get_tool_schemas()
    web_search_schema = next((s for s in schemas if s.get("name") == "web_search"), None)
    assert web_search_schema is not None
    assert web_search_schema["type"] == "web_search_20250305"
    assert "max_uses" in web_search_schema
    assert "allowed_domains" in web_search_schema


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.openai_api
@pytest.mark.asyncio
async def test_openai_web_search_api_integration():
    """Test OpenAI web search with actual API integration."""
    program = LLMProgram.from_yaml("examples/openai_web_search.yaml")
    process = await program.start()

    # Verify web search tool is registered
    tool_names = process.tool_manager.runtime_registry.get_tool_names()
    assert "web_search" in tool_names

    # Verify it's the OpenAI tool
    web_search_tool = process.tool_manager.runtime_registry.get_tool("web_search")
    assert web_search_tool.provider == "openai"
    assert web_search_tool.schema["type"] == "web_search"

    # Test that tool schemas are properly formatted for API
    schemas = process.tool_manager.get_tool_schemas()
    web_search_schema = next((s for s in schemas if s.get("name") == "web_search"), None)
    assert web_search_schema is not None
    assert web_search_schema["type"] == "web_search"
    assert "search_context_size" in web_search_schema
    assert web_search_schema["search_context_size"] == "high"
