"""Tests for MessageIDPlugin integration with the goto tool."""

import pytest
from unittest.mock import Mock

from llmproc.program import LLMProgram
from llmproc.plugins.message_id import MessageIDPlugin
from llmproc.config.schema import MessageIDPluginConfig


class TestMessageIDPluginIntegration:
    """Test message ID plugin optionally providing goto."""

    def test_plugin_must_be_explicitly_added(self):
        """goto tool now requires explicit plugin registration."""
        from llmproc.config.schema import MessageIDPluginConfig

        program = LLMProgram(model_name="m", provider="p")
        # goto tool is no longer a builtin tool - must come from plugin
        program.add_plugins(MessageIDPlugin(MessageIDPluginConfig(enable_goto=True)))
        program.compile()

        plugin_types = {type(p) for p in program.plugins}
        assert MessageIDPlugin in plugin_types

    @pytest.mark.asyncio
    async def test_hook_user_input(self):
        """MessageIDPlugin prefixes message IDs correctly."""
        plugin = MessageIDPlugin()
        process = Mock()
        process.state = ["msg1", "msg2"]

        result = await plugin.hook_user_input("Hello world", process)

        assert result == "[msg_2] Hello world"

    def test_tool_override(self):
        """ToolConfig entries override goto metadata."""
        from llmproc.common.metadata import get_tool_meta
        from llmproc.plugins.override_utils import apply_tool_overrides
        from llmproc.config.tool import ToolConfig

        cfg = MessageIDPluginConfig(
            enable_goto=True,
            tools=[ToolConfig(name="goto", description="jump")]
        )
        plugin = MessageIDPlugin(cfg)

        # Get the plugin's goto tool
        provided_tools = plugin.hook_provide_tools()
        assert len(provided_tools) == 1
        goto_tool = provided_tools[0]

        # Apply tool overrides
        tools = apply_tool_overrides(provided_tools, cfg.tools)
        assert len(tools) == 1
        assert get_tool_meta(tools[0]).description == "jump"

        # No need to restore metadata since we're not modifying the original
