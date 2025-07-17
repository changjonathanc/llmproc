"""Tests for the unified :class:`PluginProtocol` typing helper."""

from typing import get_type_hints

import pytest

from llmproc.common.results import ToolResult
from llmproc.plugin.datatypes import ToolCallHookResult
from llmproc.plugin.events import CallbackEvent
from llmproc.plugin.protocol import PluginProtocol


class TestCallbackProtocol:
    """Test callback protocol for IDE support and documentation."""

    def test_protocol_can_be_imported(self):
        """Protocol should be importable without errors."""
        from llmproc.plugin.protocol import PluginProtocol

        assert PluginProtocol is not None

    def test_protocol_has_all_callback_events(self):
        """Protocol should have methods for all callback events."""
        protocol_methods = set(dir(PluginProtocol))

        # Check that all callback events have corresponding methods
        missing_methods = []
        for event in CallbackEvent:
            method_name = event.value
            if method_name not in protocol_methods:
                missing_methods.append(method_name)

        assert not missing_methods, f"Protocol missing methods for events: {missing_methods}"

    def test_protocol_type_hints_work(self):
        """Protocol should provide proper type hints."""
        # Protocol methods are accessible via __annotations__ or by checking methods exist
        protocol_methods = [
            name
            for name in dir(PluginProtocol)
            if not name.startswith("_") and callable(getattr(PluginProtocol, name, None))
        ]

        # Should have methods defined
        assert len(protocol_methods) > 0, "Protocol should have methods defined"
        assert "tool_start" in protocol_methods

    def test_class_can_implement_protocol_partially(self):
        """Classes should be able to implement partial protocol."""

        class PartialCallback:
            def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
                pass

            def response(self, content: str, *, process) -> None:
                pass

        # This should work - partial implementation is allowed
        callback = PartialCallback()
        # Protocol is structural, so this should type-check
        assert hasattr(callback, "tool_start")
        assert hasattr(callback, "response")

    def test_protocol_preserves_duck_typing(self):
        """Protocol should not interfere with duck typing."""

        # This is the key test - classes don't need to inherit from protocol
        class DuckTypedCallback:
            def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
                self.last_tool = tool_name

        callback = DuckTypedCallback()

        # Should work without explicit protocol inheritance
        assert hasattr(callback, "tool_start")
        callback.tool_start("test_tool", {}, process=None)
        assert callback.last_tool == "test_tool"

    def test_protocol_signatures_match_expected_usage(self):
        """Protocol signatures should match how callbacks are actually called."""

        class TestCallback:
            def __init__(self):
                self.calls = []

            def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
                self.calls.append(("tool_start", tool_name, tool_args))

            def tool_end(self, tool_name: str, result: ToolResult, *, process) -> None:
                self.calls.append(("tool_end", tool_name, result))

            def response(self, content: str, *, process) -> None:
                self.calls.append(("response", content))

        callback = TestCallback()

        # Test that the signatures work as expected
        result = ToolResult(content="2")
        callback.tool_start("calc", {"x": 1}, process=None)
        callback.tool_end("calc", result, process=None)
        callback.response("Hello", process=None)

        assert len(callback.calls) == 3
        assert callback.calls[0] == ("tool_start", "calc", {"x": 1})
        assert callback.calls[1] == ("tool_end", "calc", result)
        assert callback.calls[2] == ("response", "Hello")

    def test_type_checking_imports_work(self):
        """TYPE_CHECKING imports should not cause runtime errors."""
        # This test ensures that the TYPE_CHECKING imports don't break
        protocol = PluginProtocol
        assert protocol is not None

        # The protocol should have methods defined
        assert hasattr(protocol, "tool_start")

    def test_performance_difference_between_approaches(self):
        """Demonstrate performance difference between type hints and inheritance."""

        # Type hint approach - no inheritance
        class TypeHintCallback:
            def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
                pass

        # Inheritance approach - inherits all methods
        class InheritanceCallback(PluginProtocol):
            def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
                pass

        type_hint_cb = TypeHintCallback()
        inheritance_cb = InheritanceCallback()

        # Type hint: only implemented methods exist
        assert hasattr(type_hint_cb, "tool_start") is True
        assert hasattr(type_hint_cb, "api_request") is False

        # Inheritance: all protocol methods exist (as no-ops)
        assert hasattr(inheritance_cb, "tool_start") is True
        assert hasattr(inheritance_cb, "api_request") is True  # No-op exists

        # Test that inherited methods are no-ops
        result = inheritance_cb.api_request({}, process=None)
        assert result is None  # No-op returns None


class TestPluginProtocolForHooks:
    """Test PluginProtocol covers hook methods."""

    def test_plugin_protocol_has_hook_methods(self):
        """PluginProtocol should define expected hook methods."""
        hooks = set(dir(PluginProtocol))
        expected = {
            "hook_user_input",
            "hook_tool_call",
            "hook_tool_result",
            "hook_system_prompt",
            "hook_response",
        }
        assert expected.issubset(hooks)

    def test_class_can_implement_hook_protocol(self):
        """Classes can implement PluginProtocol partially for hooks."""

        class MyPlugin:
            def hook_user_input(self, user_input: str, process) -> str:
                return user_input

        plugin: PluginProtocol = MyPlugin()
        assert plugin.hook_user_input("x", None) == "x"
