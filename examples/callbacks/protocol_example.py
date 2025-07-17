"""Example showing :class:`PluginProtocol` usage for IDE support.

This example demonstrates how to use ``PluginProtocol`` for callbacks and hooks
with full IDE autocomplete while preserving duck typing flexibility.
"""

from llmproc.common.results import ToolResult
from llmproc.plugin.datatypes import ToolCallHookResult
from llmproc.plugin.protocol import PluginProtocol
from llmproc.plugin.protocol import PluginProtocol as HookProtocol


# Option 1: Type hint for IDE support (recommended - best performance)
class EnhancedCallback:
    """Callback with enhanced IDE support via type hint. No inheritance overhead."""

    def __init__(self):
        self.tool_count = 0
        self.responses = []

    def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
        self.tool_count += 1
        print(f"🔧 Tool {self.tool_count}: {tool_name} starting with args: {tool_args}")

    def tool_end(self, tool_name: str, result: ToolResult, *, process) -> None:
        status = "✅" if not result.is_error else "❌"
        print(f"{status} Tool {tool_name} completed")

    def response(self, content: str, *, process) -> None:
        self.responses.append(content)
        print(f"💬 Response: {content[:50]}...")

    def hook_user_input(self, user_input: str, process) -> str:
        """Enhance user input with a timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {user_input}"


# Option 2: Explicit inheritance (has method dispatch overhead)
class TypedCallback(PluginProtocol):
    """Callback with explicit protocol inheritance. Creates no-op methods."""

    def tool_start(self, tool_name: str, tool_args: dict, *, process) -> None:
        print(f"📊 Typed callback: {tool_name}")

    def hook_tool_call(self, tool_name: str, args: dict, process) -> ToolCallHookResult:
        """Add metadata to all tool calls."""
        modified_args = args.copy()
        modified_args["_callback_id"] = "typed_callback"
        return ToolCallHookResult(modified_args=modified_args)


# Option 3: Duck typing (no protocol)
class SimpleCallback:
    """Simple callback using duck typing."""

    def response(self, content, *, process):
        print(f"📝 Simple: {len(content)} characters")


def demo_protocol_usage():
    """Demonstrate protocol usage patterns."""
    print("=== PluginProtocol Usage Demo ===\n")

    # Type hint provides full IDE autocomplete
    enhanced: PluginProtocol = EnhancedCallback()
    typed_plugin: PluginProtocol = TypedCallback()
    simple_callback = SimpleCallback()

    print("✅ All callback styles work together:")
    print(f"   Enhanced callback: {type(enhanced).__name__} (type hint - efficient)")
    print(f"   Typed plugin: {type(typed_plugin).__name__} (inheritance - overhead)")
    print(f"   Simple callback: {type(simple_callback).__name__} (duck typing)")

    # Simulate callback usage
    print("\n🔄 Simulating callback execution:")

    # Tool start
    enhanced.tool_start("calculator", {"expression": "2+2"}, process=None)
    typed_plugin.tool_start("calculator", {"expression": "2+2"}, process=None)

    # Tool end
    result = ToolResult(content="4")
    enhanced.tool_end("calculator", result, process=None)

    # Response
    enhanced.response("The answer is 4", process=None)
    simple_callback.response("The answer is 4", process=None)

    # Hook
    enhanced_input = enhanced.hook_user_input("What is 2+2?", process=None)
    print(f"🔀 Enhanced input: {enhanced_input}")

    hook_result = typed_plugin.hook_tool_call("calculator", {"x": 1}, process=None)
    print(f"🔧 Modified args: {hook_result.modified_args}")

    print(f"\n📈 Stats: {enhanced.tool_count} tools, {len(enhanced.responses)} responses")

    # Demonstrate performance difference
    print("\n⚡ Performance Comparison:")
    print("Enhanced (type hint): Only implemented methods exist")
    print(f"  - has tool_start: {hasattr(enhanced, 'tool_start')}")
    print(f"  - has api_request: {hasattr(enhanced, 'api_request')}")  # Should be False

    print("Typed (inheritance): All protocol methods exist (no-ops)")
    print(f"  - has tool_start: {hasattr(typed_plugin, 'tool_start')}")
    print(f"  - has api_request: {hasattr(typed_plugin, 'api_request')}")  # Should be True

    print("\n💡 Recommendation: Use type hints for best performance!")


if __name__ == "__main__":
    demo_protocol_usage()
