#!/usr/bin/env python3
"""
Comprehensive Callback Patterns Cookbook
=========================================

This example demonstrates the flexible callback signature capabilities of LLMProc.
The callback system uses Flask/pytest-style parameter injection - your callbacks
only need to declare the parameters they actually use.

Key Features:
- Minimal signatures that declare only what they need
- Automatic parameter filtering based on method signatures
- Performance benefits with caching
- Backward compatibility maintained
- Clean, readable code

Run with:
    python examples/callbacks/flexible_signatures_cookbook.py
"""

import asyncio
import time
from typing import Optional

from llmproc import LLMProgram
from llmproc.common.results import RunResult, ToolResult

# =============================================================================
# Pattern 1: Basic Patterns - Minimal Signatures
# =============================================================================


class BasicPatterns:
    """Start with the simplest possible signatures."""

    def __init__(self):
        self.tool_count = 0

    def tool_start(self, tool_name):
        """Basic pattern: Just the tool name."""
        self.tool_count += 1
        print(f"🔧 Tool #{self.tool_count}: {tool_name}")

    def tool_end(self, tool_name, result):
        """Selective pattern: Name and result, no process."""
        status = "✅" if not result.is_error else "❌"
        print(f"{status} {tool_name} completed")

    def response(self, content):
        """Minimal pattern: Just the content."""
        print(f"💬 Response: {content[:50]}...")


# =============================================================================
# Pattern 2: Selective Parameters - Choose What You Need
# =============================================================================


class SelectivePatterns:
    """Show how to selectively pick only needed parameters."""

    def __init__(self):
        self.timings = {}

    def tool_start(self, tool_name, tool_args):
        """Need both name and args for logging."""
        self.timings[tool_name] = time.time()
        print(f"📊 {tool_name} starting with: {list(tool_args.keys())}")

    def tool_end(self, tool_name, result):
        """Only need name and result for timing."""
        if tool_name in self.timings:
            duration = time.time() - self.timings[tool_name]
            print(f"⏱️  {tool_name} took {duration:.2f}s")

    def turn_end(self, response, tool_results):
        """Track conversation turns without needing process."""
        print(f"🔄 Turn completed: {len(tool_results)} tools, {len(response)} chars")


# =============================================================================
# Pattern 3: Advanced Patterns - Full Context When Needed
# =============================================================================


class AdvancedPatterns:
    """Use full context when you need process access."""

    def __init__(self):
        self.process_stats = {}

    def tool_start(self, tool_name, tool_args, process):
        """Full context for advanced process inspection."""
        pid = id(process)  # Use process ID as key
        if pid not in self.process_stats:
            self.process_stats[pid] = {"tools": 0, "tokens": 0}

        self.process_stats[pid]["tools"] += 1
        token_count = process.count_tokens()
        self.process_stats[pid]["tokens"] = token_count

        print(f"🎯 {tool_name} (Process {pid}: {self.process_stats[pid]['tools']} tools, {token_count} tokens)")

    def run_end(self, run_result, process):
        """Full context for detailed run analysis."""
        pid = id(process)
        stats = self.process_stats.get(pid, {})

        print("📈 Run completed:")
        print(f"   - Duration: {run_result.duration_ms}ms")
        print(f"   - API calls: {run_result.api_call_count}")
        print(f"   - Tools used: {stats.get('tools', 0)}")
        print(f"   - Final tokens: {stats.get('tokens', 0)}")


# =============================================================================
# Pattern 4: Legacy Compatibility - Old Style Still Works
# =============================================================================


class LegacyPatterns:
    """Show that old-style signatures still work."""

    def tool_start(self, tool_name, tool_args, *, process):
        """Legacy pattern: keyword-only process parameter."""
        print(f"🔄 Legacy style: {tool_name} starting")

    def response(self, content, *, process):
        """Legacy pattern: keyword-only process parameter."""
        print(f"📝 Legacy response: {len(content)} chars")


# =============================================================================
# Pattern 5: Mixed Patterns - Use Different Styles Together
# =============================================================================


class MixedPatterns:
    """Show how different signature styles work together."""

    def __init__(self):
        self.api_calls = 0

    def tool_start(self, tool_name):
        """Minimal: Just the tool name."""
        print(f"🚀 Starting {tool_name}")

    def api_request(self, api_request, process):
        """Selective: Need both request and process."""
        self.api_calls += 1
        model = getattr(process, "model_name", "unknown")
        print(f"📡 API call #{self.api_calls} to {model}")

    def turn_start(self, process, run_result: Optional[RunResult] = None):
        """Advanced: Optional parameters work too."""
        tokens = process.count_tokens()
        if run_result:
            print(f"🔄 Turn starting (continuing from {run_result.duration_ms}ms run, {tokens} tokens)")
        else:
            print(f"🔄 Turn starting (fresh conversation, {tokens} tokens)")


# =============================================================================
# Pattern 6: Performance Comparison - Why Flexible Signatures Matter
# =============================================================================


class PerformanceDemo:
    """Demonstrate the performance benefits of flexible signatures."""

    def __init__(self):
        self.call_count = 0

    def tool_start(self, tool_name):
        """Minimal signature = minimal overhead."""
        self.call_count += 1
        # No unnecessary parameters passed or processed

    def get_stats(self):
        return f"Efficient callback called {self.call_count} times"


# =============================================================================
# Demo Runner
# =============================================================================


def calculator(a: float, b: float, operation: str = "add") -> dict:
    """Simple calculator tool for demonstration."""
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    else:
        result = 0
    return {"result": result, "operation": operation}


async def demonstrate_patterns():
    """Run through all callback patterns."""
    print("=" * 70)
    print("📚 LLMProc Flexible Callback Signatures Cookbook")
    print("=" * 70)

    # Create a simple program
    program = LLMProgram(
        model_name="claude-3-5-haiku-20241022",
        provider="anthropic",
        system_prompt="You are a helpful calculator assistant.",
        parameters={"max_tokens": 1000},
        tools=[calculator],
    )

    process = await program.start()

    # Register all callback patterns
    basic = BasicPatterns()
    selective = SelectivePatterns()
    advanced = AdvancedPatterns()
    legacy = LegacyPatterns()
    mixed = MixedPatterns()
    performance = PerformanceDemo()

    process.add_plugins(basic)
    process.add_plugins(selective)
    process.add_plugins(advanced)
    process.add_plugins(legacy)
    process.add_plugins(mixed)
    process.add_plugins(performance)

    print("\n🎬 Running demonstration...")
    print("=" * 50)

    # Run a simple calculation
    result = await process.run("What is 15 multiplied by 4?")

    print("=" * 50)
    print(f"🎯 Final result: {process.get_last_message()}")
    print(f"🏆 {performance.get_stats()}")

    print("\n" + "=" * 70)
    print("🎉 Key Takeaways:")
    print("   ✅ Callbacks only need parameters they actually use")
    print("   ✅ Flask/pytest-style parameter injection works automatically")
    print("   ✅ Performance benefits from minimal signatures")
    print("   ✅ Legacy signatures still work for backward compatibility")
    print("   ✅ Mix and match different styles freely")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demonstrate_patterns())
