#!/usr/bin/env python3
"""
Progressive Callback Complexity Example
=======================================

This example shows how to start with simple callbacks and gradually add complexity
as your needs grow. Each section builds on the previous one, demonstrating the
natural progression from basic logging to sophisticated process monitoring.

Key Learning Points:
- Start minimal, add complexity only when needed
- Flexible signatures let you evolve callbacks naturally
- Mix different complexity levels freely
- Performance scales with what you actually use

Run with:
    python examples/callbacks/progressive_complexity.py
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import Optional

from llmproc import LLMProgram
from llmproc.common.results import RunResult, ToolResult

# =============================================================================
# Level 1: Basic Logging - Start Here
# =============================================================================


class BasicLogger:
    """Level 1: Just log what's happening."""

    def tool_start(self, tool_name):
        print(f"🔧 Tool: {tool_name}")

    def response(self, content):
        print(f"💬 Response: {len(content)} chars")


# =============================================================================
# Level 2: Simple Metrics - Add Basic Counting
# =============================================================================


class SimpleMetrics:
    """Level 2: Count tools and track basic stats."""

    def __init__(self):
        self.tool_count = 0
        self.total_response_length = 0

    def tool_start(self, tool_name):
        self.tool_count += 1
        print(f"🔧 Tool #{self.tool_count}: {tool_name}")

    def response(self, content):
        self.total_response_length += len(content)
        print(f"💬 Response: {len(content)} chars (total: {self.total_response_length})")

    def get_summary(self):
        return f"Used {self.tool_count} tools, {self.total_response_length} response chars"


# =============================================================================
# Level 3: Timing and Performance - Add Time Tracking
# =============================================================================


class TimingMetrics:
    """Level 3: Track timing and performance metrics."""

    def __init__(self):
        self.tool_times = {}
        self.tool_stats = defaultdict(lambda: {"count": 0, "total_time": 0})

    def tool_start(self, tool_name):
        self.tool_times[tool_name] = time.time()
        print(f"⏱️  Starting {tool_name}")

    def tool_end(self, tool_name, result):
        if tool_name in self.tool_times:
            duration = time.time() - self.tool_times[tool_name]
            self.tool_stats[tool_name]["count"] += 1
            self.tool_stats[tool_name]["total_time"] += duration

            avg_time = self.tool_stats[tool_name]["total_time"] / self.tool_stats[tool_name]["count"]
            status = "✅" if not result.is_error else "❌"
            print(f"{status} {tool_name}: {duration:.2f}s (avg: {avg_time:.2f}s)")

    def get_performance_report(self):
        report = "Performance Report:\n"
        for tool, stats in self.tool_stats.items():
            avg = stats["total_time"] / stats["count"]
            report += f"  {tool}: {stats['count']} calls, {avg:.2f}s avg\n"
        return report


# =============================================================================
# Level 4: Process Monitoring - Add Process Context
# =============================================================================


class ProcessMonitor:
    """Level 4: Monitor process state and resource usage."""

    def __init__(self):
        self.process_stats = {}

    def tool_start(self, tool_name, process):
        pid = id(process)
        if pid not in self.process_stats:
            self.process_stats[pid] = {"tools": 0, "start_tokens": process.count_tokens()}

        self.process_stats[pid]["tools"] += 1
        current_tokens = process.count_tokens()

        print(f"🎯 {tool_name} (Process {pid}: {self.process_stats[pid]['tools']} tools, {current_tokens} tokens)")

    def run_end(self, run_result, process):
        pid = id(process)
        stats = self.process_stats.get(pid, {})
        final_tokens = process.count_tokens()
        start_tokens = stats.get("start_tokens", 0)

        print(f"📊 Process {pid} completed:")
        print(f"   Duration: {run_result.duration_ms}ms")
        print(f"   Tools used: {stats.get('tools', 0)}")
        print(f"   Token growth: {start_tokens} → {final_tokens} (+{final_tokens - start_tokens})")


# =============================================================================
# Level 5: Advanced Analytics - Full Context Analysis
# =============================================================================


class AdvancedAnalytics:
    """Level 5: Comprehensive analysis with full context."""

    def __init__(self):
        self.session_data = {"tools": [], "api_calls": [], "turns": [], "errors": []}

    def tool_start(self, tool_name, tool_args, process):
        self.session_data["tools"].append(
            {"name": tool_name, "args": tool_args, "timestamp": time.time(), "process_tokens": process.count_tokens()}
        )

    def api_request(self, api_request, process):
        self.session_data["api_calls"].append(
            {
                "timestamp": time.time(),
                "model": getattr(process, "model_name", "unknown"),
                "tokens": process.count_tokens(),
            }
        )

    def turn_end(self, response, tool_results, process):
        errors = [r for r in tool_results if r.is_error]
        self.session_data["turns"].append(
            {
                "timestamp": time.time(),
                "response_length": len(response),
                "tools_used": len(tool_results),
                "errors": len(errors),
                "final_tokens": process.count_tokens(),
            }
        )

        if errors:
            self.session_data["errors"].extend(errors)

    def get_analytics_report(self):
        tools_used = set(tool["name"] for tool in self.session_data["tools"])
        total_turns = len(self.session_data["turns"])
        total_errors = len(self.session_data["errors"])

        return {
            "summary": {
                "total_tools": len(self.session_data["tools"]),
                "unique_tools": len(tools_used),
                "total_turns": total_turns,
                "total_errors": total_errors,
                "api_calls": len(self.session_data["api_calls"]),
            },
            "tool_usage": list(tools_used),
            "error_rate": total_errors / max(1, total_turns) * 100,
        }


# =============================================================================
# Level 6: Adaptive Callbacks - Context-Aware Behavior
# =============================================================================


class AdaptiveCallbacks:
    """Level 6: Callbacks that adapt based on context."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.error_count = 0
        self.consecutive_errors = 0

    def tool_start(self, tool_name, tool_args):
        # Adaptive verbosity based on error state
        if self.verbose or self.consecutive_errors > 0:
            print(f"🔧 {tool_name} with {len(tool_args)} args")
        else:
            print(f"🔧 {tool_name}")

    def tool_end(self, tool_name, result):
        if result.is_error:
            self.error_count += 1
            self.consecutive_errors += 1
            print(f"❌ {tool_name} failed (consecutive errors: {self.consecutive_errors})")

            # Become verbose after errors
            if self.consecutive_errors >= 2:
                print("⚠️  Enabling verbose mode due to consecutive errors")
                self.verbose = True
        else:
            if self.consecutive_errors > 0:
                print(f"✅ {tool_name} succeeded (errors cleared)")
                self.consecutive_errors = 0
                self.verbose = False
            else:
                print(f"✅ {tool_name}")

    def response(self, content, process):
        # Adaptive response based on process state
        tokens = process.count_tokens()
        if tokens > 1000:  # High token usage
            print(f"💬 Response: {len(content)} chars, {tokens} tokens (HIGH USAGE)")
        else:
            print(f"💬 Response: {len(content)} chars")


# =============================================================================
# Demo Functions
# =============================================================================


def simple_calculator(a: float, b: float, operation: str = "add") -> dict:
    """Simple calculator tool with potential for errors."""
    if operation == "add":
        return {"result": a + b}
    elif operation == "multiply":
        return {"result": a * b}
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        return {"result": a / b}
    else:
        raise ValueError(f"Unknown operation: {operation}")


def flaky_tool(success_rate: float = 0.7) -> dict:
    """A tool that fails sometimes to demonstrate error handling."""
    import random

    if random.random() < success_rate:
        return {"result": "success"}
    else:
        raise RuntimeError("Flaky tool failed")


async def demonstrate_progression():
    """Show the progression from simple to complex callbacks."""
    print("=" * 80)
    print("🚀 Progressive Callback Complexity Demonstration")
    print("=" * 80)

    # Create program
    program = LLMProgram(
        model_name="claude-3-5-haiku-20241022",
        provider="anthropic",
        system_prompt="You are a helpful assistant. Use the calculator for math problems.",
        parameters={"max_tokens": 500},
        tools=[simple_calculator, flaky_tool],
    )

    # Level 1: Basic Logging
    print("\n🔹 Level 1: Basic Logging")
    print("-" * 40)

    process1 = await program.start()
    basic = BasicLogger()
    process1.add_plugins(basic)

    await process1.run("What is 10 + 5?")

    # Level 2: Simple Metrics
    print("\n🔹 Level 2: Simple Metrics")
    print("-" * 40)

    process2 = await program.start()
    metrics = SimpleMetrics()
    process2.add_plugins(metrics)

    await process2.run("What is 20 * 3?")
    print(f"📊 {metrics.get_summary()}")

    # Level 3: Timing
    print("\n🔹 Level 3: Timing and Performance")
    print("-" * 40)

    process3 = await program.start()
    timing = TimingMetrics()
    process3.add_plugins(timing)

    await process3.run("Calculate 100 / 4 and then multiply by 6")
    print(timing.get_performance_report())

    # Level 4: Process Monitoring
    print("\n🔹 Level 4: Process Monitoring")
    print("-" * 40)

    process4 = await program.start()
    monitor = ProcessMonitor()
    process4.add_plugins(monitor)

    await process4.run("What is 15 * 8?")

    # Level 5: Advanced Analytics
    print("\n🔹 Level 5: Advanced Analytics")
    print("-" * 40)

    process5 = await program.start()
    analytics = AdvancedAnalytics()
    process5.add_plugins(analytics)

    await process5.run("Calculate 25 + 17 and then divide by 2")

    report = analytics.get_analytics_report()
    print(f"📈 Analytics: {json.dumps(report, indent=2)}")

    # Level 6: Adaptive Callbacks
    print("\n🔹 Level 6: Adaptive Callbacks")
    print("-" * 40)

    process6 = await program.start()
    adaptive = AdaptiveCallbacks()
    process6.add_plugins(adaptive)

    # This might trigger some errors and show adaptive behavior
    await process6.run("Try the flaky tool and then calculate 50 / 2")

    print("\n" + "=" * 80)
    print("🎯 Key Takeaways:")
    print("   ✅ Start simple, add complexity only when needed")
    print("   ✅ Flexible signatures let you evolve naturally")
    print("   ✅ Mix different complexity levels freely")
    print("   ✅ Performance scales with what you actually use")
    print("   ✅ Callbacks can be stateful and adaptive")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_progression())
