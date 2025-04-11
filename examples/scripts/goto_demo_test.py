#!/usr/bin/env python3
"""
Test script for the GOTO time travel tool.

This script demonstrates the GOTO feature with non-interactive inputs for testing.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from llmproc import LLMProgram

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("goto_demo_test")


# Custom callback handler with special tracking for GOTO tool
class GotoTracker:
    def __init__(self):
        self.active_tools = set()
        self.tool_calls = []
        self.goto_count = 0
        self.found_error = False
        self.single_run_count = 0  # Count per user message
        self.current_message = 0
        
    def on_tool_start(self, tool_name, tool_args):
        """Callback for when a tool starts execution"""
        self.active_tools.add(tool_name)
        self.tool_calls.append({"tool": tool_name, "args": tool_args, "status": "started"})
        
        if tool_name == "goto":
            position = tool_args.get("position", "unknown")
            message = tool_args.get("message", "")
            self.goto_count += 1
            self.single_run_count += 1
            
            if self.single_run_count > 1:
                self.found_error = True
                error_msg = f"⛔️ ERROR: GOTO CALLED MULTIPLE TIMES ({self.single_run_count}) FOR THE SAME USER MESSAGE."
                print(f"\n{error_msg}")
                raise RuntimeError(error_msg)
            
            print(f"\n🕰️ TIME TRAVEL #{self.goto_count} INITIATED")
            print(f"├── Target position: {position}")
            if message:
                print(f"└── Message: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            else:
                print(f"└── No message provided")
                
    def on_tool_end(self, tool_name, result):
        """Callback for when a tool completes execution"""
        if tool_name in self.active_tools:
            self.active_tools.remove(tool_name)
            
        self.tool_calls.append({"tool": tool_name, "result": result, "status": "completed"})
            
        if tool_name == "goto":
            # Check if result is ToolResult or another type with different attributes
            if hasattr(result, 'error') and result.error:
                print(f"\n❌ TIME TRAVEL FAILED: {result.error}")
            elif hasattr(result, 'success') and not result.success:
                print(f"\n❌ TIME TRAVEL FAILED: {getattr(result, 'message', 'Unknown error')}")
            else:
                success_msg = getattr(result, 'result', None)
                if not success_msg and hasattr(result, 'message'):
                    success_msg = result.message
                print(f"\n✅ TIME TRAVEL COMPLETED: {success_msg or 'Successfully'}")


async def main():
    # Non-interactive test inputs
    # Single GOTO test for faster iteration
    test_inputs = [
        "What is your name?",
        "Use the GOTO tool to reset to message 0 (msg_0) and prepare to tell me a joke in your next response.",
        "Tell me a short joke please - something clean and clever."
    ]
    
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        program_path = "./examples/features/goto.toml"
        
    try:
        # Create the GOTO tracker
        tracker = GotoTracker()
        
        # Step 1: Load the program
        print(f"Loading program from: {program_path}")
        program = LLMProgram.from_toml(program_path)
        
        # Ensure GOTO tool is enabled
        current_tools = program.get_enabled_tools() if hasattr(program, 'get_enabled_tools') else []
        if not current_tools:  # Try to get from attributes if method doesn't exist
            print("⚠️  Falling back to direct access for tools")
            # Try different attributes that might contain tools
            for attr in ['tools', '_tools', 'enabled_tools']:
                if hasattr(program, attr):
                    attr_value = getattr(program, attr)
                    if isinstance(attr_value, list):
                        current_tools = attr_value
                    elif isinstance(attr_value, dict) and 'enabled' in attr_value:
                        current_tools = attr_value['enabled']
                    break
                    
        # Add GOTO tool if needed
        if "goto" not in current_tools:
            print(f"⚠️  Adding GOTO tool to enabled tools: {current_tools}")
            new_tools = list(current_tools) + ["goto"]
            program.set_enabled_tools(new_tools)
            print(f"GOTO tool enabled. Enabled tools: {new_tools}")
                
        # Step 2: Start the process
        print("Starting process...")
        process = await program.start()
        
        # Step 3: Prepare callbacks
        callbacks = {
            "on_tool_start": tracker.on_tool_start,
            "on_tool_end": tracker.on_tool_end
        }
        
        # Step 4: Run with pre-defined inputs
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n--- Test Input #{i} ---")
            print(f"You> {user_input}")
            
            # Reset the counter for each new user message
            tracker.single_run_count = 0
            tracker.current_message += 1
            
            # Run the process
            start_time = time.time()
            run_result = await process.run(user_input, callbacks=callbacks)
            elapsed = time.time() - start_time
            
            # Get the response
            response = process.get_last_message()
            
            # Display result metrics
            print(f"\nRun completed in {elapsed:.2f}s")
            print(f"API calls: {run_result.api_calls}")
            
            # Display the response
            print(f"\n{process.display_name}> {response}")
            
        # Final summary
        print("\n--- Test Summary ---")
        print(f"Total GOTO tool uses: {tracker.goto_count}")
        
        # Count tool usage
        tool_counts = {}
        for call in tracker.tool_calls:
            if call["status"] == "started":
                tool_name = call["tool"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                
        print(f"Tool usage: {tool_counts}")
        
    except Exception as e:
        import traceback
        
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))