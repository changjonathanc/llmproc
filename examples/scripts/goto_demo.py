#!/usr/bin/env python3
"""
Interactive demo script for the GOTO time travel tool.

This script demonstrates how to:
1. Load and start a program with the GOTO tool
2. Use callbacks to monitor time travel operations
3. Display state changes during time travel
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from llmproc import LLMProgram

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("goto_demo")


# Custom callback handler with special tracking for GOTO tool
class GotoTracker:
    def __init__(self):
        self.active_tools = set()
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_idx = 0
        self.should_run = False
        self.task = None
        self.time_travel_count = 0
        self.state_before_goto = []
        
    def start(self):
        self.should_run = True
        self.task = asyncio.create_task(self._run_spinner())
        
    async def _run_spinner(self):
        while self.should_run and self.active_tools:
            tools_str = ", ".join(self.active_tools)
            spinner = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
            
            if "goto" in self.active_tools:
                sys.stdout.write(f"\r{spinner} 🕰️ Time Travel in progress...")
            else:
                sys.stdout.write(f"\r{spinner} Processing tools: {tools_str}")
                
            sys.stdout.flush()
            self.spinner_idx += 1
            await asyncio.sleep(0.1)
            
        # Clear the spinner line
        if self.active_tools:
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
            
    def stop(self):
        self.should_run = False
        if self.task:
            self.task.cancel()
            
        # Clear any remaining spinner
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        
    def on_tool_start(self, tool_name, tool_args):
        """Callback for when a tool starts execution"""
        self.active_tools.add(tool_name)
        
        if tool_name == "goto":
            position = tool_args.get("position", "unknown")
            message = tool_args.get("message", "")
            self.time_travel_count += 1
            
            print(f"\n🕰️ TIME TRAVEL #{self.time_travel_count} INITIATED")
            print(f"├── Target position: {position}")
            if message:
                print(f"└── Message: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            else:
                print(f"└── No message provided")
                
        if not self.task or self.task.done():
            self.start()
            
    def on_tool_end(self, tool_name, result):
        """Callback for when a tool completes execution"""
        if tool_name in self.active_tools:
            self.active_tools.remove(tool_name)
            
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
                
        if not self.active_tools:
            self.stop()


def print_help():
    """Print help information for GOTO tool demo."""
    print("\n🕰️ GOTO TOOL DEMO HELP")
    print("----------------------")
    print("This demo shows the GOTO time travel capability of LLMProc.")
    print("The model can use a tool to \"travel back in time\" to an earlier point in the conversation.")
    print()
    print("Special commands:")
    print("  /help      - Show this help")
    print("  /exit      - Exit the demo")
    print("  /reset     - Reset the conversation")
    print("  /state     - Show current conversation state")
    print()
    print("To use the GOTO tool, simply ask the model to go back to an earlier point in time.")
    print("Examples:")
    print("  \"Go back to the beginning of our conversation.\"")
    print("  \"Please use the GOTO tool to reset to message 0 and start over.\"")
    print("  \"I think we took a wrong approach. Let's go back to message 2.\"")
    print()


async def main():
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
        start_time = time.time()
        process = await program.start()
        init_time = time.time() - start_time
        print(f"Process initialized in {init_time:.2f} seconds")
        
        # Step 3: Prepare callbacks
        callbacks = {
            "on_tool_start": tracker.on_tool_start,
            "on_tool_end": tracker.on_tool_end
        }
        
        # Show initial help
        print_help()
        
        # Step 4: Run with user input
        while True:
            # Get user input
            print()
            user_input = input("You> ")
            
            # Handle special commands
            if user_input.lower() in ["/exit", "/quit"]:
                break
            elif user_input.lower() == "/help":
                print_help()
                continue
            elif user_input.lower() == "/reset":
                process = await program.start()
                print("Conversation reset.")
                continue
            elif user_input.lower() == "/state":
                print("\nCurrent conversation state:")
                for i, msg in enumerate(process.state):
                    msg_id = msg.get("id", f"msg_{i}")
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 100:
                        content = content[:100] + "..."
                    print(f"{msg_id} ({role}): {content}")
                continue
                
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
            
    except Exception as e:
        import traceback
        
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))