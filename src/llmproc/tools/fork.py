"""Fork system call for LLMProcess to create a copy of the current process."""

import asyncio
import copy
import os
import sys
import json
from typing import Any, Dict, List, Optional

from llmproc.llm_process import LLMProcess

# Detailed fork tool description explaining the Unix metaphor and usage patterns
fork_tool_description = """
You can use this tool to fork the conversation into multiple instances of yourself and let each instance continue answering and using tools.
This is analogous to the fork() system call in Unix.

pid = fork([PROMPT]) # prompt to yourself to continue the conversation

if pid == 0:
    # child process
    You'll receive PROMPT as the tool_result and continue the conversation
else:
    # parent process
    You'll wait for all children to finish and receive the final message from each instance in the following format
    [
        {
            "id": 0,
            "message": "the final message from the child process"
        },
    ]

Different from Unix, you can fork multiple children at once:
fork([PROMPT0, PROMPT1, PROMPT2, ...])

When to use this tool:
You can fork yourself to do tasks that would otherwise fill up the context length but only the final result matters.
For example, if you need to read a large file to find certain details, or if you need to execute multiple tools step by step but you don't need the intermediate results.

You can fork multiple instances to perform tasks in parallel without performing them in serial which would quickly fill up the context length.
Each forked process has a complete copy of the conversation history up to the fork point, ensuring continuity and context preservation.
"""

# Definition of the fork tool for Anthropic API
fork_tool_def = {
    "name": "fork",
    "description": fork_tool_description,
    "input_schema": {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "description": "List of prompts/instructions for each forked process",
                "items": {
                    "type": "string",
                    "description": "A specific task or query to be handled by a forked process"
                }
            }
        },
        "required": ["prompts"]
    }
}

async def fork_tool(
    prompts: List[str],
    llm_process: Optional[LLMProcess] = None,
) -> Dict[str, Any]:
    """Create a copy of the current process to handle multiple tasks in parallel.
    
    This system call allows an LLM process to fork itself into multiple child processes,
    with each child inheriting the full conversation history of the parent process.
    
    Args:
        prompts: List of prompts/instructions for each forked process
        llm_process: The parent LLMProcess instance to fork
        
    Returns:
        A dictionary with responses from all forked processes
        
    Raises:
        ValueError: If the process cannot be forked
    """
    debug = getattr(llm_process, 'debug_tools', False) and os.environ.get("LLMPROC_DEBUG", "").lower() == "true"
    
    if not llm_process:
        error_msg = "Fork system call requires a parent LLMProcess"
        if debug:
            print(f"FORK ERROR: {error_msg}", file=sys.stderr)
        return {
            "error": error_msg,
            "is_error": True,
        }
    
    try:
        # Start a list to track all forked processes and their results
        forked_results = []
        
        for i, prompt in enumerate(prompts):
            if debug:
                print(f"FORK: Creating fork {i+1} with prompt: {prompt[:100]}...", file=sys.stderr)
                
            # Create a deep copy of the parent process
            # We'll use a new method to ensure a proper deep copy
            forked_process = llm_process.fork_process()
            
            if debug:
                print(f"FORK: Successfully created fork {i+1}", file=sys.stderr)
            
            # Run the forked process with the specified prompt
            # The prompt will be processed as a user query
            forked_response = await forked_process.run(prompt)
            
            # Add the result to our list
            forked_results.append({
                "id": i,
                "message": forked_response
            })
            
            if debug:
                print(f"FORK: Fork {i+1} completed with response length: {len(forked_response)}", file=sys.stderr)
        
        # Return all results from the forked processes
        result = {
            "results": forked_results,
        }
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Error forking process: {str(e)}"
        if debug:
            print(f"FORK ERROR: {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return {
            "error": error_msg,
            "is_error": True,
        }