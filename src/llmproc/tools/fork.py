"""Fork system call for LLMProcess to create a copy of the current process."""

import asyncio
import copy
import os
import sys
from typing import Any, Dict, List, Optional

from llmproc.llm_process import LLMProcess


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