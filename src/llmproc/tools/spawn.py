"""Spawn tool for LLMProcess to interact with linked programs."""

import asyncio
import os
import sys
from typing import Any, Dict, Optional

from llmproc.llm_process import LLMProcess


async def spawn_tool(
    program_name: str,
    query: str,
    llm_process: Optional[LLMProcess] = None,
) -> Dict[str, Any]:
    """Execute a query with a linked LLM program.
    
    This tool allows one LLM process to spawn a query to another linked LLM process
    that may be specialized for specific tasks.
    
    Args:
        program_name: The name of the linked program to call
        query: The query to send to the linked program
        llm_process: The parent LLMProcess instance with access to linked programs
        
    Returns:
        A dictionary with the response from the linked program
        
    Raises:
        ValueError: If the program_name is not found in linked programs
    """
    import sys
    debug = getattr(llm_process, 'debug_tools', False) and os.environ.get("LLMPROC_DEBUG", "").lower() == "true"
    
    if not llm_process or not hasattr(llm_process, "linked_programs"):
        error_msg = "Spawn tool requires a parent LLMProcess with linked_programs defined"
        if debug:
            print(f"SPAWN ERROR: {error_msg}", file=sys.stderr)
        return {
            "error": error_msg,
            "is_error": True,
        }
    
    linked_programs = llm_process.linked_programs
    if program_name not in linked_programs:
        available_programs = ", ".join(linked_programs.keys())
        error_msg = f"Program '{program_name}' not found. Available programs: {available_programs}"
        if debug:
            print(f"SPAWN ERROR: {error_msg}", file=sys.stderr)
        return {
            "error": error_msg,
            "is_error": True,
        }
    
    try:
        # Get the linked program instance
        linked_program = linked_programs[program_name]
        
        # Execute the query on the linked program
        response = await linked_program.run(query)
        
        result = {
            "program": program_name,
            "query": query,
            "response": response
        }
        return result
    except Exception as e:
        import traceback
        error_msg = f"Error executing query with program '{program_name}': {str(e)}"
        if debug:
            print(f"SPAWN ERROR: {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return {
            "error": error_msg,
            "is_error": True,
        }