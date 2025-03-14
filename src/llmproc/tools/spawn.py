"""Spawn tool for LLMProcess to interact with linked programs."""

import asyncio
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
    if not llm_process or not hasattr(llm_process, "linked_programs"):
        return {
            "error": "Spawn tool requires a parent LLMProcess with linked_programs defined",
            "is_error": True,
        }
    
    linked_programs = llm_process.linked_programs
    if program_name not in linked_programs:
        available_programs = ", ".join(linked_programs.keys())
        return {
            "error": f"Program '{program_name}' not found. Available programs: {available_programs}",
            "is_error": True,
        }
    
    try:
        # Get the linked program instance
        linked_program = linked_programs[program_name]
        
        # Execute the query on the linked program
        response = await linked_program.run(query)
        
        return {
            "program": program_name,
            "query": query,
            "response": response
        }
    except Exception as e:
        return {
            "error": f"Error executing query with program '{program_name}': {str(e)}",
            "is_error": True,
        }