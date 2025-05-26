"""Reference handling for the file descriptor system.

This module handles the extraction and management of references from
assistant messages. References allow content to be marked up in responses
and later referenced by reference ID.
"""

import logging
import re
import time
from typing import Any

from llmproc.file_descriptors.paginator import calculate_total_pages

# Set up logger
logger = logging.getLogger(__name__)


def extract_references(
    assistant_message: str,
    file_descriptors: dict[str, dict[str, Any]],
    default_page_size: int,
    index_lines_func=None,
) -> list[dict[str, Any]]:
    """Extract references from an assistant message and store them in the FD system.

    Args:
        assistant_message: The message from the assistant to process
        file_descriptors: Dictionary to store the created file descriptors
        default_page_size: Default page size for pagination
        index_lines_func: Function to index lines (dependency injection for testing)

    Returns:
        A list of dictionaries containing reference information (id, content)
    """
    # If index_lines_func wasn't provided, use local implementation
    if index_lines_func is None:

        def default_index_lines(content: str) -> tuple[list[int], int]:
            lines = [0]  # First line always starts at index 0
            for i, char in enumerate(content):
                if char == "\n" and i + 1 < len(content):
                    lines.append(i + 1)

            return lines, len(lines)

        index_lines_func = default_index_lines

    # Use an optimized regex pattern for better performance
    # This pattern is non-greedy and handles attributes more efficiently
    # It also caches the compiled regex pattern for repeated use
    pattern = re.compile(r'<ref\s+id="([^"]+)"[^>]*>(.*?)</ref>', re.DOTALL)

    # Process the message looking for references
    # First find all potential matches (using the pre-compiled pattern)
    all_matches = list(pattern.finditer(assistant_message))

    # Map to store processed references by ID
    ref_map = {}

    # Process matches to handle nesting and duplicates
    for match in all_matches:
        ref_id = match.group(1)
        content = match.group(2)
        start_pos = match.start()
        end_pos = match.end()

        # Store the reference with position info
        ref_map[ref_id] = {
            "id": ref_id,
            "content": content,
            "start": start_pos,
            "end": end_pos,
            "full_tag": match.group(0),  # The complete tag including content
        }

    # Process nested references in O(n log n) time
    # Algorithm:
    # 1. Sort references by start position (and secondarily by end position in reverse)
    # 2. Use a stack to track potential "container" references
    # 3. For each reference, check if it's nested inside any reference currently on the stack
    # 4. Only add non-nested references to the stack (as they may contain others)
    refs_sorted = sorted(ref_map.items(), key=lambda x: (x[1]["start"], -x[1]["end"]))
    nested_refs = set()
    stack = []  # Stack contains only outermost references that may contain others

    for ref_id, ref_data in refs_sorted:
        # Pop refs that can't possibly contain current ref (they end too early)
        while stack and stack[-1][1]["end"] < ref_data["start"]:
            stack.pop()

        if stack:
            # If stack isn't empty, this ref is nested inside another ref
            # (because there's at least one ref that starts before this one and ends after it starts)
            nested_refs.add(ref_id)
        else:
            # Only non-nested refs are added to stack as potential containers
            stack.append((ref_id, ref_data))

    # Remove all nested refs
    for nested_id in nested_refs:
        del ref_map[nested_id]

    # Convert to the format used by the original function
    matches = [(ref_id, ref_data["content"]) for ref_id, ref_data in ref_map.items()]
    references = []

    for ref_id, content in matches:
        # Store the reference in the file descriptor system
        # with the ref: prefix to distinguish it from regular file descriptors
        fd_id = f"ref:{ref_id}"

        # Create a file descriptor with the reference ID as the FD ID
        # Note: Even if the reference already exists, we override it with the new content
        # following a "last one wins" policy for duplicate references

        # Check if we're overwriting an existing reference
        if fd_id in file_descriptors:
            logger.warning(f"Overwriting existing reference '{ref_id}' with new content")

        # Index the content for line-aware pagination
        lines, total_lines = index_lines_func(content)
        page_size = default_page_size

        # Calculate the total number of pages using the standard pagination function
        # This ensures references use the same pagination logic as regular file descriptors,
        # preserving line boundaries and maintaining consistent page sizes
        num_pages = calculate_total_pages(content, lines, page_size)

        # Store or update the file descriptor entry
        file_descriptors[fd_id] = {
            "content": content,
            "lines": lines,
            "total_lines": total_lines,
            "page_size": page_size,
            "creation_time": time.time(),
            "source": "reference",
            "total_pages": num_pages,
        }

        # Store information about the reference
        references.append({"id": ref_id, "fd_id": fd_id, "content": content, "length": len(content)})

        logger.info(f"Created reference '{ref_id}' with {len(content)} characters")

    return references


def format_user_input_reference(user_input: str, fd_id: str, max_preview_chars: int = 150) -> str:
    """Format a user input reference for display to the LLM.

    Args:
        user_input: The original user input
        fd_id: The file descriptor ID
        max_preview_chars: Maximum characters to show in preview

    Returns:
        Formatted reference string
    """
    # Generate a preview with approximately max_preview_chars characters
    preview = user_input[:max_preview_chars].strip()
    if len(user_input) > max_preview_chars:
        preview += "..."

    # Format a user message that references the file descriptor
    formatted_message = (
        f'<fd:{fd_id} preview="{preview}" type="user_input" size="{len(user_input)}">\n'
        f"This large user input has been stored in file descriptor {fd_id}. "
        f'Use read_fd(fd="{fd_id}") to access the content.'
    )

    logger.info(f"Large user input ({len(user_input)} chars) stored in {fd_id}. Preview: {preview[:50]}...")

    return formatted_message
