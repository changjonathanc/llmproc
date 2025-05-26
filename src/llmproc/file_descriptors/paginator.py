"""Pagination utilities for the file descriptor system.

This module handles line-aware pagination and content extraction for file descriptors.
It enables efficient access to large content by page, line, or character position.
"""

from typing import Any


def index_lines(content: str) -> tuple[list[int], int]:
    """Create an index of line start positions.

    Args:
        content: The content to index

    Returns:
        Tuple of (list of line start indices, total line count)
    """
    lines = [0]  # First line always starts at index 0
    for i, char in enumerate(content):
        if char == "\n" and i + 1 < len(content):
            lines.append(i + 1)

    return lines, len(lines)


def get_page_content(content: str, lines: list[int], page_size: int, start_pos: int) -> tuple[str, dict[str, Any]]:
    """Get content for a specific page position with line-aware pagination.

    Args:
        content: The full content string
        lines: List of line start indices
        page_size: Maximum characters per page
        start_pos: The starting page position (1-based)

    Returns:
        Tuple of (content, position information)
    """
    total_lines = len(lines)

    # Calculate page boundaries
    start_char = (start_pos - 1) * page_size

    # Handle case where start_char is beyond the content length
    if start_char >= len(content):
        # Return empty content with info showing we're beyond content
        return "", {
            "start_line": total_lines,
            "end_line": total_lines,
            "continued": False,
            "truncated": False,
            "empty_page": True,
        }

    end_char = min(start_char + page_size, len(content))

    # Find line boundaries for better pagination
    start_line = 1
    end_line = 1
    continued = False
    truncated = False

    # Find the start line (the line containing start_char)
    for i, line_start in enumerate(lines):
        if line_start > start_char:
            start_line = i  # The previous line
            break
        start_line = i + 1

    # Check if we're continuing from previous page (not starting at line boundary)
    if start_char > 0 and start_line > 1 and start_char != lines[start_line - 1]:
        continued = True

    # Find the end line (the line containing or after end_char)
    for i, line_start in enumerate(lines):
        if line_start >= end_char:
            end_line = i  # The previous line
            break
        end_line = i + 1

    # Check if we're truncating (not ending at line boundary)
    next_line_start = len(content)
    if end_line < len(lines):
        next_line_start = lines[end_line]

    if end_char < next_line_start:
        truncated = True

    # Extract the actual content for this section
    section_content = content[start_char:end_char]

    # Return content and metadata
    position_info = {
        "start_line": start_line,
        "end_line": end_line,
        "continued": continued,
        "truncated": truncated,
    }

    return section_content, position_info


def calculate_total_pages(content: str, lines: list[int], page_size: int) -> int:
    """Calculate the total number of pages in content using line-aware pagination.

    This simulates the line-aware pagination algorithm to get an accurate page count.

    Args:
        content: The content to paginate
        lines: List of line start indices
        page_size: Maximum characters per page

    Returns:
        The total number of pages
    """
    # For very small content, just return 1 page
    if len(content) <= page_size:
        return 1

    # For larger content, iterate through the pages
    start_char = 0
    page_count = 0

    while start_char < len(content):
        page_count += 1

        # Calculate end of current page
        end_char = min(start_char + page_size, len(content))

        # Find the end line for this page
        end_line = 1

        # Find the end line for this page
        for i, line_start in enumerate(lines):
            if line_start >= end_char:
                end_line = i  # The previous line
                break
            end_line = i + 1

        # Determine the start of the next page
        if end_line < len(lines):
            start_char = lines[end_line]
        else:
            # No more lines, we're done
            break

    return page_count


def extract_content_by_mode(
    content: str,
    lines: list[int],
    mode: str,
    start: int,
    count: int,
    total_lines: int,
    page_size: int,
    total_pages: int,
) -> tuple[str, dict[str, Any]]:
    """Extract content from a string based on positioning mode.

    Args:
        content: The full content string
        lines: List of line start indices
        mode: Positioning mode: "page", "line", or "char"
        start: Starting position in the specified mode's units
        count: Number of units to read
        total_lines: Total number of lines in the content
        page_size: Size of each page in characters
        total_pages: Total number of pages

    Returns:
        Tuple of (extracted content, position metadata)
    """
    # Line-based positioning
    if mode == "line":
        # Validate line range
        if start < 1 or start > total_lines:
            raise ValueError(f"Invalid line start position. Valid range: 1-{total_lines}")

        end_line = min(start + count - 1, total_lines)

        # Get content by line range
        line_start_index = lines[start - 1]  # Convert to 0-indexed

        # Handle the end line index
        if end_line >= len(lines):
            # Read to the end of the content
            line_end_index = len(content)
        else:
            line_end_index = lines[end_line]  # End index is start of next line

        content_to_return = content[line_start_index:line_end_index]

        # Create the response metadata
        metadata = {
            "pages": total_pages,
            "continued": False,
            "truncated": False,
            "lines": f"{start}-{end_line}",
            "total_lines": total_lines,
            "mode": "line",
            "start": start,
            "count": end_line - start + 1,
        }

        return content_to_return, metadata

    # Character-based positioning
    elif mode == "char":
        content_length = len(content)

        # Validate char range
        if start < 0 or start >= content_length:
            raise ValueError(f"Invalid character start position. Valid range: 0-{content_length - 1}")

        end_char = min(start + count, content_length)

        # Extract the content range
        content_to_return = content[start:end_char]

        # For line numbering in metadata, find the lines that contain these characters
        # Find the line number for the start character
        start_line_num = 1
        for i, line_start in enumerate(lines):
            if line_start > start:
                start_line_num = i  # Previous line contains the start character
                break
            start_line_num = i + 1

        # Find the line number for the end character
        end_line_num = start_line_num
        for i in range(start_line_num - 1, len(lines)):
            if i + 1 < len(lines) and lines[i + 1] > end_char:
                end_line_num = i + 1
                break
            if i + 1 == len(lines):
                end_line_num = total_lines

        # Create the response metadata
        metadata = {
            "pages": total_pages,
            "continued": False,
            "truncated": False,
            "lines": f"{start_line_num}-{end_line_num}",
            "total_lines": total_lines,
            "mode": "char",
            "start": start,
            "count": end_char - start,
        }

        return content_to_return, metadata

    # Default page-based positioning
    else:  # mode == "page"
        # Validate page number
        if start < 1 or start > total_pages:
            raise ValueError(f"Invalid page number. Valid range: 1-{total_pages}")

        # Handle multi-page ranges
        if count > 1:
            end_page = min(start + count - 1, total_pages)

            # Collect content from all pages in the range
            all_content = []
            first_page_info = None
            last_page_info = None

            for p in range(start, end_page + 1):
                section_content, position_info = get_page_content(content, lines, page_size, p)
                all_content.append(section_content)

                if p == start:
                    first_page_info = position_info
                if p == end_page:
                    last_page_info = position_info

            content_to_return = "".join(all_content)

            # Create the response metadata for multi-page
            metadata = {
                "pages": total_pages,
                "continued": first_page_info.get("continued", False),
                "truncated": last_page_info.get("truncated", False),
                "lines": f"{first_page_info.get('start_line', 1)}-{last_page_info.get('end_line', 1)}",
                "total_lines": total_lines,
                "mode": "page",
                "start": start,
                "count": count,
            }

            return content_to_return, metadata

        else:
            # Single page case
            content_to_return, position_info = get_page_content(content, lines, page_size, start)

            # Create the response metadata
            metadata = {
                "page": start,
                "pages": total_pages,
                "continued": position_info.get("continued", False),
                "truncated": position_info.get("truncated", False),
                "lines": f"{position_info.get('start_line', 1)}-{position_info.get('end_line', 1)}",
                "total_lines": total_lines,
                "mode": "page",
                "start": start,
                "count": 1,
            }

            return content_to_return, metadata
