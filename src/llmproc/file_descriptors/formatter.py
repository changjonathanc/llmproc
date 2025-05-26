"""XML formatting utilities for file descriptor results.

This module handles the formatting of file descriptor results and errors into
standardized XML formats for consistent presentation to the LLM.
"""

from typing import Any


def format_fd_result(result: dict[str, Any]) -> str:
    """Format a file descriptor result in XML format.

    Args:
        result: Dictionary with file descriptor information

    Returns:
        Formatted XML content
    """
    # Add source attribute if present
    source_attr = f' source="{result["source"]}"' if "source" in result else ""

    xml = (
        f'<fd_result fd="{result["fd"]}" pages="{result["pages"]}" '
        f'truncated="{str(result["truncated"]).lower()}" '
        f'lines="{result["lines"]}" total_lines="{result["total_lines"]}"'
        f"{source_attr}>\n"
        f"  <message>{result['message']}</message>\n"
        f"  <preview>\n"
        f"  {result['preview']}\n"
        f"  </preview>\n"
        f"</fd_result>"
    )

    return xml


def format_fd_content(content: dict[str, Any]) -> str:
    """Format file descriptor content in XML format.

    Args:
        content: Dictionary with content and metadata

    Returns:
        Formatted XML content
    """
    # Add additional attributes for mode, start, and count if present
    mode_attr = f' mode="{content["mode"]}"' if "mode" in content else ""
    start_attr = f' start="{content["start"]}"' if "start" in content else ""
    count_attr = f' count="{content["count"]}"' if "count" in content else ""

    # Handle the all-pages case differently
    if content.get("page") == "all":
        xml = (
            f'<fd_content fd="{content["fd"]}" page="all" pages="{content["pages"]}" '
            f'continued="false" truncated="false" '
            f'lines="{content["lines"]}" total_lines="{content["total_lines"]}"'
            f"{mode_attr}{start_attr}{count_attr}>\n"
            f"{content['content']}\n"
            f"</fd_content>"
        )
    elif "page" in content:
        # Page-based positioning
        xml = (
            f'<fd_content fd="{content["fd"]}" page="{content["page"]}" '
            f'pages="{content["pages"]}" '
            f'continued="{str(content["continued"]).lower()}" '
            f'truncated="{str(content["truncated"]).lower()}" '
            f'lines="{content["lines"]}" total_lines="{content["total_lines"]}"'
            f"{mode_attr}{start_attr}{count_attr}>\n"
            f"{content['content']}\n"
            f"</fd_content>"
        )
    else:
        # Line or char based positioning
        xml = (
            f'<fd_content fd="{content["fd"]}" '
            f'pages="{content["pages"]}" '
            f'continued="{str(content.get("continued", False)).lower()}" '
            f'truncated="{str(content.get("truncated", False)).lower()}" '
            f'lines="{content["lines"]}" total_lines="{content["total_lines"]}"'
            f"{mode_attr}{start_attr}{count_attr}>\n"
            f"{content['content']}\n"
            f"</fd_content>"
        )

    return xml


def format_fd_file_result(result: dict[str, Any]) -> str:
    """Format file descriptor file operation result in XML format.

    Args:
        result: Dictionary with file operation result information

    Returns:
        Formatted XML content
    """
    # Include create and exist_ok attributes if present
    create_attr = f' create="{str(result.get("create", True)).lower()}"' if "create" in result else ""
    exist_ok_attr = f' exist_ok="{str(result.get("exist_ok", True)).lower()}"' if "exist_ok" in result else ""

    xml = (
        f'<fd_file_result fd="{result["fd"]}" file_path="{result["file_path"]}" '
        f'mode="{result["mode"]}" char_count="{result["char_count"]}" '
        f'size_bytes="{result["size_bytes"]}" success="{str(result["success"]).lower()}"'
        f"{create_attr}{exist_ok_attr}>\n"
        f"  <message>{result['message']}</message>\n"
        f"</fd_file_result>"
    )

    return xml


def format_fd_extraction(result: dict[str, Any]) -> str:
    """Format file descriptor extraction result in XML format.

    Args:
        result: Dictionary with extraction result information

    Returns:
        Formatted XML content
    """
    # Common attributes for all extraction results
    attributes = [
        f'source_fd="{result["source_fd"]}"',
        f'new_fd="{result["new_fd"]}"',
        f'mode="{result["mode"]}"',
        f'content_size="{result["content_size"]}"',
    ]

    # Add position/range information based on provided data
    if "position" in result:
        attributes.append(f'position="{result["position"]}"')
    if "start" in result:
        attributes.append(f'start="{result["start"]}"')
    if "count" in result:
        attributes.append(f'count="{result["count"]}"')

    # For backwards compatibility, if we're in page mode, also include page attribute
    if result["mode"] == "page" and "start" in result:
        attributes.append(f'page="{result["start"]}"')

    # Combine attributes and create the XML
    xml = f"<fd_extraction {' '.join(attributes)}>\n  <message>{result['message']}</message>\n</fd_extraction>"

    return xml


def format_fd_error(error_type: str, fd_id: str, message: str) -> str:
    """Format a file descriptor error in XML format.

    Args:
        error_type: Type of error (e.g., "not_found", "invalid_page")
        fd_id: The file descriptor ID
        message: Error message

    Returns:
        Formatted XML error content
    """
    xml = f'<fd_error type="{error_type}" fd="{fd_id}">\n  <message>{message}</message>\n</fd_error>'

    return xml
