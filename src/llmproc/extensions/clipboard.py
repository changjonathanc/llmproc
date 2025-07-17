from __future__ import annotations

import re
from pathlib import Path

from llmproc.common.results import ToolResult
from llmproc.tools.function_tools import register_tool


class ClipboardPlugin:
    """Simple clipboard plugin with copy and paste tools."""

    def __init__(self) -> None:
        self.clipboard: str = ""

    def fork(self) -> ClipboardPlugin:
        clone = ClipboardPlugin()
        clone.clipboard = self.clipboard
        return clone

    @register_tool(
        name="copy",
        description="Copy text from file between start and end markers to clipboard.",
        param_descriptions={
            "file_path": "Path to the file to read",
            "start": "Start marker string",
            "end": "End marker string",
        },
        required=["file_path", "start", "end"],
    )
    async def copy_tool(self, file_path: str, start: str, end: str) -> ToolResult:
        try:
            text = Path(file_path).read_text()
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult.from_error(f"Error reading file: {exc}")

        pattern = re.escape(start) + r"(.*?)" + re.escape(end)
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if not matches:
            return ToolResult.from_error("No matching text found")
        if len(matches) > 1:
            return ToolResult.from_error("Multiple matching regions found")

        matched = matches[0].group(0)
        self.clipboard = matched
        return ToolResult.from_success(matched)

    @register_tool(
        name="paste",
        description="Paste clipboard content into a file at the specified position.",
        param_descriptions={
            "file_path": "Path to the file to modify",
            "before": "Text immediately before the cursor position",
            "after": "Text immediately after the cursor position",
        },
        required=["file_path"],
    )
    async def paste_tool(self, file_path: str, before: str = "", after: str = "") -> ToolResult:
        if not self.clipboard:
            return ToolResult.from_error("Clipboard is empty")

        try:
            path = Path(file_path)
            text = path.read_text()
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult.from_error(f"Error reading file: {exc}")

        insert_idx = 0
        if before and after:
            pattern = re.escape(before) + re.escape(after)
            occurrences = list(re.finditer(pattern, text))
            if not occurrences:
                return ToolResult.from_error("Before/after pattern not found")
            if len(occurrences) > 1:
                return ToolResult.from_error("Multiple positions match before/after")
            insert_idx = occurrences[0].start() + len(before)
        elif before:
            idx = text.find(before)
            if idx == -1:
                return ToolResult.from_error("Before string not found")
            insert_idx = idx + len(before)
        elif after:
            idx = text.find(after)
            if idx == -1:
                return ToolResult.from_error("After string not found")
            insert_idx = idx
        else:
            if text:
                return ToolResult.from_error("Target file must be empty when no position is specified")
            insert_idx = 0

        new_text = text[:insert_idx] + self.clipboard + text[insert_idx:]
        try:
            path.write_text(new_text)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult.from_error(f"Error writing file: {exc}")

        return ToolResult.from_success(self.clipboard)

    def hook_provide_tools(self) -> list:
        return [self.copy_tool, self.paste_tool]


# Module-level aliases
copy_tool = ClipboardPlugin.copy_tool
paste_tool = ClipboardPlugin.paste_tool
