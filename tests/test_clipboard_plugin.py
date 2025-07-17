"""Tests for the ClipboardPlugin copy and paste tools.

Covered behaviors:
1. `copy_tool` returns the unique region between start and end markers.
2. Errors when markers appear multiple times or are missing.
3. `paste_tool` inserts between ``before`` and ``after`` markers.
4. Errors when the clipboard is empty or markers are not found.
5. Pasting without specifying a position only works on an empty file.
"""

from pathlib import Path

import pytest

from llmproc.extensions.clipboard import ClipboardPlugin


@pytest.mark.asyncio
async def test_copy_single_match(tmp_path: Path) -> None:
    """Copy a single matching region into the clipboard."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("AAA START copy this END BBB")

    plugin = ClipboardPlugin()
    result = await plugin.copy_tool(file_path=str(file_path), start="START", end="END")

    assert not result.is_error
    assert plugin.clipboard == "START copy this END"


@pytest.mark.asyncio
async def test_copy_multiple_matches(tmp_path: Path) -> None:
    """Fail when more than one region matches the markers."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("START one END START two END")

    plugin = ClipboardPlugin()
    result = await plugin.copy_tool(file_path=str(file_path), start="START", end="END")

    assert result.is_error


@pytest.mark.asyncio
async def test_paste_between_markers(tmp_path: Path) -> None:
    """Insert clipboard text between a before and after marker."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("beforeAFTER")

    plugin = ClipboardPlugin()
    plugin.clipboard = "-CLIP-"

    result = await plugin.paste_tool(
        file_path=str(file_path), before="before", after="AFTER"
    )

    assert not result.is_error
    assert file_path.read_text() == "before-CLIP-AFTER"


@pytest.mark.asyncio
async def test_copy_no_match(tmp_path: Path) -> None:
    """Return error when markers are not found in the file."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("no markers here")

    plugin = ClipboardPlugin()
    result = await plugin.copy_tool(file_path=str(file_path), start="START", end="END")

    assert result.is_error


@pytest.mark.asyncio
async def test_paste_clipboard_empty(tmp_path: Path) -> None:
    """Fail when attempting to paste with an empty clipboard."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")

    plugin = ClipboardPlugin()
    result = await plugin.paste_tool(file_path=str(file_path))

    assert result.is_error


@pytest.mark.asyncio
async def test_paste_before_not_found(tmp_path: Path) -> None:
    """Fail when the before marker does not exist."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")

    plugin = ClipboardPlugin()
    plugin.clipboard = "CLIP"

    result = await plugin.paste_tool(file_path=str(file_path), before="missing")

    assert result.is_error


@pytest.mark.asyncio
async def test_paste_after_not_found(tmp_path: Path) -> None:
    """Fail when the after marker does not exist."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")

    plugin = ClipboardPlugin()
    plugin.clipboard = "CLIP"

    result = await plugin.paste_tool(file_path=str(file_path), after="missing")

    assert result.is_error


@pytest.mark.asyncio
async def test_paste_no_position_error_when_file_not_empty(tmp_path: Path) -> None:
    """Fail if file is not empty and no position is provided."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("ABC")

    plugin = ClipboardPlugin()
    plugin.clipboard = "X"

    result = await plugin.paste_tool(file_path=str(file_path))

    assert result.is_error
    assert file_path.read_text() == "ABC"


@pytest.mark.asyncio
async def test_paste_no_position_empty_file(tmp_path: Path) -> None:
    """Insert clipboard text when file is empty and no position is given."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("")

    plugin = ClipboardPlugin()
    plugin.clipboard = "X"

    result = await plugin.paste_tool(file_path=str(file_path))

    assert not result.is_error
    assert file_path.read_text() == "X"
