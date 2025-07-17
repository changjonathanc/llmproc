from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_path(base_dir: Path, file_path_str: str) -> Path:
    """Resolve ``file_path_str`` relative to ``base_dir``."""
    path = Path(file_path_str)
    if not path.is_absolute():
        return (base_dir / path).resolve()
    return path.resolve()


def _warn_preload(message: str, specified_path: str, resolved_path: Path, error: Optional[Exception] = None) -> None:
    """Issue a consistent warning for preload file issues."""
    full = f"{message} - Specified: '{specified_path}', Resolved: '{resolved_path}'"
    if error:
        full += f", Error: {error}"
    warnings.warn(full, stacklevel=3)


def load_files(file_paths: list[str], base_dir: Optional[Path] = None) -> dict[str, str]:
    """Load content from a list of files."""
    if not file_paths:
        return {}

    base_dir = base_dir or Path(".")
    content: dict[str, str] = {}

    for file_path_str in file_paths:
        path = _resolve_path(base_dir, file_path_str)
        try:
            if not path.exists():
                _warn_preload("Preload file not found", file_path_str, path)
                continue
            if not path.is_file():
                _warn_preload("Preload path is not a file", file_path_str, path)
                continue
            content[str(path)] = path.read_text()
            logger.debug("Successfully preloaded content from: %s", path)
        except OSError as e:
            _warn_preload("Error reading preload file", file_path_str, path, e)
        except Exception as e:  # pragma: no cover - defensive
            _warn_preload("Unexpected error preloading file", file_path_str, path, e)

    return content


def build_preload_content(preloaded: dict[str, str]) -> str:
    """Return formatted preload content block."""
    if not preloaded:
        return ""
    files = [f'<file path="{Path(file_path).name}">\n{content}\n</file>' for file_path, content in preloaded.items()]
    return "<preload>\n" + "\n".join(files) + "\n</preload>"


class PreloadFilesPlugin:
    """Plugin that adds file contents to the system prompt."""

    def __init__(
        self,
        file_paths: Optional[list[str]] = None,
        base_dir: Optional[Path] = None,
        relative_to: str = "program",
    ) -> None:
        self.file_paths: list[str] = file_paths or []
        if base_dir is None:
            base_dir = Path(".")
        self.base_dir = base_dir
        self.relative_to = relative_to
        self._preloaded: Optional[str] = None
        self.validate()

    def fork(self) -> PreloadFilesPlugin:
        """Return ``self`` for a forked process."""
        return self

    def extend_files(self, more_files: Optional[list[str]]) -> None:
        """Extend the list of files to preload."""
        if not more_files:
            return
        self.file_paths.extend(more_files)
        # Reset cached content so new files are loaded on next hook
        self._preloaded = None

    def validate(self) -> None:
        """Load file content and cache the formatted block."""
        base = Path.cwd() if self.relative_to == "cwd" else self.base_dir
        content = load_files(self.file_paths, base)
        self._preloaded = build_preload_content(content)

    async def hook_system_prompt(self, system_prompt: str, process) -> str | None:
        if self._preloaded is None:
            self.validate()
        if self._preloaded:
            return f"{system_prompt}\n\n{self._preloaded}"
        return None
