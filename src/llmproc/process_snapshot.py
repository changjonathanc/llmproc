"""Lightweight immutable snapshot of an LLMProcess conversation state.

This object is meant for *internal* use by `LLMProcess.fork_process` (and
potential future features such as time‑travel) so that the details of how we
serialise state are not scattered across the code‑base.

Implementation Note:
This provides a clean, immutable representation of process state that can be used when forking
processes. It helps ensure proper isolation between parent and child processes
by providing a frozen snapshot at the fork point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    """Conversation slice that can be applied to a fresh LLMProcess."""

    state: list[Any] = field(default_factory=list)
    enriched_system_prompt: str | None = None
    fd_manager_state: dict | None = None
