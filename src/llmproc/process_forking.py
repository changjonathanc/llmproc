"""Forking helpers for :class:`LLMProcess`."""

from __future__ import annotations

import copy
import logging
import warnings
from typing import TYPE_CHECKING

from llmproc.common.access_control import AccessLevel
from llmproc.process_snapshot import ProcessSnapshot

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from llmproc.llm_process import LLMProcess

logger = logging.getLogger(__name__)


class ProcessForkingMixin:
    """Mixin providing process forking functionality."""

    async def _fork_process(self: LLMProcess, access_level: AccessLevel = AccessLevel.WRITE) -> LLMProcess:
        """Return a deep copy of this process with preserved state.

        This follows the Unix fork model by creating a new process through
        :func:`create_process` and copying runtime state from the parent.

        Args:
            access_level: Access level to set for the child process.

        Returns:
            A new ``LLMProcess`` instance that is a copy of this one.

        Raises:
            RuntimeError: If this process does not have ``ADMIN`` access.
        """
        if not hasattr(self, "access_level") or self.access_level != AccessLevel.ADMIN:
            raise RuntimeError("Forking requires ADMIN access level and is not allowed for this process")

        display_name = "unknown"
        try:
            if hasattr(self.program, "display_name") and self.program.display_name:
                display_name = self.program.display_name
            elif hasattr(self, "model_name") and self.model_name:
                display_name = self.model_name
        except (AttributeError, TypeError):
            pass

        logger.info("Forking process for program: %s", display_name)

        from llmproc.program_exec import create_process

        forked_process = await create_process(self.program)

        snapshot = ProcessSnapshot(
            state=copy.deepcopy(self.state),
            enriched_system_prompt=getattr(self, "enriched_system_prompt", None),
        )

        if hasattr(forked_process, "_apply_snapshot"):
            forked_process._apply_snapshot(snapshot)
        else:  # pragma: no cover - degraded mode for heavily mocked objects
            forked_process.state = snapshot.state
            forked_process.enriched_system_prompt = snapshot.enriched_system_prompt

        if getattr(self, "file_descriptor_enabled", False) and getattr(self, "fd_manager", None):
            forked_process.file_descriptor_enabled = True
            try:
                forked_process.fd_manager = self.fd_manager.clone()
            except AttributeError:
                forked_process.fd_manager = copy.deepcopy(self.fd_manager)

            forked_process.references_enabled = getattr(self, "references_enabled", False)

        if hasattr(self, "callbacks") and self.callbacks:
            forked_process.callbacks = self.callbacks.copy()

        if hasattr(forked_process, "tool_manager"):
            forked_process.tool_manager.set_process_access_level(access_level)
            logger.debug("Set access level for forked process to %s", access_level.value)

        forked_display = "unknown"
        try:
            if hasattr(forked_process, "display_name") and forked_process.display_name:
                forked_display = forked_process.display_name
            elif hasattr(forked_process, "model_name") and forked_process.model_name:
                forked_display = forked_process.model_name
        except (AttributeError, TypeError):
            pass

        logger.info(
            "Fork successful. New process created for %s with %s access",
            forked_display,
            access_level.value,
        )
        return forked_process

    async def fork_process(self: LLMProcess, access_level: AccessLevel = AccessLevel.WRITE) -> LLMProcess:
        """Return a deep copy of this process with preserved state.

        This method is deprecated. Use :meth:`_fork_process` instead.
        """
        warnings.warn(
            "fork_process is deprecated and will be removed in a future version. Use _fork_process instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self._fork_process(access_level)

    def _apply_snapshot(self: LLMProcess, snapshot: ProcessSnapshot) -> None:
        """Replace this process's conversation state with ``snapshot``."""
        self.state = snapshot.state
        if snapshot.enriched_system_prompt is not None:
            self.enriched_system_prompt = snapshot.enriched_system_prompt
