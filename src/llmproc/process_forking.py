"""Forking helpers for :class:`LLMProcess`."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import TYPE_CHECKING

from llmproc.common.access_control import AccessLevel
from llmproc.plugin.plugin_event_runner import PluginEventRunner
from llmproc.process_snapshot import ProcessSnapshot

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    # Imported here to avoid circular dependency with llm_process
    from llmproc.llm_process import LLMProcess

logger = logging.getLogger(__name__)


class ProcessForkingMixin:
    """Mixin providing process forking functionality."""

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _create_snapshot(self: LLMProcess) -> ProcessSnapshot:
        """Return a snapshot of this process's state."""
        return ProcessSnapshot(
            state=copy.deepcopy(self.state),
            enriched_system_prompt=getattr(self, "enriched_system_prompt", None),
        )

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

        logger.info("Forking process for program: %s", self.program.model_name)

        from llmproc.program_exec import create_process

        forked_process = await create_process(self.program)
        if isinstance(forked_process, asyncio.Future):  # pragma: no cover - test support
            forked_process = await forked_process

        snapshot = self._create_snapshot()

        if hasattr(forked_process, "_apply_snapshot"):
            forked_process._apply_snapshot(snapshot)
        else:  # pragma: no cover - degraded mode for heavily mocked objects
            forked_process.state = snapshot.state
            forked_process.enriched_system_prompt = snapshot.enriched_system_prompt

        # Clone plugins using their fork() method when available
        cloned_plugins = []
        for plugin in self.plugins:
            if hasattr(plugin, "fork"):
                try:
                    cloned = plugin.fork()
                    if cloned is not None:
                        cloned_plugins.append(cloned)
                except Exception as e:  # pragma: no cover - defensive
                    logger.error("Failed to fork plugin %s: %s", plugin, e)
            else:
                logger.debug("Skipping plugin %s during fork; no fork() method", plugin)
        forked_process.plugins = PluginEventRunner(forked_process._submit_to_loop, cloned_plugins)
        from llmproc.program_exec import setup_runtime_context

        setup_runtime_context(forked_process)

        if hasattr(forked_process, "tool_manager"):
            forked_process.tool_manager.set_process_access_level(access_level)
            logger.debug("Set access level for forked process to %s", access_level.value)

        logger.info(
            "Fork successful. New process created for %s with %s access",
            forked_process.model_name,
            access_level.value,
        )
        return forked_process

    def _apply_snapshot(self: LLMProcess, snapshot: ProcessSnapshot) -> None:
        """Replace this process's conversation state with ``snapshot``."""
        self.state = snapshot.state
        if snapshot.enriched_system_prompt is not None:
            self.enriched_system_prompt = snapshot.enriched_system_prompt
