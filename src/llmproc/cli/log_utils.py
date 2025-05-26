from __future__ import annotations

import json
import logging
from typing import Any


def log_program_info(process: Any, user_message: str | None = None, logger: logging.Logger | None = None) -> None:
    """Log program details before the first API call.

    Args:
        process: Process instance containing program state.
        user_message: Optional first user message that will be sent.
        logger: Logger for output. Defaults to ``logging.getLogger('llmproc.cli')``.
    """
    logger = logger or logging.getLogger("llmproc.cli")

    # Tools configuration
    tools = getattr(process, "tools", [])
    try:
        tools_dump = json.dumps(tools, indent=2)
    except Exception:
        tools_dump = str(tools)
    logger.info("Tools:\n%s", tools_dump)

    # Enriched system prompt
    system_prompt = getattr(process, "enriched_system_prompt", "")
    if system_prompt:
        logger.info("Enriched System Prompt:\n%s", system_prompt)

    # First user message if provided
    if user_message:
        logger.info("First User Message:\n%s", user_message)

    # Request payload (model + API params)
    payload = {"model": getattr(process, "model_name", "")}
    api_params = getattr(process, "api_params", {})
    if isinstance(api_params, dict):
        payload.update(api_params)
    try:
        payload_dump = json.dumps(payload, indent=2)
    except Exception:
        payload_dump = str(payload)
    logger.info("Request Payload:\n%s", payload_dump)
