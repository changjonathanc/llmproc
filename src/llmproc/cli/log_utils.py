from __future__ import annotations

import json
import logging
import sys
from typing import Any


class CleanFormatter(logging.Formatter):
    """Custom formatter that omits the WARNING: prefix for warning messages."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        original_levelname = record.levelname
        if record.levelname == "WARNING":
            record.levelname = ""
        result = super().format(record)
        record.levelname = original_levelname
        return result


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configure and return a logger with the custom formatter."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logger = logging.getLogger("llmproc.cli")

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.__stderr__)
    formatter = CleanFormatter("%(asctime)s - %(levelname)s %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    root_logger.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("llmproc.llm_process").setLevel(level)

    if level >= logging.ERROR:
        import warnings

        warnings.filterwarnings("ignore", module="pydantic")

    return logger


def get_logger(log_level: str = "INFO") -> logging.Logger:
    """Public helper to obtain a configured logger."""
    return setup_logger(log_level)


class CliCallbackHandler:
    """Callback implementation for CLI output."""

    def __init__(self, logger: logging.Logger) -> None:
        self.turn = 0
        self.logger = logger

    def tool_start(self, tool_name: str, args: Any) -> None:
        self.logger.info(json.dumps({"tool_start": {"tool_name": tool_name, "args": args}}, indent=2))

    def tool_end(self, tool_name: str, result: Any) -> None:
        self.logger.info(json.dumps({"tool_end": {"tool_name": tool_name, "result": result.to_dict()}}, indent=2))

    def response(self, content: str) -> None:
        self.logger.info(json.dumps({"text response": content}, indent=2))

    def api_response(self, response: Any) -> None:
        self.logger.info(json.dumps({"api response usage": response.usage.model_dump()}, indent=2))

    def stderr_write(self, text: str) -> None:
        self.logger.warning(json.dumps({"STDERR": text}, indent=2))

    async def turn_start(self, process: Any) -> None:
        self.turn += 1
        info = await process.count_tokens()
        self.logger.warning(f"--------- TURN {self.turn} start, token count {info['input_tokens']} --------")

    def turn_end(self, process: Any, response: Any, tool_results: Any) -> None:
        count = len(tool_results) if tool_results is not None else 0
        self.logger.warning(f"--------- TURN {self.turn} end, {count} tools used in this turn ----")


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
