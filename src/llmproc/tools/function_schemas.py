"""Docstring parsing and schema generation utilities for function tools."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from llmproc.common.metadata import get_tool_meta
from llmproc.tools.type_conversion import type_to_json_schema

logger = logging.getLogger(__name__)


def extract_docstring_params(func: Callable) -> dict[str, dict[str, str]]:
    """Extract parameter descriptions from a function's docstring."""
    docstring = inspect.getdoc(func)
    if not docstring:
        return {}

    params: dict[str, dict[str, str]] = {}

    args_match = re.search(r"Args:(.*?)(?:\n\n|\n\w+:|\Z)", docstring, re.DOTALL)
    if args_match:
        args_text = args_match.group(1)
        param_matches = re.finditer(r"\n\s+(\w+):\s*(.*?)(?=\n\s+\w+:|$)", args_text, re.DOTALL)
        for match in param_matches:
            param_name = match.group(1)
            param_desc = match.group(2).strip()
            params[param_name] = {"description": param_desc}

    returns_match = re.search(r"Returns:(.*?)(?:\n\n|\n\w+:|\Z)", docstring, re.DOTALL)
    if returns_match:
        return_desc = returns_match.group(1).strip()
        params["return"] = {"description": return_desc}

    return params


def function_to_tool_schema(func: Callable) -> dict[str, Any]:
    """Convert a function to a tool schema."""
    meta = get_tool_meta(func)
    if meta.custom_schema:
        return meta.custom_schema

    func_name = meta.name or func.__name__
    schema = {
        "name": func_name,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }

    docstring = inspect.getdoc(func)
    if meta.description:
        schema["description"] = meta.description
    elif docstring:
        first_line = docstring.split("\n", 1)[0].strip()
        schema["description"] = first_line
    else:
        schema["description"] = f"Tool for {func_name}"

    docstring_params = extract_docstring_params(func)
    explicit_descriptions = meta.param_descriptions
    type_hints = get_type_hints(func)
    sig = inspect.signature(func)

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "runtime_context"):
            continue
        param_type = type_hints.get(param_name, Any)
        param_schema = type_to_json_schema(param_type, param_name, docstring_params, explicit_descriptions)
        schema["input_schema"]["properties"][param_name] = param_schema
        if not meta.required_params and param.default is param.empty:
            schema["input_schema"]["required"].append(param_name)

    if meta.required_params:
        schema["input_schema"]["required"] = list(meta.required_params)

    return schema


def _apply_parameter_descriptions(schema: dict, param_descriptions: dict[str, str]) -> dict:
    """Apply parameter description overrides to schema properties."""
    if not param_descriptions:
        return schema

    if "input_schema" not in schema:
        schema["input_schema"] = {}
    if "properties" not in schema["input_schema"]:
        schema["input_schema"]["properties"] = {}

    properties = schema["input_schema"]["properties"]
    for param_name, description in param_descriptions.items():
        if param_name in properties:
            if not isinstance(properties[param_name], dict):
                properties[param_name] = {}
            properties[param_name]["description"] = description

    return schema


def _apply_metadata_overrides(schema: dict, meta) -> dict:
    """Apply metadata overrides (name, description) to schema."""
    if meta.name:
        schema["name"] = meta.name
    if meta.description is not None:
        schema["description"] = meta.description
    return schema


def _process_raw_schema(meta) -> dict:
    """Process a raw schema from metadata with overrides applied."""
    schema = meta.raw_schema.copy()
    schema = _apply_metadata_overrides(schema, meta)
    if meta.param_descriptions:
        schema = _apply_parameter_descriptions(schema, meta.param_descriptions)
    return schema


def _apply_schema_modifications(schema: dict, meta, config: dict) -> dict:
    """Apply schema modifier function if present."""
    if config and meta.schema_modifier:
        return meta.schema_modifier(schema, config)
    return schema


def create_schema_from_callable(handler: Callable, config: dict | None = None) -> dict[str, Any]:
    """Create a JSON schema for a prepared callable."""
    meta = get_tool_meta(handler)
    config = config or {}

    if meta.raw_schema is not None:
        schema = _process_raw_schema(meta)
    else:
        original = getattr(handler, "__wrapped__", handler)
        schema = function_to_tool_schema(original)

    schema = _apply_schema_modifications(schema, meta, config)
    return schema
