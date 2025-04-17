"""Tests for selective MCP server initialization."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc import LLMProgram
from llmproc.tools.mcp import MCPTool
