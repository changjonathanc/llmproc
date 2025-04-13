"""API integration tests for MCP tools.

These tests verify that MCP tools work correctly with real API calls.
They require valid API keys to be set in the environment.
"""

import os
import re
from pathlib import Path

import pytest

from llmproc import LLMProcess
from llmproc.config.program_loader import ProgramLoader
from llmproc.program import LLMProgram

# See test_mcp_add_tool.py for MCP tool tests using the add tool
