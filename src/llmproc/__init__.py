"""LLMProc - A simple framework for LLM-powered applications."""

from llmproc.llm_process import LLMProcess
from llmproc.program import (
    LLMProgram,  # Need to import LLMProgram first to avoid circular import
)
from llmproc.tools import register_tool

__all__ = ["LLMProcess", "LLMProgram", "register_tool"]
__version__ = "0.4.1"
