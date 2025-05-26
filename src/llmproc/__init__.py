"""LLMProc - A simple framework for LLM-powered applications."""

from importlib.metadata import version

from llmproc.llm_process import AsyncLLMProcess, LLMProcess, SyncLLMProcess
from llmproc.program import (
    LLMProgram,  # Need to import LLMProgram first to avoid circular import
)
from llmproc.tools import register_tool

__all__ = ["LLMProcess", "AsyncLLMProcess", "SyncLLMProcess", "LLMProgram", "register_tool"]
__version__ = version("llmproc")
