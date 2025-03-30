"""LLMProc - A simple framework for LLM-powered applications."""

from llmproc.llm_process import LLMProcess
from llmproc.program import (
    LLMProgram,  # Need to import LLMProgram first to avoid circular import
)

__all__ = ["LLMProcess", "LLMProgram"]
__version__ = "0.3.0"
