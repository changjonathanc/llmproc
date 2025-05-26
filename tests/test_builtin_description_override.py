"""Test builtin tool description override using ToolConfig."""

from llmproc.program import convert_to_callables
from llmproc.config.tool import ToolConfig
from llmproc.tools.builtin import calculator
from llmproc.common.metadata import attach_meta, get_tool_meta


def test_builtin_description_override():
    """ToolConfig description should override builtin metadata."""
    funcs = convert_to_callables([ToolConfig(name="calculator", description="calc desc")])
    func = funcs[0]
    assert func is calculator
    assert get_tool_meta(func).description == "calc desc"


def test_builtin_param_description_override():
    """Overriding one param keeps other builtin descriptions."""
    original = dict(get_tool_meta(calculator).param_descriptions)
    funcs = convert_to_callables(
        [ToolConfig(name="calculator", param_descriptions={"expression": "expr"})]
    )
    func = funcs[0]
    meta = get_tool_meta(func)
    assert meta.param_descriptions["expression"] == "expr"
    assert meta.param_descriptions["precision"] == original["precision"]

    # restore builtin metadata to avoid side effects
    meta.param_descriptions = original
    attach_meta(calculator, meta)
