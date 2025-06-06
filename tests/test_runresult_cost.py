"""Tests for the RunResult.usd_cost property."""

import pytest

from llmproc.common.results import RunResult


def test_usd_cost_basic_calculation():
    """usd_cost should calculate cost using the pricing table."""
    result = RunResult()
    result.add_api_call(
        {
            "model": "claude-sonnet-4",
            "usage": {"input_tokens": 1000, "output_tokens": 2000},
        }
    )
    expected = (1000 * 3.0 + 2000 * 15.0) / 1_000_000
    assert result.usd_cost == pytest.approx(expected)


def test_usd_cost_ignores_unknown_models():
    """Cost should be zero for models without pricing data."""
    result = RunResult()
    result.add_api_call({"model": "unknown-model", "usage": {"input_tokens": 10}})
    assert result.usd_cost == 0.0
