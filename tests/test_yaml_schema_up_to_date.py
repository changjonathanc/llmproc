"""Ensure docs/yaml_config_schema.yaml matches the generated Pydantic schema."""

from pathlib import Path

import yaml

from llmproc.config.schema import LLMProgramConfig


def test_yaml_schema_up_to_date() -> None:
    """Fail if the checked-in YAML schema is out of sync."""
    generated_schema = LLMProgramConfig.model_json_schema()
    schema_path = Path(__file__).parent.parent / "docs" / "yaml_config_schema.yaml"
    with schema_path.open() as f:
        current_schema = yaml.safe_load(f)
    assert current_schema == generated_schema
