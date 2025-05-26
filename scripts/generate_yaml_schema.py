#!/usr/bin/env python
"""Generate YAML schema for LLMProgram configuration."""

import yaml

from llmproc.config.schema import LLMProgramConfig


def main() -> None:
    """Print the YAML schema to stdout."""
    schema = LLMProgramConfig.model_json_schema()
    print(yaml.dump(schema, sort_keys=False))


if __name__ == "__main__":
    main()
