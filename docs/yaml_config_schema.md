# YAML Configuration Schema

This document contains an auto-generated YAML schema showing all configuration options
supported by `LLMProgram`. The schema is produced directly from the Pydantic models
used for configuration validation.

To regenerate the schema, run the following command from the repository root:

```bash
make yaml-schema
```

The generated file `yaml_config_schema.yaml` can be used as a reference for
building your own YAML configuration files. All fields shown are supported by
`LLMProgram.from_yaml` and have the same structure when using TOML.
