# Plugin Organization

LLMProc separates configurable plugins from manual-only extensions. Components that
can be enabled through YAML files live in `llmproc.plugins`. Any utilities or
plugins that must be added explicitly in code are placed in `llmproc.extensions`.

## YAML-Enabled Plugins

Plugins registered in `llmproc.plugins` are referenced through the `plugins`
section of a YAML program file. The `schema.yaml` documentation describes their
available options.

```yaml
plugins:
  env_info:
    variables: [platform]
```

## Manual Extensions

Modules in `llmproc.extensions` are not part of the YAML schema. Import these
manually and add them with `add_plugins()`:

```python
from llmproc.program import LLMProgram
from llmproc.extensions.clipboard import ClipboardPlugin

program = LLMProgram(model_name="claude-3-haiku-20240307", provider="anthropic")
program.add_plugins(ClipboardPlugin())
```

The same approach applies to the example plugins in
`llmproc.extensions.examples`.

---
[← Back to Documentation Index](index.md)
