import asyncio
from pathlib import Path

from llmproc.program import LLMProgram
from llmproc.plugins.spawn import SpawnPlugin


def test_spawn_plugin_loaded(tmp_path: Path):
    child = tmp_path / "child.toml"
    child.write_text(
        """
[model]
name = "child"
provider = "anthropic"

[prompt]
system_prompt = "child"
"""
    )
    main = tmp_path / "main.toml"
    main.write_text(
        f"""
[model]
name = "main"
provider = "anthropic"

[prompt]
system_prompt = "main"

[plugins.spawn]
linked_programs = {{ child = "{child}" }}
"""
    )
    program = LLMProgram.from_toml(main)
    assert any(isinstance(p, SpawnPlugin) for p in program.plugins)
    spawn_plugin = next((p for p in program.plugins if isinstance(p, SpawnPlugin)), None)
    assert spawn_plugin is not None
    assert "child" in spawn_plugin.linked_programs
