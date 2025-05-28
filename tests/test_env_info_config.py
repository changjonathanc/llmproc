import pytest
from llmproc.config.schema import LLMProgramConfig, ModelConfig, EnvInfoConfig


def test_env_info_config_defaults():
    """EnvInfoConfig defaults are applied."""
    cfg = LLMProgramConfig(
        model=ModelConfig(name="m", provider="openai"),
        env_info=EnvInfoConfig(variables=["file_map"]),
    )

    assert cfg.env_info.file_map_root is None
    assert cfg.env_info.file_map_max_files == 50
    assert cfg.env_info.file_map_show_size is True


def test_env_info_config_validation():
    """file_map_max_files must be positive."""
    with pytest.raises(ValueError):
        EnvInfoConfig(variables=["file_map"], file_map_max_files=0)
