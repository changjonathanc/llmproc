from __future__ import annotations

from pydantic import BaseModel, field_validator

from llmproc.common.access_control import AccessLevel


class ToolConfig(BaseModel):
    """Configuration item for a tool."""

    name: str
    alias: str | None = None
    description: str | None = None
    access: AccessLevel = AccessLevel.WRITE
    param_descriptions: dict[str, str] | None = None

    def __init__(
        self,
        name: str | None = None,
        access: AccessLevel | str | None = None,
        alias: str | None = None,
        param_descriptions: dict[str, str] | None = None,
        **kwargs,
    ):
        if name is not None and "name" not in kwargs:
            kwargs["name"] = name
        if access is not None and "access" not in kwargs:
            kwargs["access"] = access
        if alias is not None and "alias" not in kwargs:
            kwargs["alias"] = alias
        if param_descriptions is not None and "param_descriptions" not in kwargs:
            kwargs["param_descriptions"] = param_descriptions
        super().__init__(**kwargs)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("Tool name cannot be empty")
        return v

    def __repr__(self) -> str:
        """Return a concise representation for debugging."""
        alias_str = f", alias={self.alias}" if self.alias else ""
        desc_str = f", description='{self.description}'" if self.description else ""
        param_desc_str = f", param_descriptions={self.param_descriptions}" if self.param_descriptions else ""
        access_str = f", access={self.access.value}" if self.access != AccessLevel.WRITE else ""
        return f"<ToolConfig {self.name}{access_str}{alias_str}{desc_str}{param_desc_str}>"
