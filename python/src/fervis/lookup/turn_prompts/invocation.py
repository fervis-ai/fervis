"""Provider-facing invocation boundary for Lookup model turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fervis.model_io.backbone.dto import ToolSpec, ProviderOutputMode
from fervis.lookup.turn_prompts.sections import ModelPromptPayload


@dataclass(frozen=True)
class ProviderResponseContract:
    provider_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderToolContract:
    tool_specs: tuple[ToolSpec, ...]


@dataclass(frozen=True)
class ProviderInvocationMetadata:
    values: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelTurnInvocation:
    turn_name: str
    prompt: ModelPromptPayload
    response_contract: ProviderResponseContract
    tool_contract: ProviderToolContract
    metadata: ProviderInvocationMetadata = ProviderInvocationMetadata()

    @property
    def system_prompt(self) -> str:
        return self.prompt.system_prompt

    @property
    def prompt_text(self) -> str:
        return self.prompt.prompt_text

    @property
    def provider_schema(self) -> dict[str, Any]:
        return dict(self.response_contract.provider_schema)

    @property
    def tool_specs(self) -> tuple[ToolSpec, ...]:
        return self.tool_contract.tool_specs

    def to_provider_payload(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "prompt": self.prompt_text,
            "output_mode": ProviderOutputMode.TOOL_CALL,
            "tool_specs": self.tool_specs,
            **dict(self.metadata.values),
        }
