"""Raw model-turn artifacts for Lookup model-boundary diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.model_io.backbone.dto import ToolSpec


@dataclass(frozen=True)
class ModelTurnArtifact:
    system_prompt: str
    prompt_text: str
    provider_schema: dict[str, Any]
    tool_specs: tuple[ToolSpec, ...]
    submitted_payload: dict[str, Any]
    raw_output: str = ""
    parsed_payload: dict[str, Any] | None = None
    derived_payload: dict[str, Any] | None = None
    selected_tool_name: str = ""
    verifier_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tool_specs:
            raise ValueError("model turn artifact requires tool specs")


def model_turn_artifact(
    *,
    system_prompt: str,
    prompt_text: str,
    provider_schema: dict[str, Any],
    tool_specs: tuple[ToolSpec, ...],
    submitted_payload: dict[str, Any],
    raw_output: str = "",
    parsed_payload: dict[str, Any] | None = None,
    derived_payload: dict[str, Any] | None = None,
    selected_tool_name: str = "",
    verifier_diagnostics: tuple[str, ...] = (),
) -> ModelTurnArtifact:
    return ModelTurnArtifact(
        system_prompt=system_prompt,
        prompt_text=prompt_text,
        provider_schema=provider_schema,
        tool_specs=tool_specs,
        submitted_payload=submitted_payload,
        raw_output=raw_output,
        parsed_payload=parsed_payload,
        derived_payload=derived_payload,
        selected_tool_name=selected_tool_name,
        verifier_diagnostics=verifier_diagnostics,
    )
