"""Provider-specific structured-output projection for model-turn budget checks."""

from __future__ import annotations

from typing import Any

from fervis.model_io.backbone.dto import ToolSpec
from fervis.model_io.backbone.registry import registrations


def provider_budget_tool_specs(
    *,
    provider: str,
    tool_specs: tuple[ToolSpec, ...],
) -> tuple[Any, ...]:
    provider_name = str(provider or "").strip()
    if not provider_name:
        return tool_specs
    registration = registrations().get(provider_name)
    if registration is None:
        return tool_specs
    return registration.budget_tool_specs(tool_specs)
