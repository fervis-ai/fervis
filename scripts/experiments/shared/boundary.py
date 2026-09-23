"""Serialize one invocation produced by a production turn prompt."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fervis.model_io.turn_artifacts import ModelTurnArtifact


def captured_boundary(
    artifact: ModelTurnArtifact,
    *,
    purpose: str,
    provider: str,
    model_key: str,
    assertion_context: dict[str, Any],
) -> dict[str, object]:
    return {
        "source_run_id": f"{purpose}-isolated-production-invocation",
        "sequence": 1,
        "purpose": purpose,
        "provider": provider,
        "model_key": model_key,
        "system_prompt": artifact.system_prompt,
        "prompt": artifact.prompt_text,
        "tool_specs": [asdict(item) for item in artifact.tool_specs],
        "assertion_context": assertion_context,
    }


__all__ = ["captured_boundary"]
