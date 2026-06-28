"""Model-turn prompt sizing and fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MODEL_TURN_PROMPT_BUDGET_CHARS = 400_000


class ModelTurnPromptBudgetError(ValueError):
    pass


def prompt_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def serialized_tool_specs(tool_specs: tuple[Any, ...]) -> str:
    return json.dumps(
        [_provider_tool_spec_payload(spec) for spec in tool_specs],
        default=str,
        sort_keys=True,
    )


def _provider_tool_spec_payload(spec: Any) -> object:
    if not hasattr(spec, "__dict__"):
        return str(spec)
    payload = dict(spec.__dict__)
    payload.pop("transport_context", None)
    return payload


def model_turn_event_telemetry(
    *,
    system_prompt: str,
    prompt: str,
    tool_specs: tuple[Any, ...],
    usage: dict[str, Any] | None,
    prompt_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from fervis.observability.event_contracts import EventPayloadKey

    tool_spec_text = serialized_tool_specs(tool_specs)
    return {
        EventPayloadKey.USAGE: dict(usage or {}),
        EventPayloadKey.PROMPT_CHARS: len(prompt),
        EventPayloadKey.PROMPT_HASH: prompt_fingerprint(prompt),
        EventPayloadKey.TOOL_SPEC_CHARS: len(tool_spec_text),
        EventPayloadKey.TOOL_SPEC_HASH: prompt_fingerprint(tool_spec_text),
        EventPayloadKey.PROMPT_FRAME: dict(prompt_metadata or {}),
        EventPayloadKey.RAW_SYSTEM_PROMPT: system_prompt,
        EventPayloadKey.RAW_PROMPT: prompt,
        EventPayloadKey.RAW_TOOL_SPECS: json.loads(tool_spec_text),
    }


def enforce_model_turn_prompt_budget(
    *,
    prompt: str,
    tool_specs: tuple[Any, ...] = (),
    schema: Any = None,
) -> None:
    schema_text = (
        "" if schema is None else json.dumps(schema, default=str, sort_keys=True)
    )
    total_chars = (
        len(prompt) + len(serialized_tool_specs(tool_specs)) + len(schema_text)
    )
    if total_chars > MODEL_TURN_PROMPT_BUDGET_CHARS:
        raise ModelTurnPromptBudgetError(
            "model turn prompt budget exceeded: "
            f"{total_chars}>{MODEL_TURN_PROMPT_BUDGET_CHARS}"
        )
