"""Lookup model-turn event payload helpers."""

import json
from typing import Any

from fervis.model_io.telemetry import (
    model_turn_event_telemetry,
    prompt_fingerprint,
)
from fervis.observability.event_contracts import EventPayloadKey, ToolResultKey
from fervis.lookup.orchestration.request import LookupRequest


def _model_turn_event_payload(
    *,
    request: LookupRequest,
    phase: str,
    turn: int,
    provider: str,
    model_key: str,
    system_prompt: str,
    prompt_text: str,
    usage: dict[str, Any],
    duration_ms: int,
    tool_specs: tuple[Any, ...],
    schema: dict[str, Any],
    submitted_payload: dict[str, Any] | None = None,
    raw_output: str = "",
    parsed_payload: dict[str, Any] | None = None,
    derived_payload: dict[str, Any] | None = None,
    selected_tool_name: str = "",
    error_code: str = "",
    error_class: str = "",
    error_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema_text = json.dumps(schema, default=str, sort_keys=True)
    payload = {
        EventPayloadKey.RUN_ID: request.run_id,
        EventPayloadKey.TURN: turn,
        EventPayloadKey.PURPOSE: phase,
        EventPayloadKey.PROVIDER: provider,
        EventPayloadKey.MODEL_KEY: model_key,
        ToolResultKey.DURATION_MS: duration_ms,
        EventPayloadKey.SCHEMA_CHARS: len(schema_text),
        EventPayloadKey.SCHEMA_HASH: prompt_fingerprint(schema_text),
        EventPayloadKey.RAW_SCHEMA: schema,
        **model_turn_event_telemetry(
            system_prompt=system_prompt,
            prompt=prompt_text,
            tool_specs=tool_specs,
            usage=usage,
            prompt_metadata=_prompt_metadata(request, phase=phase),
        ),
    }
    if error_code:
        payload[EventPayloadKey.ERROR_CODE] = error_code
    if error_class:
        payload[EventPayloadKey.ERROR_CLASS] = error_class
    if error_context:
        payload[EventPayloadKey.ERROR_CONTEXT] = dict(error_context)
    if submitted_payload is not None:
        payload[EventPayloadKey.ARGUMENTS] = dict(submitted_payload)
    if raw_output:
        payload[EventPayloadKey.MODEL_OUTPUT] = raw_output
    if parsed_payload is not None:
        payload[EventPayloadKey.PARSED_ARGUMENTS] = dict(parsed_payload)
    if derived_payload is not None:
        payload[EventPayloadKey.DERIVED_ARGUMENTS] = dict(derived_payload)
    if selected_tool_name:
        payload[EventPayloadKey.SELECTED_TOOL_NAME] = selected_tool_name
    return payload


_PROMPT_METADATA_CONTEXT_KEYS = (
    "conversationId",
    "caseId",
    "goldsetRunId",
    "certificationRunId",
)


def _prompt_metadata(request: LookupRequest, *, phase: str) -> dict[str, str]:
    metadata = {"mode": "lookup", "phase": phase}
    for key in _PROMPT_METADATA_CONTEXT_KEYS:
        value = request.user_context.get(key)
        if value not in (None, ""):
            metadata[key] = str(value)
    return metadata
