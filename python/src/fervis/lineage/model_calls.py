"""Model-call lineage write projection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
)
from fervis.lineage.ids import lineage_id
from fervis.lineage.recorder import (
    ModelCallAuditWrite,
    ModelCallUsageWrite,
    ModelCallWrite,
    RunArtifactWrite,
    RunStepWrite,
)
from fervis.observability.usage_types import CostSource, UsageKey


@dataclass(frozen=True)
class ModelCallCapture:
    provider: str
    model_key: str
    status: ModelCallStatus
    duration_ms: int
    system_prompt: str
    prompt_text: str
    provider_schema: dict[str, Any]
    tool_specs: tuple[Any, ...]
    usage: dict[str, Any]
    submitted_payload: dict[str, Any]
    raw_output: str = ""
    parsed_payload: dict[str, Any] | None = None
    provider_request_id: str = ""


def model_call_audit_write(
    *,
    run_id: str,
    step: RunStepWrite,
    capture: ModelCallCapture,
) -> ModelCallAuditWrite:
    model_call_id = lineage_id("model_call", step.step_id, 1)
    return ModelCallAuditWrite(
        model_call=_model_call_write(
            capture,
            run_id=run_id,
            step_id=step.step_id,
            model_call_id=model_call_id,
        ),
        usage_rows=_usage_writes(
            capture,
            run_id=run_id,
            model_call_id=model_call_id,
        ),
        artifacts=_artifact_writes(
            capture,
            run_id=run_id,
            step_id=step.step_id,
            model_call_id=model_call_id,
        ),
    )


def _model_call_write(
    capture: ModelCallCapture,
    *,
    run_id: str,
    step_id: str,
    model_call_id: str,
) -> ModelCallWrite:
    return ModelCallWrite(
        model_call_id=model_call_id,
        run_id=run_id,
        step_id=step_id,
        call_index=1,
        provider=capture.provider,
        model_key=capture.model_key,
        provider_request_id=capture.provider_request_id,
        status=capture.status,
        duration_ms=capture.duration_ms,
        prompt_chars=len(capture.prompt_text),
        schema_chars=len(_json_text(capture.provider_schema)),
        tool_spec_chars=len(_tool_specs_text(capture.tool_specs)),
        submitted_payload_chars=len(_json_text(capture.submitted_payload)),
        raw_output_chars=len(capture.raw_output) if capture.raw_output else None,
        model_subcalls_json=_model_subcalls(capture.usage),
    )


def _usage_writes(
    capture: ModelCallCapture,
    *,
    run_id: str,
    model_call_id: str,
) -> tuple[ModelCallUsageWrite, ...]:
    input_cost, output_cost, thinking_cost, cost_source, pricing_version = (
        _cost_breakdown(capture.usage)
    )
    price_basis = {
        key: value
        for key, value in {
            UsageKey.COST_SOURCE: cost_source,
            UsageKey.PRICING_VERSION: pricing_version,
        }.items()
        if value
    }
    rows: list[ModelCallUsageWrite] = []
    for usage_key, usage_kind, cost_usd in (
        (UsageKey.INPUT_TOKENS, ModelUsageKind.INPUT_TOKENS, input_cost),
        (UsageKey.OUTPUT_TOKENS, ModelUsageKind.OUTPUT_TOKENS, output_cost),
        (UsageKey.THINKING_TOKENS, ModelUsageKind.THINKING_TOKENS, thinking_cost),
    ):
        quantity = _int(capture.usage.get(usage_key), usage_key)
        if quantity == 0:
            continue
        cost_micros = None
        currency = ""
        if cost_source != CostSource.PROVIDER_USAGE_UNPRICED:
            cost_micros = _decimal_micros(cost_usd)
            currency = "USD"
        rows.append(
            ModelCallUsageWrite(
                usage_id=lineage_id("model_usage", model_call_id, usage_key),
                run_id=run_id,
                model_call_id=model_call_id,
                usage_kind=usage_kind,
                quantity=quantity,
                unit=ModelUsageUnit.TOKENS,
                provider_usage_key=usage_key,
                cost_micros=cost_micros,
                currency=currency,
                price_basis_json=dict(price_basis),
            )
        )
    return tuple(rows)


def _artifact_writes(
    capture: ModelCallCapture,
    *,
    run_id: str,
    step_id: str,
    model_call_id: str,
) -> tuple[RunArtifactWrite, ...]:
    artifacts = [
        _artifact(
            ArtifactKind.SYSTEM_PROMPT,
            capture.system_prompt,
            content_type="text/plain",
        ),
        _artifact(ArtifactKind.PROMPT, capture.prompt_text, content_type="text/plain"),
        _artifact(
            ArtifactKind.SCHEMA,
            _json_text(capture.provider_schema),
            content_type="application/json",
        ),
        _artifact(
            ArtifactKind.TOOL_SPEC,
            _tool_specs_text(capture.tool_specs),
            content_type="application/json",
        ),
        _artifact(
            ArtifactKind.SUBMITTED_PAYLOAD,
            _json_text(capture.submitted_payload),
            content_type="application/json",
        ),
    ]
    if capture.raw_output:
        artifacts.append(
            _artifact(
                ArtifactKind.RAW_OUTPUT,
                capture.raw_output,
                content_type="text/plain",
            )
        )
    if capture.parsed_payload is not None:
        artifacts.append(
            _artifact(
                ArtifactKind.PARSED_PAYLOAD,
                _json_text(capture.parsed_payload),
                content_type="application/json",
            )
        )
    return tuple(
        RunArtifactWrite(
            artifact_id=lineage_id("artifact", model_call_id, artifact_kind.value),
            run_id=run_id,
            step_id=step_id,
            model_call_id=model_call_id,
            artifact_kind=artifact_kind,
            content_hash=_content_hash(content),
            content=content,
            content_type=content_type,
            size_bytes=len(content.encode("utf-8")),
        )
        for artifact_kind, content, content_type in artifacts
    )


def _artifact(
    artifact_kind: ArtifactKind,
    content: str,
    *,
    content_type: str,
) -> tuple[ArtifactKind, str, str]:
    return artifact_kind, content, content_type


def _cost_breakdown(
    usage: dict[str, Any],
) -> tuple[Decimal, Decimal, Decimal, str, str]:
    input_cost = _optional_decimal(usage.get(UsageKey.INPUT_COST_USD))
    output_cost = _optional_decimal(usage.get(UsageKey.OUTPUT_COST_USD))
    thinking_cost = _optional_decimal(usage.get(UsageKey.THINKING_COST_USD))
    if input_cost is not None and output_cost is not None and thinking_cost is not None:
        total_cost = _decimal(usage.get(UsageKey.COST_USD))
        if input_cost + output_cost + thinking_cost != total_cost:
            raise ValueError("model-call usage cost breakdown total mismatch")
        return (
            input_cost,
            output_cost,
            thinking_cost,
            str(usage.get(UsageKey.COST_SOURCE) or "usage_breakdown"),
            str(usage.get(UsageKey.PRICING_VERSION) or ""),
        )
    total_cost = _decimal(usage.get(UsageKey.COST_USD))
    if total_cost != 0:
        raise ValueError(
            "model-call usage requires inputCostUsd, outputCostUsd, "
            "and thinkingCostUsd when costUsd is non-zero"
        )
    return (
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        str(usage.get(UsageKey.COST_SOURCE) or CostSource.PROVIDER_USAGE_UNPRICED),
        str(usage.get(UsageKey.PRICING_VERSION) or ""),
    )


def _model_subcalls(usage: dict[str, Any]) -> tuple[dict[str, object], ...]:
    raw_items = usage.get(UsageKey.MODEL_SUBCALLS)
    if not isinstance(raw_items, list):
        return ()
    return tuple(dict(item) for item in raw_items if isinstance(item, dict))


def _tool_specs_text(tool_specs: tuple[Any, ...]) -> str:
    return _json_text(tuple(_tool_spec_payload(spec) for spec in tool_specs))


def _tool_spec_payload(spec: Any) -> object:
    if not hasattr(spec, "__dict__"):
        return str(spec)
    payload = dict(spec.__dict__)
    payload.pop("transport_context", None)
    return payload


def _json_text(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _content_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _int(value: Any, key: str) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise ValueError(
            f"model-call usage has invalid integer value for {key}: {value}"
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value)
    else:
        raise ValueError(
            f"model-call usage has invalid integer value for {key}: {value}"
        )
    if parsed < 0:
        raise ValueError(
            f"model-call usage has invalid integer value for {key}: {value}"
        )
    return parsed


def _decimal(value: Any) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    return Decimal(str(value))


def _optional_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _decimal_micros(value: Decimal) -> int:
    return int(value * Decimal("1000000"))
