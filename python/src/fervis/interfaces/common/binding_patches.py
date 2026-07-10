"""Strict HTTP projection into typed answer-program binding patches."""

from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program import (
    BindingPatch,
    BindingProvenance,
    BindingProvenanceKind,
    CapabilityApplication,
    ParameterBinding,
    SetParameter,
    UnsetParameter,
)
from fervis.lookup.answer_program.values import FactValue, LiteralType


def binding_patch_from_payload(payload: object) -> BindingPatch:
    values = _exact_object(payload, {"operations"}, "patch")
    operations = values["operations"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("patch.operations must be a non-empty array")
    return BindingPatch(
        operations=tuple(_operation(item) for item in operations),
    )


def capability_application_from_payload(payload: object) -> CapabilityApplication:
    values = _exact_object(
        payload,
        {"capabilityId", "binding"},
        "capability application",
    )
    capability_id = _required_string(values["capabilityId"], "capabilityId")
    binding = _exact_object(
        values["binding"],
        {"parameterId", "value"},
        "capability binding",
    )
    parameter_id = _required_string(binding["parameterId"], "parameterId")
    return CapabilityApplication(
        capability_id=capability_id,
        binding=ParameterBinding(
            parameter_id=parameter_id,
            value=_fact_value(
                binding["value"],
                value_id=f"capability:{parameter_id}",
            ),
            provenance=BindingProvenance(
                kind=BindingProvenanceKind.SEMANTIC_CHOICE,
                refs=(f"capability:{capability_id}",),
            ),
        ),
    )


def _operation(payload: object):
    if not isinstance(payload, dict):
        raise ValueError("patch operation must be an object")
    kind = _required_string(payload.get("kind"), "patch operation kind")
    if kind == "unset":
        values = _exact_object(payload, {"kind", "parameterId"}, "unset operation")
        return UnsetParameter(
            parameter_id=_required_string(values["parameterId"], "parameterId")
        )
    if kind == "set":
        values = _exact_object(
            payload,
            {"kind", "parameterId", "value"},
            "set operation",
        )
        parameter_id = _required_string(values["parameterId"], "parameterId")
        return SetParameter(
            parameter_id=parameter_id,
            value=_fact_value(values["value"], value_id=f"patch:{parameter_id}"),
        )
    raise ValueError(f"unsupported patch operation kind: {kind}")


def _fact_value(payload: object, *, value_id: str) -> FactValue:
    if not isinstance(payload, dict):
        raise ValueError("binding value must be an object")
    kind = _required_string(payload.get("kind"), "patch value kind")
    if kind in {"identity", "identity_set"}:
        raise ValueError("identity patch values require current-authority grounding")
    if kind == "named":
        values = _exact_object(
            payload,
            {"kind", "text", "referenceText"},
            "named value",
            optional={"referenceText"},
        )
        return FactValue.named(
            id=value_id,
            text=_required_string(values["text"], "text"),
            reference_text=_optional_string(
                values.get("referenceText"), "referenceText"
            ),
        )
    if kind == "time":
        values = _exact_object(
            payload,
            {
                "kind",
                "expression",
                "intent",
                "resolvedStart",
                "resolvedEnd",
                "granularity",
            },
            "time value",
            optional={"intent"},
        )
        intent = values.get("intent", {})
        if not isinstance(intent, dict):
            raise ValueError("intent must be an object")
        return FactValue.time(
            id=value_id,
            expression=_required_string(values["expression"], "expression"),
            intent=dict(intent),
            resolved_start=_required_string(values["resolvedStart"], "resolvedStart"),
            resolved_end=_required_string(values["resolvedEnd"], "resolvedEnd"),
            granularity=_required_string(values["granularity"], "granularity"),
        )
    if kind == "string_set":
        values = _exact_object(payload, {"kind", "values"}, "string-set value")
        return FactValue.string_set(
            id=value_id,
            values=_string_tuple(values["values"], "values"),
        )
    if kind == "number":
        values = _exact_object(payload, {"kind", "value"}, "number value")
        value = values["value"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("number value must be numeric")
        return FactValue.literal(
            id=value_id,
            literal_type=LiteralType.NUMBER,
            value=str(value),
        )
    if kind == "string":
        values = _exact_object(payload, {"kind", "value"}, "string value")
        return FactValue.literal(
            id=value_id,
            literal_type=LiteralType.STRING,
            value=_required_string(values["value"], "value"),
        )
    if kind == "boolean":
        values = _exact_object(payload, {"kind", "value"}, "boolean value")
        value = values["value"]
        if not isinstance(value, bool):
            raise ValueError("boolean value must be boolean")
        return FactValue.literal(
            id=value_id,
            literal_type=LiteralType.BOOLEAN,
            value=str(value).lower(),
        )
    raise ValueError(f"unsupported patch value kind: {kind}")


def _exact_object(
    payload: object,
    fields: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    optional_fields = optional or set()
    required = fields - optional_fields
    actual = set(payload)
    if not required.issubset(actual) or not actual.issubset(fields):
        raise ValueError(
            f"{label} fields do not match contract: "
            f"missing={sorted(required - actual)}, unknown={sorted(actual - fields)}"
        )
    return dict(payload)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: object, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a non-empty string array")
    return tuple(value)
