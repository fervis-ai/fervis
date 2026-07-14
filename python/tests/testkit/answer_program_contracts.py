from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program import (
    BindingPatch,
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    CapabilityApplication,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRole,
    ParameterValueType,
    SetParameter,
    UnsetParameter,
    canonical_fact_value,
)
from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.canonical_data import entity_key_value


def parameter_declarations_from_payload(
    payload: dict[str, Any],
) -> tuple[ParameterDeclaration, ...]:
    return tuple(
        ParameterDeclaration(
            id=str(item["id"]),
            role=ParameterRole(str(item["role"])),
            value_type=ParameterValueType(str(item["value_type"])),
            required=bool(item.get("required", True)),
            allowed_values=tuple(
                str(value) for value in item.get("allowed_values") or ()
            ),
            semantic_control_ref=str(item.get("semantic_control_ref") or ""),
        )
        for item in payload.get("parameters") or ()
    )


def binding_set_from_payload(payload: dict[str, Any]) -> BindingSet:
    return BindingSet.from_bindings(
        tuple(
            ParameterBinding(
                parameter_id=str(item["parameter_id"]),
                value=fact_value_from_payload(
                    item["value"],
                    value_id=f"binding:{item['parameter_id']}",
                ),
                provenance=_provenance(item.get("provenance") or {}),
            )
            for item in payload.get("bindings") or ()
        )
    )


def binding_patch_from_payload(payload: dict[str, Any]) -> BindingPatch:
    operations = []
    for item in payload.get("operations") or ():
        if item["kind"] == "set":
            operations.append(
                SetParameter(
                    parameter_id=str(item["parameter_id"]),
                    value=fact_value_from_payload(
                        item["value"],
                        value_id=f"patch:{item['parameter_id']}",
                    ),
                )
            )
        elif item["kind"] == "unset":
            operations.append(UnsetParameter(parameter_id=str(item["parameter_id"])))
        else:
            raise ValueError("unsupported test binding patch operation")
    return BindingPatch(operations=tuple(operations))


def capability_application_from_payload(
    payload: dict[str, Any],
) -> CapabilityApplication:
    binding = payload["binding"]
    return CapabilityApplication(
        capability_id=str(payload["capability_id"]),
        binding=ParameterBinding(
            parameter_id=str(binding["parameter_id"]),
            value=fact_value_from_payload(
                binding["value"],
                value_id=f"capability:{binding['parameter_id']}",
            ),
            provenance=_provenance(binding.get("provenance") or {}),
        ),
    )


def binding_payload(bindings: BindingSet) -> dict[str, Any]:
    return {
        binding.parameter_id: canonical_fact_value(binding.value)
        for binding in bindings.bindings
    }


def _provenance(payload: dict[str, Any]) -> BindingProvenance:
    return BindingProvenance(
        kind=BindingProvenanceKind(str(payload.get("kind") or "plan_choice")),
        refs=tuple(str(ref) for ref in payload.get("refs") or ()),
    )


def fact_value_from_payload(
    payload: dict[str, Any],
    *,
    value_id: str,
) -> FactValue:
    kind = str(payload["kind"])
    common = {
        "id": value_id,
        "proof_refs": tuple(str(item) for item in payload.get("proof_refs") or ()),
        "source_refs": tuple(str(item) for item in payload.get("source_refs") or ()),
        "known_input_id": str(payload.get("known_input_id") or ""),
        "applies_to_requested_fact_ids": tuple(
            str(item) for item in payload.get("applies_to_requested_fact_ids") or ()
        ),
    }
    if kind == "identity":
        return FactValue.identity(
            **common,
            key=entity_key_value(
                str(payload["entity_kind"]),
                str(payload["key_id"]),
                {str(payload["key_component_id"]): str(payload["value"])},
            ),
            display_value=str(payload.get("display_value") or ""),
        )
    if kind == "identity_set":
        return FactValue.identity_set(
            **common,
            keys=tuple(
                entity_key_value(
                    str(payload["entity_kind"]),
                    str(payload["key_id"]),
                    {str(payload["key_component_id"]): str(value)},
                )
                for value in payload.get("values") or ()
            ),
            display_value=str(payload.get("display_value") or ""),
        )
    if kind == "named":
        return FactValue.named(
            **common,
            text=str(payload["text"]),
            reference_text=str(payload.get("reference_text") or ""),
        )
    if kind == "time":
        return FactValue.time(
            **common,
            expression=str(payload["expression"]),
            resolved_start=str(payload.get("resolved_start") or ""),
            resolved_end=str(payload.get("resolved_end") or ""),
            granularity=str(payload.get("granularity") or ""),
        )
    if kind == "string_set":
        return FactValue.string_set(
            **common,
            values=tuple(str(value) for value in payload.get("values") or ()),
        )
    literal_type = LiteralType(kind)
    value = payload.get("value")
    return FactValue.literal(
        **common,
        literal_type=literal_type,
        value=str(value).lower() if isinstance(value, bool) else str(value),
    )
