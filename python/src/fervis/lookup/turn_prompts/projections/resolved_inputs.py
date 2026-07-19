"""Fact-scoped resolved-input projections shared by planning turns."""

from __future__ import annotations

from typing import TypeAlias

from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralValuePayload,
    NamedValuePayload,
    TimeValuePayload,
    ValueFilterOperator,
)
from fervis.lookup.question_contract import RequestedFact


ResolvedInputField: TypeAlias = str | int | list[str]
ResolvedInputPayload: TypeAlias = dict[str, ResolvedInputField]


def resolved_inputs_for_requested_fact(
    requested_fact: RequestedFact,
    *,
    available_values: tuple[FactValue, ...],
) -> tuple[ResolvedInputPayload, ...]:
    """Project the fixed values of one fact's known inputs."""

    resolved_values = resolved_values_for_requested_fact(
        requested_fact,
        available_values=available_values,
    )
    values_by_known_input_id: dict[str, list[FactValue]] = {}
    for value in resolved_values:
        values_by_known_input_id.setdefault(value.known_input_id, []).append(value)

    resolved_inputs: list[ResolvedInputPayload] = []
    for known_input in requested_fact.known_inputs:
        for value in values_by_known_input_id.get(known_input.id, ()):
            resolved_inputs.append(
                _resolved_input_payload(
                    value,
                    known_input_id=known_input.id,
                    source_text=known_input.text,
                )
            )
    return tuple(resolved_inputs)


def resolved_values_for_requested_fact(
    requested_fact: RequestedFact,
    *,
    available_values: tuple[FactValue, ...],
) -> tuple[FactValue, ...]:
    """Return the typed values owned by one requested fact's known inputs."""

    return tuple(
        value
        for value in available_values
        if _value_applies_to_fact(value, requested_fact=requested_fact)
    )


def _value_applies_to_fact(
    value: FactValue,
    *,
    requested_fact: RequestedFact,
) -> bool:
    if not value.known_input_id:
        return False
    fact_input_ids = {known_input.id for known_input in requested_fact.known_inputs}
    if value.known_input_id not in fact_input_ids:
        return False
    fact_scope = value.applies_to_requested_fact_ids
    return not fact_scope or requested_fact.id in fact_scope


def _resolved_input_payload(
    value: FactValue,
    *,
    known_input_id: str,
    source_text: str,
) -> ResolvedInputPayload:
    output: ResolvedInputPayload = {
        "known_input_id": known_input_id,
        "source_text": source_text,
        "value_id": value.id,
        "kind": value.kind.value,
    }
    output.update(_resolved_value_detail_payload(value))
    if isinstance(
        value.payload,
        (IdentityValuePayload, IdentitySetValuePayload),
    ) and not output.get("display_value"):
        output["display_value"] = source_text
    return output


def fact_value_prompt_payload(value: FactValue) -> ResolvedInputPayload:
    """Project one typed fact value without fact- or turn-specific context."""

    output: ResolvedInputPayload = {
        "value_id": value.id,
        "kind": value.kind.value,
    }
    output.update(_resolved_value_detail_payload(value))
    return output


def _resolved_value_detail_payload(value: FactValue) -> ResolvedInputPayload:
    output: ResolvedInputPayload = {}
    payload = value.payload
    if isinstance(payload, IdentityValuePayload):
        output.update(
            {
                "entity_kind": payload.entity_kind,
                "key_id": payload.key_id,
                "key_components": [
                    component.component_id for component in payload.key.components
                ],
                "display_value": payload.display_value or value.label,
            }
        )
        return output
    if isinstance(payload, IdentitySetValuePayload):
        output.update(
            {
                "entity_kind": payload.entity_kind,
                "key_id": payload.key_id,
                "key_components": [
                    component.component_id for component in payload.keys[0].components
                ],
                "count": len(payload.keys),
                "display_value": payload.display_value or value.label,
            }
        )
        return output
    if isinstance(payload, TimeValuePayload):
        output.update(
            {
                "resolved_start": payload.resolved_start,
                "resolved_end": payload.resolved_end,
            }
        )
        return output
    if isinstance(payload, LiteralValuePayload):
        output.update(
            {
                "literal_type": payload.literal_type.value,
                "value": payload.value,
            }
        )
        return output
    if isinstance(payload, NamedValuePayload):
        output["text"] = payload.text
        if payload.filter_operator is not ValueFilterOperator.EQUALS:
            output["operator"] = payload.filter_operator.value
        return output
    return output
