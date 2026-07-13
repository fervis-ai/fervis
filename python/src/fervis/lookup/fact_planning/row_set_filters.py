"""Model-facing row-source filters filled by canonical values."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fervis.lookup.fact_planning.grounded_params import GroundedParamValue
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralValuePayload,
    NamedValuePayload,
    TimeValuePayload,
    ValueFilterOperator,
    ValueKind,
    known_input_id_for_value,
)
from fervis.lookup.fact_plan.row_sources import RowSource


def row_set_filters_payload(
    source: Any,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    values_by_id: dict[str, FactValue],
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    filters_by_value_id: dict[str, dict[str, Any]] = {}
    _append_grounded_field_filters(
        filters,
        filters_by_value_id=filters_by_value_id,
        source=source,
        grounded_params=grounded_params,
        values_by_id=values_by_id,
    )
    _append_matching_identity_field_filters(
        filters,
        filters_by_value_id=filters_by_value_id,
        source=source,
        values_by_id=values_by_id,
    )
    for param in source.params:
        grounded_param = grounded_params.get((source.id, param.id))
        if grounded_param is None:
            continue
        existing = filters_by_value_id.get(grounded_param.value_id)
        if existing is not None:
            param_ids = existing.setdefault("param_ids", [])
            if isinstance(param_ids, list) and grounded_param.param_id not in param_ids:
                param_ids.append(grounded_param.param_id)
            continue
        payload = _row_set_filter_payload(
            value=values_by_id.get(grounded_param.value_id),
        )
        if payload:
            payload["param_ids"] = [grounded_param.param_id]
            filters_by_value_id[grounded_param.value_id] = payload
            filters.append(payload)
    return filters


def row_set_filters_for_sources_payload(
    sources: tuple[Any, ...],
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    values_by_id: dict[str, FactValue],
    requested_fact_id: str = "",
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    available_values = tuple(values_by_id.values())
    for source in sources:
        for item in row_set_filters_payload(
            source,
            grounded_params=grounded_params,
            values_by_id=values_by_id,
        ):
            key = repr(sorted(item.items()))
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
    if not requested_fact_id:
        return output
    return filter_row_set_filters_for_requested_fact(
        tuple(output),
        requested_fact_id=requested_fact_id,
        available_values=available_values,
    )


def filter_row_set_filters_for_requested_fact(
    filters: tuple[dict[str, Any], ...],
    *,
    requested_fact_id: str,
    available_values: tuple[FactValue, ...],
) -> list[dict[str, Any]]:
    values_by_known_input_id = {
        known_input_id: value
        for value in available_values
        for known_input_id in (known_input_id_for_value(value),)
        if known_input_id
    }
    output: list[dict[str, Any]] = []
    for item in filters:
        known_input_id = str(item.get("known_input_id") or "")
        value = values_by_known_input_id.get(known_input_id)
        if value is not None and not value_applies_to_requested_fact(
            value,
            requested_fact_id,
        ):
            continue
        output.append(item)
    return output


def value_applies_to_requested_fact(
    value: FactValue,
    requested_fact_id: str,
) -> bool:
    if not value.applies_to_requested_fact_ids:
        return True
    return requested_fact_id in value.applies_to_requested_fact_ids


def _append_grounded_field_filters(
    filters: list[dict[str, Any]],
    *,
    filters_by_value_id: dict[str, dict[str, Any]],
    source: Any,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    values_by_id: dict[str, FactValue],
) -> None:
    for grounded in grounded_params.values():
        value = values_by_id.get(grounded.value_id)
        matching_field_ids = _grounded_filter_field_ids(
            source,
            grounded=grounded,
            value=value,
        )
        if not matching_field_ids:
            continue
        existing = filters_by_value_id.get(grounded.value_id)
        if existing is not None:
            field_ids_payload = existing.setdefault("field_ids", [])
            if isinstance(field_ids_payload, list):
                for field_id in matching_field_ids:
                    if field_id not in field_ids_payload:
                        field_ids_payload.append(field_id)
            continue
        payload = _row_set_filter_payload(value=value)
        if not payload:
            continue
        payload["field_ids"] = list(matching_field_ids)
        filters_by_value_id[grounded.value_id] = payload
        filters.append(payload)


def _grounded_filter_field_ids(
    source: RowSource,
    *,
    grounded: GroundedParamValue,
    value: FactValue | None,
) -> tuple[str, ...]:
    if grounded.row_source_id == source.id:
        return (grounded.field_id,) if grounded.field_id else ()
    if not grounded.entity_kind or not _source_represents_entity(
        source,
        entity_kind=grounded.entity_kind,
    ):
        return ()
    payload = value.payload if value is not None else None
    if not isinstance(payload, NamedValuePayload) or not payload.matched_field_ref:
        return ()
    return tuple(
        field.id
        for field in source.fields
        if field.field_ref == payload.matched_field_ref
    )


def _source_represents_entity(source: RowSource, *, entity_kind: str) -> bool:
    return any(key.entity_kind == entity_kind for key in source.candidate_keys)


def _append_matching_identity_field_filters(
    filters: list[dict[str, Any]],
    *,
    filters_by_value_id: dict[str, dict[str, Any]],
    source: RowSource,
    values_by_id: dict[str, FactValue],
) -> None:
    for value in _unique_identity_values(values_by_id.values()):
        field_ids = _identity_field_ids(source, value=value)
        if not field_ids:
            continue
        existing = filters_by_value_id.get(value.id)
        if existing is not None:
            field_ids_payload = existing.setdefault("field_ids", [])
            if isinstance(field_ids_payload, list):
                for field_id in field_ids:
                    if field_id not in field_ids_payload:
                        field_ids_payload.append(field_id)
            continue
        payload = _row_set_filter_payload(value=value)
        if not payload:
            continue
        payload["field_ids"] = list(field_ids)
        filters_by_value_id[value.id] = payload
        filters.append(payload)


def _unique_identity_values(values: Iterable[FactValue]) -> tuple[FactValue, ...]:
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[FactValue]] = {}
    for value in values:
        identity = _identity_payload(value)
        if identity is None:
            continue
        known_input_id = known_input_id_for_value(value)
        if not known_input_id:
            continue
        grouped.setdefault(
            (
                known_input_id,
                identity.entity_kind,
                identity.key_id,
                _identity_component_ids(identity),
            ),
            [],
        ).append(value)
    return tuple(items[0] for items in grouped.values() if len(items) == 1)


def _identity_field_ids(
    source: RowSource,
    *,
    value: FactValue,
) -> tuple[str, ...]:
    value_identity = _identity_payload(value)
    if value_identity is None:
        return ()
    component_ids = frozenset(_identity_component_ids(value_identity))
    key_field_ids = (
        component.field_id
        for key in source.candidate_keys
        if key.entity_kind == value_identity.entity_kind
        and key.id == value_identity.key_id
        for component in key.components
        if component.id in component_ids
    )
    reference_field_ids = (
        component.local_field_id
        for reference in source.entity_references
        if reference.target_entity_kind == value_identity.entity_kind
        and reference.target_key_id == value_identity.key_id
        for component in reference.components
        if component.target_component_id in component_ids
    )
    return tuple(dict.fromkeys((*key_field_ids, *reference_field_ids)))


def _identity_component_ids(
    identity: IdentityValuePayload | IdentitySetValuePayload,
) -> tuple[str, ...]:
    key = identity.key if isinstance(identity, IdentityValuePayload) else identity.keys[0]
    return tuple(component.component_id for component in key.components)


def _identity_payload(
    value: FactValue,
) -> IdentityValuePayload | IdentitySetValuePayload | None:
    if value.kind == ValueKind.IDENTITY and isinstance(
        value.payload,
        IdentityValuePayload,
    ):
        return value.payload
    if value.kind == ValueKind.IDENTITY_SET and isinstance(
        value.payload,
        IdentitySetValuePayload,
    ):
        return value.payload
    return None


def _row_set_filter_payload(
    *,
    value: FactValue | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if value is None:
        return payload
    payload["kind"] = value.kind.value
    payload["value_id"] = value.id
    known_input_id = known_input_id_for_value(value)
    if known_input_id:
        payload["known_input_id"] = known_input_id
    if value.kind == ValueKind.IDENTITY and isinstance(
        value.payload,
        IdentityValuePayload,
    ):
        payload["entity_kind"] = value.payload.entity_kind
        payload["key_id"] = value.payload.key_id
        payload["key_components"] = [
            {
                "component_id": component.component_id,
                "value": str(component.value),
            }
            for component in value.payload.key.components
        ]
        payload["display_value"] = value.payload.display_value or value.label
        if value.payload.matched_field_ref:
            payload["matched_field_ref"] = value.payload.matched_field_ref
        if value.payload.matched_field_path:
            payload["matched_field_path"] = value.payload.matched_field_path
        return payload
    if value.kind == ValueKind.TIME and isinstance(value.payload, TimeValuePayload):
        payload["display_value"] = value.label
        payload["resolved_start"] = value.payload.resolved_start
        payload["resolved_end"] = value.payload.resolved_end
        return payload
    if value.kind == ValueKind.LITERAL and isinstance(
        value.payload,
        LiteralValuePayload,
    ):
        payload["display_value"] = value.label or value.payload.value
        payload["literal_type"] = value.payload.literal_type.value
        return payload
    if value.kind == ValueKind.NAMED and isinstance(value.payload, NamedValuePayload):
        payload["display_value"] = value.payload.reference_text or value.payload.text
        if value.payload.filter_operator is not ValueFilterOperator.EQUALS:
            payload["operator"] = value.payload.filter_operator.value
        if value.payload.matched_field_ref:
            payload["matched_field_ref"] = value.payload.matched_field_ref
        if value.payload.matched_field_path:
            payload["matched_field_path"] = value.payload.matched_field_path
        return payload
    if value.label:
        payload["display_value"] = value.label
    return payload
