"""Model-facing relation surface for catalog-selected reads."""

from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import (
    ParamSource,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.fact_planning.fact_requirements import (
    RequestedFactEndpointRequirement,
    fact_endpoint_requirements,
)
from fervis.lookup.fact_planning.grounded_params import (
    GroundedParamValue,
    unique_grounded_param_values,
)
from fervis.lookup.fact_planning.required_inputs import (
    RequiredInput,
    param_has_bindable_value,
)
from fervis.lookup.fact_plan.row_sources import (
    RowCardinality,
    RowSource,
    RowSourceField,
    RowSourceKind,
    RowSourceParam,
    build_row_source_catalog,
    api_read_source_groups,
    memory_row_source_prompt_payload,
    read_evidence_ref,
    read_row_source_counts,
    required_input_evidence_ref,
    row_source_param_prompt_payload,
)
from fervis.lookup.fact_planning.row_set_filters import (
    row_set_filters_for_sources_payload,
)
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentityValuePayload,
    NamedValuePayload,
    ValueKind,
    known_input_id_for_value,
)
from fervis.lookup.turn_prompts.projections.resolved_inputs import (
    fact_value_prompt_payload,
)
from fervis.lookup.grounding.model import GroundedInputUse


def selected_relation_catalog_payload(
    catalog: RelationCatalog,
    *,
    catalog_selection: CatalogSelectionResult,
    memory_inputs: dict[str, Any],
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[GroundedInputUse, ...],
) -> dict[str, Any]:
    """Project catalog-selected rows for downstream candidate assessment."""

    row_sources = build_row_source_catalog(catalog)
    row_sources_by_id = {source.id: source for source in row_sources.sources}
    grounded_params = unique_grounded_param_values(
        values=available_values,
        grounded_input_uses=available_value_uses,
    )
    values_by_id = {value.id: value for value in available_values}
    requirements = fact_endpoint_requirements(
        catalog=catalog,
        catalog_selection=catalog_selection,
        available_values=available_values,
        available_value_uses=available_value_uses,
        row_sources=row_sources,
    )

    missing_inputs: list[dict[str, Any]] = []
    for source in row_sources.sources:
        if source.id in requirements.executable_row_source_ids:
            continue
        missing_inputs.extend(
            _missing_required_input_payload(item)
            for item in requirements.clarifiable_missing_inputs
            if item.row_source_id == source.id
        )
    payload: dict[str, Any] = {
        "requested_fact_relations": [
            _requested_fact_relation_payload(
                item,
                catalog_selection=catalog_selection,
                row_sources_by_id=row_sources_by_id,
                available_values=available_values,
                available_value_uses=available_value_uses,
                values_by_id=values_by_id,
            )
            for item in requirements.requested_facts
        ]
    }
    utility_relations = [
        _generated_relation_payload(
            source,
            grounded_params=grounded_params,
            values_by_id=values_by_id,
        )
        for source in row_sources.sources
        if source.kind == RowSourceKind.GENERATED_CALENDAR
        and _source_is_executable(
            source,
            grounded_params=grounded_params,
            available_values=available_values,
        )
    ]
    if utility_relations:
        payload["utility_relations"] = utility_relations
    memory_relations = [
        _memory_available_relation_payload(item)
        for item in memory_row_source_prompt_payload(memory_inputs)
    ]
    if memory_relations:
        payload["memory_relations"] = memory_relations
    if missing_inputs:
        payload["missing_required_inputs"] = missing_inputs
    return payload

def _requested_fact_relation_payload(
    requirement: RequestedFactEndpointRequirement,
    *,
    catalog_selection: CatalogSelectionResult,
    row_sources_by_id: dict[str, RowSource],
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[GroundedInputUse, ...],
    values_by_id: dict[str, FactValue],
) -> dict[str, Any]:
    requested_fact_id = requirement.requested_fact_id
    grounded_params = unique_grounded_param_values(
        values=available_values,
        grounded_input_uses=available_value_uses,
        requested_fact_id=requested_fact_id,
    )
    selected_sources = tuple(
        source
        for row_source_id in requirement.selected_row_source_ids
        for source in (row_sources_by_id.get(row_source_id),)
        if source is not None
    )
    return {
        "requested_fact_id": requested_fact_id,
        "query_terms": list(
            _query_terms_for_fact(
                catalog_selection,
                requested_fact_id=requested_fact_id,
            )
        ),
        "available_relations": _available_read_payloads(
            sources=selected_sources,
            grounded_params=grounded_params,
            values_by_id=values_by_id,
            requested_fact_id=requested_fact_id,
        ),
    }


def _query_terms_for_fact(
    catalog_selection: CatalogSelectionResult,
    *,
    requested_fact_id: str,
) -> tuple[str, ...]:
    for selection in catalog_selection.requested_fact_selections:
        if selection.requested_fact_id == requested_fact_id:
            return tuple(selection.query_terms)
    return ()


def _available_read_payloads(
    *,
    sources: tuple[RowSource, ...],
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    values_by_id: dict[str, FactValue],
    requested_fact_id: str,
) -> list[dict[str, Any]]:
    api_source_groups = api_read_source_groups(sources)
    row_source_counts = read_row_source_counts(api_source_groups)
    return [
        _available_api_read_payload(
            read_id=group_sources[0].read_id,
            sources=tuple(group_sources),
            read_row_source_count=row_source_counts[group_sources[0].read_id],
            grounded_params=grounded_params,
            values_by_id=values_by_id,
            requested_fact_id=requested_fact_id,
        )
        for group_sources in api_source_groups
    ]


def _available_api_read_payload(
    *,
    read_id: str,
    sources: tuple[RowSource, ...],
    read_row_source_count: int = 1,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    values_by_id: dict[str, FactValue],
    requested_fact_id: str,
) -> dict[str, Any]:
    representative = _representative_api_source(sources)
    params = _read_bindable_params(sources=sources)
    payload: dict[str, Any] = {
        "read_id": read_id,
        "label": representative.label,
        "kind": RowSourceKind.API_READ.value,
        "evidence_ref": read_evidence_ref(read_id),
        "cardinality": _combined_cardinality(sources),
        "resource_names": list(representative.resource_names),
        "row_source_id": representative.id if len(sources) == 1 else "",
        "row_path_id": representative.row_path_id,
        "read_row_source_count": read_row_source_count,
        **(
            {"description": representative.description}
            if representative.description
            else {}
        ),
        "fields": _combined_source_field_payloads(sources),
        "result_grains": _result_grain_payloads(sources),
    }
    if params:
        payload["params"] = [
            _available_param_payload(
                representative,
                param,
                grounded_params=grounded_params,
            )
            for param in params
        ]
    filters = row_set_filters_for_sources_payload(
        sources=sources,
        grounded_params=grounded_params,
        values_by_id=values_by_id,
        requested_fact_id=requested_fact_id,
    )
    if filters:
        payload["applied_filters"] = filters
    return payload


def _representative_api_source(sources: tuple[RowSource, ...]) -> RowSource:
    return sorted(
        sources,
        key=lambda source: (
            len(tuple(part for part in source.row_path.split(".") if part)),
            len(source.fields),
            source.id,
        ),
        reverse=True,
    )[0]


def _combined_source_field_payloads(
    sources: tuple[RowSource, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        for field in source.fields:
            if field.id in seen:
                continue
            seen.add(field.id)
            output.append(
                _field_payload(
                    read_id=source.read_id,
                    field=field,
                    row_cardinality=source.row_cardinality.value,
                )
            )
    return output


def _result_grain_payloads(sources: tuple[RowSource, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in sources:
        parent_grain_id = _parent_grain_id(source, sources=sources)
        item = {
            "grain_id": source.row_path_id or "root",
            "row_path_id": source.row_path_id or "root",
            "row_source_id": source.id,
            "cardinality": source.row_cardinality.value,
            "evidence_items": [
                {
                    **_field_payload(
                        read_id=source.read_id,
                        field=field,
                        row_cardinality=source.row_cardinality.value,
                    ),
                    "field_ref": field.field_ref,
                }
                for field in source.fields
                if _field_belongs_to_source_grain(source, field)
            ],
            "candidate_keys": [
                {
                    "key_id": key.id,
                    "entity_kind": key.entity_kind,
                    "components": [
                        {
                            "component_id": component.id,
                            "field_id": component.field_id,
                        }
                        for component in key.components
                    ],
                    "primary": key.primary,
                    "stable": key.stable,
                    "context_field_ids": list(key.context_field_ids),
                }
                for key in source.candidate_keys
            ],
            "entity_references": [
                {
                    "reference_id": reference.id,
                    "target_entity_kind": reference.target_entity_kind,
                    "target_key_id": reference.target_key_id,
                    "components": [
                        {
                            "component_id": component.target_component_id,
                            "field_id": component.local_field_id,
                        }
                        for component in reference.components
                    ],
                    "context_field_ids": list(reference.context_field_ids),
                }
                for reference in source.entity_references
            ],
        }
        if parent_grain_id:
            item["parent_grain_id"] = parent_grain_id
        output.append(item)
    return output


def _parent_grain_id(source: RowSource, *, sources: tuple[RowSource, ...]) -> str:
    if not source.parent_row_path:
        return ""
    for candidate in sources:
        if candidate.row_path == source.parent_row_path:
            return candidate.row_path_id or "root"
    return ""


def _field_belongs_to_source_grain(
    source: RowSource,
    field: RowSourceField,
) -> bool:
    row_path = source.row_path
    if not row_path:
        return True
    return field.path.startswith(f"{row_path}.")


def _combined_cardinality(sources: tuple[RowSource, ...]) -> str:
    if any(source.row_cardinality == RowCardinality.MANY for source in sources):
        return RowCardinality.MANY.value
    return sources[0].row_cardinality.value


def _read_bindable_params(
    *,
    sources: tuple[RowSource, ...],
) -> tuple[RowSourceParam, ...]:
    params_by_id: dict[str, RowSourceParam] = {}
    for source in sources:
        for param in source.params:
            if not _param_needs_source_binding(param):
                continue
            existing = params_by_id.get(param.id)
            if existing is None or _prefer_row_source_param(param, existing):
                params_by_id[param.id] = param
    return tuple(params_by_id.values())


def _prefer_row_source_param(
    candidate: RowSourceParam,
    existing: RowSourceParam,
) -> bool:
    if candidate.default_source == "source_variant" and (
        existing.default_source != "source_variant"
    ):
        return True
    return False


def _param_needs_source_binding(param: RowSourceParam) -> bool:
    if param.source == ParamSource.QUERY:
        return True
    return param.required and param.default is None


def _available_param_payload(
    source: RowSource,
    param: RowSourceParam,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
) -> dict[str, Any]:
    payload = row_source_param_prompt_payload(source, param)
    if param.default is not None or (source.id, param.id) in grounded_params:
        payload.pop("required_catalog_input_id", None)
        payload.pop("required_catalog_input_evidence_ref", None)
        payload.pop("required_catalog_choice_input_id", None)
        payload.pop("required_catalog_choice_input_evidence_ref", None)
    return payload


def _field_payload(
    *,
    read_id: str,
    field: RowSourceField,
    row_cardinality: str = "",
) -> dict[str, Any]:
    del read_id
    payload: dict[str, Any] = {
        "field_id": field.id,
        "type": field.type,
        "roles": [role.value for role in field.allowed_roles],
    }
    if row_cardinality:
        payload["row_cardinality"] = row_cardinality
    if field.label and field.label != field.id:
        payload["label"] = field.label
    if field.description:
        payload["description"] = field.description
    if field.answer_output_ids:
        payload["answer_output_ids"] = list(field.answer_output_ids)
    return payload


def _generated_relation_payload(
    source: RowSource,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    values_by_id: dict[str, FactValue],
) -> dict[str, Any]:
    params = source.params
    payload: dict[str, Any] = {
        "calendar_id": "calendar_days",
        "kind": RowSourceKind.GENERATED_CALENDAR.value,
        "fields": [
            _field_payload(read_id="calendar_days", field=field)
            for field in source.fields
        ],
        "result_grains": _result_grain_payloads((source,)),
    }
    if params:
        payload["params"] = [
            _available_param_payload(
                source,
                param,
                grounded_params=grounded_params,
            )
            for param in params
        ]
    filters = row_set_filters_for_sources_payload(
        sources=(source,),
        grounded_params=grounded_params,
        values_by_id=values_by_id,
    )
    if filters:
        payload["applied_filters"] = filters
    return payload


def operation_input_values_payload(
    *,
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[GroundedInputUse, ...],
) -> dict[str, Any]:
    """Return canonical values that still need a model-authored operation sink."""

    row_set_value_ids = {
        item.value_id
        for item in unique_grounded_param_values(
            values=available_values,
            grounded_input_uses=available_value_uses,
        ).values()
    }
    return {
        "values": [
            _operation_value_payload(value)
            for value in available_values
            if value.id not in row_set_value_ids
        ]
    }


def _memory_available_relation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output.pop("params", None)
    return output


def _missing_required_input_payload(item: RequiredInput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "row_source_id": item.row_source_id,
        "input_label": item.param_label,
        "input_type": item.param_type,
        "evidence_ref": required_input_evidence_ref(required_input_id=item.id),
    }
    if item.choices:
        payload["required_catalog_choice_input_id"] = item.id
        payload["choices"] = list(item.choices)
        if item.choice_labels:
            payload["choice_labels"] = dict(item.choice_labels)
        return payload
    payload["required_catalog_input_id"] = item.id
    return payload


def _source_is_executable(
    source: RowSource,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    available_values: tuple[FactValue, ...],
) -> bool:
    return all(
        not param.required
        or param.default is not None
        or (source.id, param.id) in grounded_params
        or param_has_bindable_value(param, available_values=available_values)
        for param in source.params
    )


def _operation_value_payload(value: FactValue) -> dict[str, object]:
    payload: dict[str, object] = dict(fact_value_prompt_payload(value))
    if value.label:
        payload["label"] = value.label
    known_input_id = known_input_id_for_value(value)
    if known_input_id:
        payload["known_input_id"] = known_input_id
    if value.applies_to_requested_fact_ids:
        payload["applies_to_requested_facts"] = list(
            value.applies_to_requested_fact_ids
        )
    if value.kind == ValueKind.IDENTITY and isinstance(
        value.payload,
        IdentityValuePayload,
    ):
        if value.payload.matched_field_ref:
            payload["matched_field_ref"] = value.payload.matched_field_ref
        if value.payload.matched_field_path:
            payload["matched_field_path"] = value.payload.matched_field_path
    elif value.kind == ValueKind.NAMED and isinstance(value.payload, NamedValuePayload):
        if value.payload.matched_field_ref:
            payload["matched_field_ref"] = value.payload.matched_field_ref
        if value.payload.matched_field_path:
            payload["matched_field_path"] = value.payload.matched_field_path
    return payload
