"""Bound-source prompt payload projection."""

from ._shared import Any, BoundSource, SourceKind


def _bound_sources_prompt_payload(
    *,
    bound_sources: tuple[BoundSource, ...],
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    for bound in bound_sources:
        item: dict[str, Any] = {
            "source_binding_id": bound.id,
            "source_candidate_id": bound.source_candidate_id or bound.id,
            "kind": _bound_source_kind(bound),
        }
        if bound.answer_population is not None:
            item["answer_population"] = {
                "population_binding_id": (
                    bound.answer_population.population_binding_id
                ),
                "intent_text": bound.answer_population.intent_text,
                "match_basis_explanation": (
                    bound.answer_population.match_basis_explanation
                ),
            }
        if bound.requested_fact_id:
            item["requested_fact_id"] = bound.requested_fact_id
        if bound.fulfillments:
            item["fulfills"] = [
                _fulfillment_prompt_payload(fulfillment)
                for fulfillment in bound.fulfillments
            ]
        cardinality = bound.cardinality
        if cardinality:
            item["cardinality"] = cardinality
        if bound.source is not None:
            if bound.source.read_id:
                item["read_id"] = bound.source.read_id
            if bound.source.memory_relation_id:
                item["memory_relation_id"] = bound.source.memory_relation_id
            if bound.source.calendar_id:
                item["calendar_id"] = bound.source.calendar_id
            if bound.source.param_bindings:
                item["bound_params"] = [
                    _bound_param_payload(param) for param in bound.source.param_bindings
                ]
            if len(bound.source_invocations) > 1:
                item["execution"] = {
                    "kind": "consolidated_source_invocations",
                    "pagination_policy": "all_invocations_must_be_complete",
                    "source_invocations": [
                        {
                            "bound_params": [
                                _bound_param_payload(param)
                                for param in source.param_bindings
                            ]
                        }
                        for source in bound.source_invocations
                    ],
                }
            fields = _source_fields_with_evidence_ids(
                bound,
                fulfilled_evidence_ids=_bound_source_fulfilled_evidence_ids(bound),
            )
            if fields:
                item["fields"] = fields
            if bound.applied_filters:
                item["applied_filters"] = [
                    dict(applied_filter) for applied_filter in bound.applied_filters
                ]
        if bound.value_id:
            item["value_id"] = bound.value_id
        output.append(item)
    return {"bound_sources": output}


def _bound_param_payload(param: Any) -> dict[str, Any]:
    output = {"param_id": param.param_id, "value": param.value}
    if param.proof_refs:
        output["proof_refs"] = list(param.proof_refs)
    return output


def _bound_source_kind(bound: BoundSource) -> str:
    if bound.value_id:
        return "value"
    if bound.source is None:
        return "value"
    if bound.source.kind == SourceKind.API_READ:
        return "new_api_read"
    if bound.source.kind == SourceKind.GENERATED_CALENDAR:
        return "generated_calendar"
    if bound.source.kind == SourceKind.MEMORY_READ:
        return "prior_answer_rows"
    return bound.source.kind.value


def _fulfillment_prompt_payload(fulfillment: Any) -> dict[str, Any]:
    output = {
        "requested_fact_id": fulfillment.requested_fact_id,
        "answer_output_id": fulfillment.answer_output_id,
        "match_basis_explanation": fulfillment.match_basis_explanation,
        "metric_measure_evidence_ids": list(fulfillment.metric_measure_evidence_ids),
        "row_count_basis_evidence_ids": list(fulfillment.row_count_basis_evidence_ids),
        "scope_evidence_ids": list(fulfillment.scope_evidence_ids),
    }
    if fulfillment.group_key_evidence_ids:
        output["group_key_evidence_ids"] = list(fulfillment.group_key_evidence_ids)
    return output


def _source_fields_with_evidence_ids(
    bound: BoundSource,
    *,
    fulfilled_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    return _fields_with_evidence_ids(
        _bound_available_fields(bound),
        evidence_items=_bound_evidence_items(bound),
        fulfilled_evidence_ids=fulfilled_evidence_ids,
    )


def _fields_with_evidence_ids(
    fields: tuple[dict[str, Any], ...],
    *,
    evidence_items: Any,
    fulfilled_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    evidence_id_by_field_id = {
        str(item.get("field_id") or ""): str(item.get("evidence_id") or "")
        for item in evidence_items or ()
        if isinstance(item, dict)
        and str(item.get("field_id") or "")
        and str(item.get("evidence_id") or "")
    }
    output: list[dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        item = dict(field)
        field_id = str(item.get("field_id") or item.get("id") or "")
        evidence_id = evidence_id_by_field_id.get(field_id)
        if evidence_id and evidence_id in fulfilled_evidence_ids:
            item["evidence_id"] = evidence_id
        output.append(item)
    return output


def _bound_available_fields(bound: BoundSource) -> tuple[dict[str, Any], ...]:
    cardinality_by_field_id = _bound_field_cardinalities(bound)
    if not bound.available_fields:
        return tuple(
            {
                "field_id": field_id,
                **(
                    {"row_cardinality": cardinality_by_field_id[field_id]}
                    if field_id in cardinality_by_field_id
                    else {}
                ),
            }
            for field_id in bound.available_field_ids
            if field_id
        )
    return tuple(
        {
            "field_id": field.field_id,
            **({"label": field.label} if field.label else {}),
            **({"type": field.type} if field.type else {}),
            **({"roles": list(field.roles)} if field.roles else {}),
            **_bound_field_row_cardinality(field, cardinality_by_field_id),
        }
        for field in bound.available_fields
    )


def _bound_field_row_cardinality(
    field: Any,
    cardinality_by_field_id: dict[str, str],
) -> dict[str, str]:
    row_cardinality = field.row_cardinality or cardinality_by_field_id.get(
        field.field_id,
        "",
    )
    if not row_cardinality:
        return {}
    return {"row_cardinality": row_cardinality}


def _bound_field_cardinalities(bound: BoundSource) -> dict[str, str]:
    values_by_field_id: dict[str, set[str]] = {}
    for item in bound.evidence_items:
        if item.field_id and item.row_cardinality:
            values_by_field_id.setdefault(item.field_id, set()).add(
                item.row_cardinality
            )
    return {
        field_id: next(iter(values))
        for field_id, values in values_by_field_id.items()
        if len(values) == 1
    }


def _bound_evidence_items(bound: BoundSource) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "evidence_id": item.evidence_id,
            **({"field_id": item.field_id} if item.field_id else {}),
            **({"value_id": item.value_id} if item.value_id else {}),
            **({"type": item.type} if item.type else {}),
            **(
                {"row_cardinality": item.row_cardinality}
                if item.row_cardinality
                else {}
            ),
        }
        for item in bound.evidence_items
    )


def _bound_source_fulfilled_evidence_ids(bound: BoundSource) -> set[str]:
    return {
        evidence_id
        for fulfillment in bound.fulfillments
        for evidence_id in fulfillment.all_evidence_ids()
        if evidence_id
    }
