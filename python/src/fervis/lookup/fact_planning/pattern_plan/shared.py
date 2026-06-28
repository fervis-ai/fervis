"""Shared primitives for pattern fact-plan compilation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from fervis.lookup.fact_plan.fact_plan import FactFulfillment, FactValue
from fervis.lookup.fact_plan.operations import (
    Operation,
    SortDirection,
    UnionSpec,
)
from fervis.lookup.fact_plan.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    RelationSourceAppliedFilter,
    RelationSourceRowFilter,
)
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
)
from fervis.lookup.fact_plan.values import ValueUse
from fervis.lookup.fact_planning.fulfillment_evidence import (
    evidence_is_compatible_with_plan_shape,
    field_id_for_fulfillment_evidence,
    group_key_evidence_ids,
    required_fulfillment_evidence_ids,
    source_cardinality_by_evidence_id,
    source_field_id_by_evidence_id,
    value_evidence_ids_for_plan,
)
from fervis.lookup.fact_planning.metric_options import metric_for_selection
from fervis.lookup.source_binding import BoundSource

from .render_ids import _safe_field_id


def _compiled_pattern(
    *,
    payload: dict[str, Any],
    relation_id: str,
    relation_fields: tuple[RelationField, ...],
    operations: tuple[Operation, ...],
    relation_outputs: tuple[RenderRelationOutput, ...],
    fulfillment_render_ids: tuple[str, ...],
    bound_sources: dict[str, BoundSource],
    values: tuple[FactValue, ...] = (),
    value_uses: tuple[ValueUse, ...] = (),
    required_answer_evidence_ids_by_output: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    requested_fact_id = _text(payload.get("requested_fact_id"))
    explicit_answer_output_ids = _required_strings(
        payload.get("answer_output_ids"), "answer_output_ids"
    )
    _validate_no_implicit_answer_output_coverage(
        payload=payload,
        relation_fields=relation_fields,
        explicit_answer_output_ids=explicit_answer_output_ids,
        bound_sources=bound_sources,
    )
    _validate_fulfillment_render_ids(
        answer_output_ids=explicit_answer_output_ids,
        fulfillment_render_ids=fulfillment_render_ids,
    )
    fulfillment = tuple(
        FactFulfillment(
            requested_fact_id=requested_fact_id,
            answer_output_id=answer_output_id,
            render_output_id=fulfillment_render_ids[index],
        )
        for index, answer_output_id in enumerate(explicit_answer_output_ids)
    )
    relation_inputs = _relations_for_bound_source(
        relation_id=relation_id,
        payload=payload,
        relation_fields=relation_fields,
        bound_sources=bound_sources,
        required_answer_evidence_ids_by_output=(required_answer_evidence_ids_by_output),
    )
    return {
        "fulfillment": fulfillment,
        "values": values,
        "value_uses": value_uses,
        "relations": relation_inputs["relations"],
        "operations": (*relation_inputs["operations"], *operations),
        "relation_outputs": relation_outputs,
        "scalar_outputs": (),
    }


def _validate_no_implicit_answer_output_coverage(
    *,
    payload: dict[str, Any],
    relation_fields: tuple[RelationField, ...],
    explicit_answer_output_ids: tuple[str, ...],
    bound_sources: dict[str, BoundSource],
) -> None:
    bound = bound_sources.get(_text(payload.get("source_binding_id")))
    if bound is None:
        return
    requested_fact_id = _text(payload.get("requested_fact_id"))
    selected_field_ids = {field.field_id for field in relation_fields}
    selected_field_ids.update(_bound_param_field_ids(bound))
    field_id_by_evidence_id = source_field_id_by_evidence_id(bound)
    explicit_ids = set(explicit_answer_output_ids)
    for fulfillment in bound.fulfillments:
        if fulfillment.requested_fact_id != requested_fact_id:
            continue
        if fulfillment.answer_output_id in explicit_ids:
            continue
        required_field_ids = {
            field_id
            for evidence_id in value_evidence_ids_for_plan(fulfillment)
            for field_id in (
                field_id_for_fulfillment_evidence(
                    evidence_id,
                    field_id_by_evidence_id=field_id_by_evidence_id,
                    available_field_ids=set(bound.available_field_ids),
                ),
            )
            if field_id
        }
        if required_field_ids and required_field_ids <= selected_field_ids:
            raise ValueError("fact plan implicitly covers unlisted answer output")


def _validate_fulfillment_render_ids(
    *,
    answer_output_ids: tuple[str, ...],
    fulfillment_render_ids: tuple[str, ...],
) -> None:
    if len(fulfillment_render_ids) < len(answer_output_ids):
        raise ValueError("fact plan missing render output for answer output")


def _relations_for_bound_source(
    *,
    relation_id: str,
    payload: dict[str, Any],
    relation_fields: tuple[RelationField, ...],
    bound_sources: dict[str, BoundSource],
    required_answer_evidence_ids_by_output: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    bound = _bound_source(payload, bound_sources=bound_sources)
    invocations = bound.source_invocations or (
        (bound.source,) if bound.source is not None else ()
    )
    if len(invocations) <= 1:
        return {
            "relations": (
                _relation_for_bound(
                    relation_id=relation_id,
                    payload=payload,
                    bound=bound,
                    relation_fields=relation_fields,
                    required_answer_evidence_ids_by_output=(
                        required_answer_evidence_ids_by_output
                    ),
                ),
            ),
            "operations": (),
        }
    relations = tuple(
        _relation_for_bound(
            relation_id=f"{relation_id}_invocation_{index}",
            payload=payload,
            bound=_bound_source_with_source(bound, source=source),
            relation_fields=relation_fields,
            required_answer_evidence_ids_by_output=(
                required_answer_evidence_ids_by_output
            ),
        )
        for index, source in enumerate(invocations, start=1)
    )
    return {
        "relations": relations,
        "operations": (
            Operation(
                id=f"{relation_id}_union",
                spec=UnionSpec(
                    inputs=tuple(relation.id for relation in relations),
                    output_fields=tuple(field.field_id for field in relation_fields),
                    identity_fields=tuple(
                        field.field_id
                        for field in relation_fields
                        if FieldBindingRole.IDENTITY in field.roles
                    ),
                ),
                output_relation=relation_id,
            ),
        ),
    }


def _relation_for_bound_source(
    *,
    relation_id: str,
    payload: dict[str, Any],
    relation_fields: tuple[RelationField, ...],
    bound_sources: dict[str, BoundSource],
    required_answer_evidence_ids_by_output: Mapping[str, tuple[str, ...]] | None = None,
) -> Relation:
    bound = _bound_source(payload, bound_sources=bound_sources)
    return _relation_for_bound(
        relation_id=relation_id,
        payload=payload,
        bound=bound,
        relation_fields=relation_fields,
        required_answer_evidence_ids_by_output=required_answer_evidence_ids_by_output,
    )


def _bound_source_with_source(bound: BoundSource, *, source: Any) -> BoundSource:
    return BoundSource(
        id=bound.id,
        requested_fact_id=bound.requested_fact_id,
        answer_population=bound.answer_population,
        source=source,
        value_id=bound.value_id,
        source_candidate_id=bound.source_candidate_id,
        cardinality=bound.cardinality,
        fulfillments=bound.fulfillments,
        evidence_items=bound.evidence_items,
        available_field_ids=bound.available_field_ids,
        available_fields=bound.available_fields,
        applied_filters=bound.applied_filters,
    )


def _relation_for_bound(
    *,
    relation_id: str,
    payload: dict[str, Any],
    bound: BoundSource,
    relation_fields: tuple[RelationField, ...],
    required_answer_evidence_ids_by_output: Mapping[str, tuple[str, ...]] | None = None,
) -> Relation:
    if bound.source is None:
        raise ValueError("fact plan references unknown relation source binding")
    source_filters = _relation_source_filters(bound.applied_filters)
    row_filters = tuple(bound.source.row_filters) if bound.source is not None else ()
    relation_fields = _relation_fields_with_source_filters(
        relation_fields,
        source_filters=source_filters,
        row_filters=row_filters,
    )
    source = _source_with_filters(
        bound.source,
        source_filters=source_filters,
        row_filters=row_filters,
    )
    _validate_relation_fields_for_bound(
        payload=payload,
        bound=bound,
        relation_fields=relation_fields,
        required_answer_evidence_ids_by_output=required_answer_evidence_ids_by_output,
    )
    return Relation(id=relation_id, source=source, fields=relation_fields)


def _source_with_filters(
    source: RelationSource,
    *,
    source_filters: tuple[RelationSourceAppliedFilter, ...],
    row_filters: tuple[RelationSourceRowFilter, ...],
) -> RelationSource:
    if not source_filters and not row_filters:
        return source
    return replace(source, applied_filters=source_filters, row_filters=row_filters)


def _relation_source_filters(
    applied_filters: tuple[dict[str, Any], ...],
) -> tuple[RelationSourceAppliedFilter, ...]:
    return RelationSourceAppliedFilter.from_payloads(applied_filters)


def _relation_fields_with_source_filters(
    relation_fields: tuple[RelationField, ...],
    *,
    source_filters: tuple[RelationSourceAppliedFilter, ...],
    row_filters: tuple[RelationSourceRowFilter, ...],
) -> tuple[RelationField, ...]:
    if not source_filters and not row_filters:
        return relation_fields
    output = list(relation_fields)
    existing = {field.field_id for field in output}
    for field_id in (
        field_id
        for source_filter in source_filters
        for field_id in source_filter.predicate_field_ids
    ):
        if field_id in existing:
            continue
        output.append(
            RelationField(
                field_id=field_id,
                roles=(FieldBindingRole.PREDICATE,),
            )
        )
        existing.add(field_id)
    for source_filter in row_filters:
        field_id = source_filter.field_id
        if field_id in existing:
            continue
        output.append(
            RelationField(
                field_id=field_id,
                roles=(FieldBindingRole.PREDICATE,),
            )
        )
        existing.add(field_id)
    return tuple(output)


def _validate_relation_fields_for_bound(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    relation_fields: tuple[RelationField, ...],
    required_answer_evidence_ids_by_output: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    if bound.source is None:
        raise ValueError("fact plan references unknown relation source binding")
    _validate_committed_source_fields(bound, relation_fields=relation_fields)
    _validate_required_fulfillment_evidence(
        payload=payload,
        bound=bound,
        relation_fields=relation_fields,
        required_answer_evidence_ids_by_output=required_answer_evidence_ids_by_output,
    )


def _bound_source(
    payload: dict[str, Any],
    *,
    bound_sources: dict[str, BoundSource],
) -> BoundSource:
    source_binding_id = _text(payload.get("source_binding_id"))
    bound = bound_sources.get(source_binding_id)
    if bound is None:
        raise ValueError("fact plan references unknown relation source binding")
    return bound


def _validate_committed_source_fields(
    bound: BoundSource,
    *,
    relation_fields: tuple[RelationField, ...],
) -> None:
    available_field_ids = set(bound.available_field_ids)
    selected_field_ids = {field.field_id for field in relation_fields}
    if selected_field_ids and not available_field_ids:
        raise ValueError("fact plan source has no available fields")
    if selected_field_ids - available_field_ids:
        missing = tuple(sorted(selected_field_ids - available_field_ids))
        raise ValueError(f"fact plan references unavailable source field: {missing}")


def _validate_required_fulfillment_evidence(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    relation_fields: tuple[RelationField, ...],
    required_answer_evidence_ids_by_output: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    selected_field_ids = {field.field_id for field in relation_fields}
    selected_field_ids.update(_bound_param_field_ids(bound))
    count_answer_output_id = _selected_count_records_answer_output_id(
        payload=payload,
        bound=bound,
    )
    required_fields = (
        _field_ids_by_answer_output_from_selected_evidence(
            bound,
            required_answer_evidence_ids_by_output,
        )
        if required_answer_evidence_ids_by_output is not None
        else _answer_value_field_ids_by_answer_output(payload=payload, bound=bound)
    )
    for answer_output_id, field_ids in required_fields.items():
        if answer_output_id == count_answer_output_id:
            continue
        if set(field_ids) - selected_field_ids:
            raise ValueError("fact plan dropped required answer value evidence")


def _field_ids_by_answer_output_from_selected_evidence(
    bound: BoundSource,
    evidence_ids_by_answer_output: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    field_id_by_evidence_id = source_field_id_by_evidence_id(bound)
    available_field_ids = set(bound.available_field_ids)
    return {
        answer_output_id: tuple(
            field_id
            for evidence_id in evidence_ids
            for field_id in (
                field_id_for_fulfillment_evidence(
                    evidence_id,
                    field_id_by_evidence_id=field_id_by_evidence_id,
                    available_field_ids=available_field_ids,
                ),
            )
            if field_id
        )
        for answer_output_id, evidence_ids in evidence_ids_by_answer_output.items()
    }


def _selected_count_records_answer_output_id(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
) -> str:
    if not isinstance(payload.get("metric"), dict):
        return ""
    selected_metric = metric_for_selection(
        payload=payload,
        bound_sources={bound.id: bound},
    )
    return _text(selected_metric.get("answer_output_id"))


def _bound_param_field_ids(bound: BoundSource) -> set[str]:
    if bound.source is None:
        return set()
    return {binding.param_id for binding in bound.source.param_bindings}


def _required_answer_value_field_ids(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
) -> set[str]:
    return {
        field_id
        for field_ids in _answer_value_field_ids_by_answer_output(
            payload=payload,
            bound=bound,
        ).values()
        for field_id in field_ids
    }


def _answer_value_field_ids_by_answer_output(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
) -> dict[str, tuple[str, ...]]:
    return _field_ids_by_answer_output(
        payload=payload,
        bound=bound,
        evidence_ids_by_fulfillment=_required_evidence_ids_for_plan,
    )


def _render_value_field_ids_by_answer_output(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for field_ids_by_answer_output in (
        _field_ids_by_answer_output(
            payload=payload,
            bound=bound,
            evidence_ids_by_fulfillment=_group_key_evidence_ids_for_plan,
        ),
        _answer_value_field_ids_by_answer_output(payload=payload, bound=bound),
    ):
        for answer_output_id, field_ids in field_ids_by_answer_output.items():
            output.setdefault(answer_output_id, field_ids)
    return output


def _field_ids_by_answer_output(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    evidence_ids_by_fulfillment: Any,
) -> dict[str, tuple[str, ...]]:
    requested_fact_id = _text(payload.get("requested_fact_id"))
    if not requested_fact_id or "answer_output_ids" not in payload:
        return {}
    answer_output_ids = set(
        _required_strings(payload.get("answer_output_ids"), "answer_output_ids")
    )
    field_id_by_evidence_id = source_field_id_by_evidence_id(bound)
    cardinality_by_evidence_id = source_cardinality_by_evidence_id(bound)
    available_field_ids = set(bound.available_field_ids)
    plan_shape = _text(payload.get("pattern"))
    output: dict[str, tuple[str, ...]] = {}
    for fulfillment in bound.fulfillments:
        if fulfillment.requested_fact_id != requested_fact_id:
            continue
        if fulfillment.answer_output_id not in answer_output_ids:
            continue
        scope_evidence_ids = set(fulfillment.scope_evidence_ids)
        for evidence_id in evidence_ids_by_fulfillment(
            fulfillment,
            plan_shape=plan_shape,
        ):
            if evidence_id in scope_evidence_ids:
                continue
            if not evidence_is_compatible_with_plan_shape(
                cardinality_by_evidence_id.get(evidence_id, ""),
                plan_shape=plan_shape,
            ):
                continue
            field_id = field_id_for_fulfillment_evidence(
                evidence_id,
                field_id_by_evidence_id=field_id_by_evidence_id,
                available_field_ids=available_field_ids,
            )
            if field_id or evidence_id:
                output.setdefault(fulfillment.answer_output_id, ())
            output[fulfillment.answer_output_id] = (
                *output[fulfillment.answer_output_id],
                field_id or evidence_id,
            )
    return output


def _required_evidence_ids_for_plan(
    fulfillment: Any,
    *,
    plan_shape: str,
) -> tuple[str, ...]:
    return required_fulfillment_evidence_ids(fulfillment, plan_shape=plan_shape)


def _group_key_evidence_ids_for_plan(
    fulfillment: Any,
    *,
    plan_shape: str,
) -> tuple[str, ...]:
    if plan_shape not in {"aggregate_by_group", "ranked_aggregate"}:
        return ()
    return group_key_evidence_ids(fulfillment)


def _validate_metric_source_compatibility(
    *,
    payload: dict[str, Any],
    metric: dict[str, Any],
    bound_sources: dict[str, BoundSource],
) -> None:
    record_id_field_id = metric["record_id_field_id"]
    if not record_id_field_id:
        return
    bound = _bound_source(payload, bound_sources=bound_sources)
    numeric_answer_field_ids = {
        field_id
        for field_id in _required_answer_value_field_ids(payload=payload, bound=bound)
        if _source_field_is_numeric(bound, field_id)
    }
    if numeric_answer_field_ids:
        raise ValueError("count_records metric cannot replace numeric answer evidence")
    source_field = next(
        (
            field
            for field in bound.available_fields
            if field.field_id == record_id_field_id
        ),
        None,
    )
    if source_field is None:
        return
    if source_field.type.lower() in {
        "integer",
        "number",
        "decimal",
        "float",
        "double",
    }:
        raise ValueError(
            "count_records metric requires a non-numeric row identity field"
        )


def _source_field_is_numeric(bound: BoundSource, field_id: str) -> bool:
    return any(
        field.field_id == field_id
        and field.type.lower() in {"integer", "number", "decimal", "float", "double"}
        for field in bound.available_fields
    )


def _field_specs(value: Any) -> tuple[dict[str, str], ...]:
    return tuple(_field_spec(item) for item in _dicts(value))


def _field_spec(payload: dict[str, Any]) -> dict[str, str]:
    if "label" in payload:
        raise ValueError("raw field selections must not include label")
    field_id = _text(payload.get("field_id"))
    return {
        "field_id": field_id,
        "label": field_id,
        "output_field_id": _safe_field_id(field_id),
    }


def _rank_spec(payload: dict[str, Any]) -> dict[str, Any]:
    limit = payload.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("rank.limit must be a positive integer")
    limit_value_id = _text(payload.get("limit_value_id"))
    return {
        "limit": limit,
        "limit_value_id": limit_value_id,
        "sort": _enum(SortDirection, payload.get("sort"), "rank.sort"),
    }


def _scalar_output_spec(payload: dict[str, Any]) -> dict[str, str]:
    scalar_id = _safe_field_id(_text(payload.get("scalar_id")) or "value")
    label = _text(payload.get("label")) or scalar_id
    return {
        "scalar_id": scalar_id,
        "label": label,
        "output_id": _safe_field_id(label or scalar_id),
    }


def _relation_operand(payload: dict[str, Any]) -> dict[str, Any]:
    if not _text(payload.get("source_binding_id")):
        raise ValueError("relation operand requires source_binding_id")
    return dict(payload)


def _identity_relation_fields(field_ids: tuple[str, ...]) -> tuple[RelationField, ...]:
    return tuple(
        RelationField(
            field_id=field_id,
            roles=(FieldBindingRole.IDENTITY, FieldBindingRole.OUTPUT),
        )
        for field_id in field_ids
    )


def _join_key_specs(value: Any) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "left_field_id": _text(item.get("left_field_id")),
            "right_field_id": _text(item.get("right_field_id")),
        }
        for item in _required_dicts(value, "join_keys")
    )


def _joined_output_fields(value: Any) -> tuple[dict[str, str], ...]:
    return tuple(
        _field_spec(
            {
                "field_id": item.get("field_id"),
            }
        )
        for item in _required_dicts(value, "output_fields")
    )


def _relation_fields(
    fields: tuple[dict[str, str], ...],
    *,
    identity: bool = False,
    bound_source: BoundSource | None = None,
) -> tuple[RelationField, ...]:
    source_identity_field_ids = (
        _source_role_identity_field_ids(bound_source)
        if bound_source is not None
        else set()
    )
    return tuple(
        RelationField(
            field_id=item["field_id"],
            roles=_relation_field_roles(
                item["field_id"],
                identity=identity,
                source_identity_field_ids=source_identity_field_ids,
            ),
        )
        for item in fields
    )


def _relation_field_roles(
    field_id: str,
    *,
    identity: bool,
    source_identity_field_ids: set[str],
) -> tuple[FieldBindingRole, ...]:
    if identity or field_id in source_identity_field_ids:
        return (FieldBindingRole.IDENTITY, FieldBindingRole.OUTPUT)
    return (FieldBindingRole.OUTPUT,)


def _source_role_identity_field_ids(bound_source: BoundSource) -> set[str]:
    return {
        field.field_id
        for field in bound_source.available_fields
        if FieldBindingRole.IDENTITY.value in {str(role) for role in field.roles}
        or FieldBindingRole.IDENTITY in set(field.roles)
    }


def _pattern_relation_id(index: int) -> str:
    return f"answer_{index}_source"


def _pattern_output_relation_id(index: int) -> str:
    return f"answer_{index}_rows"


def _dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    return tuple(_required_dicts(value, "items"))


def _required_dicts(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{label} items must be objects")
    return tuple(value)


def _required_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a list")
    result = tuple(_text(item) for item in value)
    if any(not item for item in result):
        raise ValueError(f"{label} items must be strings")
    return result


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _enum(enum_type: type, value: Any, label: str):
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {label}: {value}") from exc


__all__ = tuple(name for name in globals() if not name.startswith("__"))
