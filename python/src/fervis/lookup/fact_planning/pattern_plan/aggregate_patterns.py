"""Aggregate pattern compilers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.lookup.fact_planning.grouped_ranked_choices import (
    GroupedRankedSelection,
    selected_grouped_ranked_operation,
)
from fervis.lookup.fact_plan.relations import (
    FieldBindingRole,
    RelationField,
    SourceKind,
)
from fervis.lookup.fact_planning.metric_options import metric_for_selection
from fervis.lookup.source_binding import BoundSource

from .aggregate_operations import (
    _aggregate_operations,
    _rank_value_uses,
    _ranked_aggregate_operations,
)
from .aggregate_outputs import (
    _aggregate_relation_outputs,
    _grouped_ranked_relation_outputs,
)
from .shared import (
    _bound_source,
    _compiled_pattern,
    _dict,
    _field_spec,
    _pattern_output_relation_id,
    _pattern_relation_id,
    _rank_spec,
    _relation_fields,
    _render_value_field_ids_by_answer_output,
    _required_strings,
    _text,
    _validate_metric_source_compatibility,
    _validate_relation_fields_for_bound,
)


def _compile_aggregate_pattern_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
) -> dict[str, Any]:
    if _text(payload.get("pattern")) == "aggregate_by_group":
        return _compile_grouped_ranked_aggregate_answer(
            index=index,
            payload=payload,
            namespace_render_outputs=namespace_render_outputs,
            bound_sources=bound_sources,
        )
    relation_id = _pattern_relation_id(index)
    output_relation_id = _pattern_output_relation_id(index)
    group_fields: tuple[dict[str, str], ...] = ()
    metric = metric_for_selection(payload=payload, bound_sources=bound_sources)
    bound = _bound_source(payload, bound_sources=bound_sources)
    row_population_basis = metric.get("row_population_basis")
    if (
        row_population_basis
        and isinstance(row_population_basis, dict)
        and bound.source is not None
    ):
        bound = _bound_source_with_row_population_basis(
            bound,
            row_population_basis=row_population_basis,
        )
    _validate_metric_source_compatibility(
        payload=payload,
        metric=metric,
        bound_sources=bound_sources,
    )
    carry_fields: tuple[dict[str, str], ...] = ()
    relation_fields = (
        *_relation_fields(group_fields),
        *_relation_fields(carry_fields, identity=False),
        *(
            (
                RelationField(
                    field_id=metric["record_id_field_id"],
                    roles=(FieldBindingRole.PREDICATE,),
                ),
            )
            if metric["record_id_field_id"]
            else ()
        ),
        *(
            (
                RelationField(
                    field_id=metric["field_id"],
                    roles=(FieldBindingRole.OUTPUT,),
                ),
            )
            if metric["field_id"]
            else ()
        ),
    )
    _validate_relation_fields_for_bound(
        payload=payload,
        bound=bound,
        relation_fields=relation_fields,
        selected_metric=metric,
    )
    relation_outputs = _aggregate_relation_outputs(
        index=index,
        output_relation_id=output_relation_id,
        group_fields=group_fields,
        carry_fields=_non_group_carry_fields(
            group_fields=group_fields,
            carry_fields=carry_fields,
        ),
        metric=metric,
        namespace_render_outputs=namespace_render_outputs,
    )
    return _compiled_pattern(
        payload=payload,
        relation_id=relation_id,
        relation_fields=relation_fields,
        operations=_aggregate_operations(
            input_relation_id=relation_id,
            output_relation_id=output_relation_id,
            group_fields=group_fields,
            carry_fields=carry_fields,
            metric=metric,
        ),
        relation_outputs=relation_outputs,
        fulfillment_render_ids=_aggregate_fulfillment_render_ids(
            payload=payload,
            bound=bound,
            relation_outputs=relation_outputs,
            metric=metric,
        ),
        bound_sources={**bound_sources, bound.id: bound},
        selected_metric=metric,
    )


def _bound_source_with_row_population_basis(
    bound: BoundSource,
    *,
    row_population_basis: dict[str, object],
) -> BoundSource:
    if bound.source is None or bound.source.kind != SourceKind.API_READ:
        return bound
    row_source_id = _text(row_population_basis.get("row_source_id"))
    if not row_source_id:
        raise ValueError("row population count requires row source")
    return replace(
        bound,
        source=replace(
            bound.source,
            row_source_id=row_source_id,
        ),
        source_invocations=tuple(
            replace(
                source,
                row_source_id=row_source_id,
            )
            for source in bound.source_invocations
        ),
    )


def _compile_ranked_aggregate_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
) -> dict[str, Any]:
    return _compile_grouped_ranked_ranked_aggregate_answer(
        index=index,
        payload=payload,
        namespace_render_outputs=namespace_render_outputs,
        bound_sources=bound_sources,
    )


def _compile_grouped_ranked_aggregate_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
) -> dict[str, Any]:
    relation_id = _pattern_relation_id(index)
    output_relation_id = _pattern_output_relation_id(index)
    selection = selected_grouped_ranked_operation(
        payload,
        bound_sources=bound_sources,
    )
    bound = bound_sources[selection.source_binding_id]
    metric = selection.metric
    row_population_basis = metric.get("row_population_basis")
    if (
        row_population_basis
        and isinstance(row_population_basis, dict)
        and bound.source is not None
    ):
        bound = _bound_source_with_row_population_basis(
            bound,
            row_population_basis=row_population_basis,
        )
        bound_sources = {**bound_sources, bound.id: bound}
    group_fields = _grouped_ranked_group_fields(selection)
    carry_fields: tuple[dict[str, str], ...] = ()
    aggregate_carry_fields = _non_group_carry_fields(
        group_fields=group_fields,
        carry_fields=carry_fields,
    )
    relation_fields = _grouped_ranked_relation_fields(
        group_fields=group_fields,
        carry_fields=carry_fields,
        metric=metric,
    )
    _validate_relation_fields_for_bound(
        payload={
            **payload,
            "source_binding_id": selection.source_binding_id,
        },
        bound=bound,
        relation_fields=relation_fields,
        required_answer_evidence_ids_by_output=(
            _grouped_ranked_answer_evidence_ids_by_output(selection)
        ),
        selected_metric=metric,
    )
    relation_outputs = _grouped_ranked_relation_outputs(
        index=index,
        output_relation_id=output_relation_id,
        answer_outputs=selection.answer_outputs,
        carry_fields=aggregate_carry_fields,
        metric=metric,
        namespace_render_outputs=namespace_render_outputs,
        answer_output_render_ids=False,
    )
    return _compiled_pattern(
        payload={
            **payload,
            "source_binding_id": selection.source_binding_id,
            "answer_output_ids": list(selection.fulfills_answer_output_ids),
        },
        relation_id=relation_id,
        relation_fields=relation_fields,
        operations=_aggregate_operations(
            input_relation_id=relation_id,
            output_relation_id=output_relation_id,
            group_fields=group_fields,
            carry_fields=aggregate_carry_fields,
            metric=metric,
            required_group_fields=_answer_rendered_group_fields(
                selection=selection,
                group_fields=group_fields,
            ),
        ),
        relation_outputs=relation_outputs,
        fulfillment_render_ids=_grouped_ranked_fulfillment_render_ids(
            selection=selection,
            relation_outputs=relation_outputs,
        ),
        bound_sources=bound_sources,
        required_answer_evidence_ids_by_output=(
            _grouped_ranked_answer_evidence_ids_by_output(selection)
        ),
        selected_metric=metric,
    )


def _compile_grouped_ranked_ranked_aggregate_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
) -> dict[str, Any]:
    relation_id = _pattern_relation_id(index)
    aggregate_relation_id = f"{_pattern_output_relation_id(index)}_aggregate"
    output_relation_id = _pattern_output_relation_id(index)
    rank_operation_id = f"{output_relation_id}_rank"
    selection = selected_grouped_ranked_operation(
        payload,
        bound_sources=bound_sources,
    )
    bound = bound_sources[selection.source_binding_id]
    metric = selection.metric
    row_population_basis = metric.get("row_population_basis")
    if (
        row_population_basis
        and isinstance(row_population_basis, dict)
        and bound.source is not None
    ):
        bound = _bound_source_with_row_population_basis(
            bound,
            row_population_basis=row_population_basis,
        )
        bound_sources = {**bound_sources, bound.id: bound}
    group_fields = _grouped_ranked_group_fields(selection)
    carry_fields: tuple[dict[str, str], ...] = ()
    aggregate_carry_fields = _non_group_carry_fields(
        group_fields=group_fields,
        carry_fields=carry_fields,
    )
    relation_fields = _grouped_ranked_relation_fields(
        group_fields=group_fields,
        carry_fields=carry_fields,
        metric=metric,
    )
    _validate_relation_fields_for_bound(
        payload={
            **payload,
            "source_binding_id": selection.source_binding_id,
        },
        bound=bound,
        relation_fields=relation_fields,
        required_answer_evidence_ids_by_output=(
            _grouped_ranked_answer_evidence_ids_by_output(selection)
        ),
        selected_metric=metric,
    )
    rank = _rank_spec(_dict(payload.get("rank"), "rank"))
    relation_outputs = _grouped_ranked_relation_outputs(
        index=index,
        output_relation_id=output_relation_id,
        answer_outputs=selection.answer_outputs,
        carry_fields=carry_fields,
        metric=metric,
        namespace_render_outputs=namespace_render_outputs,
    )
    return _compiled_pattern(
        payload={
            **payload,
            "source_binding_id": selection.source_binding_id,
            "answer_output_ids": list(selection.fulfills_answer_output_ids),
        },
        relation_id=relation_id,
        relation_fields=relation_fields,
        operations=_ranked_aggregate_operations(
            input_relation_id=relation_id,
            aggregate_relation_id=aggregate_relation_id,
            output_relation_id=output_relation_id,
            rank_operation_id=rank_operation_id,
            group_fields=group_fields,
            carry_fields=aggregate_carry_fields,
            metric=metric,
            rank=rank,
            required_group_fields=_answer_rendered_group_fields(
                selection=selection,
                group_fields=group_fields,
            ),
        ),
        relation_outputs=relation_outputs,
        fulfillment_render_ids=_grouped_ranked_fulfillment_render_ids(
            selection=selection,
            relation_outputs=relation_outputs,
        ),
        value_uses=_rank_value_uses(rank_operation_id=rank_operation_id, rank=rank),
        bound_sources=bound_sources,
        required_answer_evidence_ids_by_output=(
            _grouped_ranked_answer_evidence_ids_by_output(selection)
        ),
        selected_metric=metric,
    )


def _grouped_ranked_group_fields(
    selection: GroupedRankedSelection,
) -> tuple[dict[str, str], ...]:
    return (_field_spec({"field_id": selection.group_field_id}),)


def _grouped_ranked_relation_fields(
    *,
    group_fields: tuple[dict[str, str], ...],
    carry_fields: tuple[dict[str, str], ...],
    metric: dict[str, Any],
) -> tuple[RelationField, ...]:
    return (
        *_relation_fields(group_fields),
        *_relation_fields(
            _non_group_carry_fields(
                group_fields=group_fields,
                carry_fields=carry_fields,
            ),
            identity=False,
        ),
        *(
            (
                RelationField(
                    field_id=metric["record_id_field_id"],
                    roles=(FieldBindingRole.PREDICATE,),
                ),
            )
            if metric["record_id_field_id"]
            else ()
        ),
        *(
            (
                RelationField(
                    field_id=metric["field_id"],
                    roles=(FieldBindingRole.OUTPUT,),
                ),
            )
            if metric["field_id"]
            else ()
        ),
    )


def _non_group_carry_fields(
    *,
    group_fields: tuple[dict[str, str], ...],
    carry_fields: tuple[dict[str, str], ...],
) -> tuple[dict[str, str], ...]:
    group_field_ids = {item["field_id"] for item in group_fields}
    return tuple(
        item for item in carry_fields if item["field_id"] not in group_field_ids
    )


def _grouped_ranked_fulfillment_render_ids(
    *,
    selection: GroupedRankedSelection,
    relation_outputs: tuple[Any, ...],
) -> tuple[str, ...]:
    render_id_by_field_id = {
        item.field_id: item.id for item in relation_outputs if item.field_id
    }
    render_id_by_answer_output: dict[str, str] = {}
    for answer_output in selection.answer_outputs:
        render_id = render_id_by_field_id.get(answer_output.field_id, "")
        if not render_id:
            raise ValueError("operation support missing render output field")
        render_id_by_answer_output.setdefault(answer_output.answer_output_id, render_id)
    output: list[str] = []
    for answer_output_id in selection.fulfills_answer_output_ids:
        render_id = render_id_by_answer_output.get(answer_output_id, "")
        if not render_id:
            raise ValueError("operation support missing answer output render")
        output.append(render_id)
    return tuple(output)


def _grouped_ranked_answer_evidence_ids_by_output(
    selection: GroupedRankedSelection,
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for answer_output in selection.answer_outputs:
        if not answer_output.evidence_id:
            continue
        output.setdefault(answer_output.answer_output_id, ())
        output[answer_output.answer_output_id] = (
            *output[answer_output.answer_output_id],
            answer_output.evidence_id,
        )
    return output


def _answer_rendered_group_fields(
    *,
    selection: GroupedRankedSelection,
    group_fields: tuple[dict[str, str], ...],
) -> tuple[str, ...]:
    group_field_ids = {item["field_id"] for item in group_fields}
    return tuple(
        dict.fromkeys(
            answer.field_id
            for answer in selection.answer_outputs
            if answer.field_id in group_field_ids
        )
    )


def _metric_render_ids_by_answer_output(
    *,
    relation_outputs: tuple[Any, ...],
    metric: dict[str, Any],
) -> dict[str, str]:
    answer_output_id = _text(metric.get("answer_output_id"))
    if not answer_output_id:
        return {}
    for output in relation_outputs:
        if output.field_id == metric["output_field_id"]:
            return {answer_output_id: output.id}
    return {}


def _aggregate_fulfillment_render_ids(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    relation_outputs: tuple[Any, ...],
    metric: dict[str, Any],
) -> tuple[str, ...]:
    render_ids_by_answer_output = _aggregate_render_ids_by_answer_output(
        payload=payload,
        bound=bound,
        relation_outputs=relation_outputs,
        metric=metric,
    )
    output: list[str] = []
    for answer_output_id in _required_strings(
        payload.get("answer_output_ids"),
        "answer_output_ids",
    ):
        render_output_id = render_ids_by_answer_output.get(answer_output_id, "")
        if not render_output_id:
            raise ValueError(
                "aggregate pattern missing render output for answer output"
            )
        output.append(render_output_id)
    return tuple(output)


def _aggregate_render_ids_by_answer_output(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    relation_outputs: tuple[Any, ...],
    metric: dict[str, Any],
) -> dict[str, str]:
    render_id_by_field_id = {
        item.field_id: item.id for item in relation_outputs if item.field_id
    }
    field_ids_by_answer_output = _render_value_field_ids_by_answer_output(
        payload=payload,
        bound=bound,
    )
    output: dict[str, str] = {}
    for answer_output_id, field_ids in field_ids_by_answer_output.items():
        render_output_id = _render_output_id_for_answer_output_field(
            field_ids=field_ids,
            render_id_by_field_id=render_id_by_field_id,
        )
        if render_output_id:
            output[answer_output_id] = render_output_id
    for answer_output_id, render_output_id in _metric_render_ids_by_answer_output(
        relation_outputs=relation_outputs,
        metric=metric,
    ).items():
        output.setdefault(answer_output_id, render_output_id)
    return output


def _render_output_id_for_answer_output_field(
    *,
    field_ids: tuple[str, ...],
    render_id_by_field_id: dict[str, str],
) -> str:
    for field_id in field_ids:
        render_output_id = render_id_by_field_id.get(field_id)
        if render_output_id:
            return render_output_id
    return ""
