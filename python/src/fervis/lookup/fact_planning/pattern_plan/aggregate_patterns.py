"""Aggregate pattern compilers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.fact_planning.grouped_aggregate_choices import (
    GroupedAggregateSelection,
    selected_grouped_aggregate_operation,
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    SourceKind,
)
from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.fact_planning.metric_options import metric_for_selection
from fervis.lookup.source_binding import BoundSource
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.expressions import (
    ExpressionFunction,
    FieldRef,
    FunctionExpression,
)
from fervis.lookup.answer_program.operations import (
    NamedExpression,
    Operation,
    ProjectSpec,
)
from fervis.lookup.answer_program.values import ANCHOR_TIMEZONE_REF, EnvironmentRef
from fervis.lookup.fact_planning.provider_contract import (
    AggregateScalarAnswerOutput,
    GroupedAggregateAnswerOutput,
)
from fervis.lookup.fact_planning.executable_support import RowPopulationBasis

from .aggregate_operations import (
    _aggregate_operations,
    _ordered_aggregate_operations,
)
from .aggregate_outputs import (
    _aggregate_relation_outputs,
    _grouped_aggregate_relation_outputs,
)
from .shared import (
    RelationBuilder,
    _bound_source,
    _compiled_pattern,
    _field_spec,
    _pattern_output_relation_id,
    _pattern_relation_id,
    _relations_for_bound_source,
    _relation_fields,
    _result_value_field_ids_by_answer_output,
    _validate_metric_source_compatibility,
    _validate_relation_fields_for_bound,
)
from fervis.lookup.fact_planning.compiled_patterns import (
    CompiledMetric,
    CompiledPattern,
    CompiledOrdering,
    PatternAddress,
)
from fervis.lookup.question_contract import GroupKeySourceKind, RequestedFact


@dataclass(frozen=True)
class _ScalarAggregateCompilation:
    relations: tuple[Relation, ...]
    operations: tuple[Operation, ...]
    bound: BoundSource
    aggregate_operation_id: str


def _compile_aggregate_pattern_answer(
    *,
    index: int,
    answer: AggregateScalarAnswerOutput | GroupedAggregateAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
    input_context: CompilerInputContext,
    requested_fact: RequestedFact,
) -> CompiledPattern:
    match answer:
        case GroupedAggregateAnswerOutput():
            return _compile_grouped_aggregate_answer(
                index=index,
                answer=answer,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
                input_context=input_context,
                requested_fact=requested_fact,
            )
        case AggregateScalarAnswerOutput():
            pass
    address = PatternAddress(
        requested_fact_id=answer.requested_fact_id,
        answer_output_ids=answer.answer_output_ids,
        plan_shape=answer.pattern,
        source_binding_id=answer.source_binding_id,
    )
    relation_id = _pattern_relation_id(index)
    output_relation_id = _pattern_output_relation_id(index)
    group_fields: tuple[dict[str, str], ...] = ()
    metric = metric_for_selection(answer=answer, bound_sources=bound_sources)
    compiled_source = _compile_scalar_aggregate_source(
        address=address,
        metric=metric,
        relation_id=relation_id,
        output_relation_id=output_relation_id,
        bound_sources=bound_sources,
        relation_builder=relation_builder,
    )
    relation_outputs = _aggregate_relation_outputs(
        index=index,
        output_relation_id=output_relation_id,
        group_fields=group_fields,
        metric=metric,
        namespace_result_outputs=namespace_result_outputs,
    )
    return CompiledPattern(
        fulfillment=tuple(
            FactFulfillment(
                requested_fact_id=address.requested_fact_id,
                answer_output_id=answer_output_id,
                result_output_id=result_output_id,
            )
            for answer_output_id, result_output_id in zip(
                address.answer_output_ids,
                _aggregate_fulfillment_result_ids(
                    address=address,
                    bound=compiled_source.bound,
                    relation_outputs=relation_outputs,
                    metric=metric,
                ),
                strict=True,
            )
        ),
        relations=compiled_source.relations,
        operations=compiled_source.operations,
        relation_outputs=relation_outputs,
        scalar_outputs=(),
    )


def _compile_scalar_aggregate_source(
    *,
    address: PatternAddress,
    metric: CompiledMetric,
    relation_id: str,
    output_relation_id: str,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
) -> _ScalarAggregateCompilation:
    """Lower one selected source metric into one scalar relation row."""

    bound = _bound_source(address.source_binding_id, bound_sources=bound_sources)
    row_population_basis = metric.row_population_basis
    if row_population_basis and bound.source is not None:
        bound = _bound_source_with_row_population_basis(
            bound,
            row_population_basis=row_population_basis,
        )
    effective_sources = {**bound_sources, bound.id: bound}
    _validate_metric_source_compatibility(
        address=address,
        metric=metric,
        bound_sources=effective_sources,
    )
    relation_fields = (
        RelationField(
            field_id=metric.field_id,
            roles=(FieldBindingRole.OUTPUT,),
        ),
    ) if metric.field_id else ()
    _validate_relation_fields_for_bound(
        address=address,
        bound=bound,
        relation_fields=relation_fields,
        selected_metric=metric,
    )
    source_inputs = _relations_for_bound_source(
        relation_id=relation_id,
        address=address,
        relation_fields=relation_fields,
        bound_sources=effective_sources,
        relation_builder=relation_builder,
        selected_metric=metric,
    )
    aggregate_operations = _aggregate_operations(
        input_relation_id=relation_id,
        output_relation_id=output_relation_id,
        group_fields=(),
        metric=metric,
    )
    return _ScalarAggregateCompilation(
        relations=source_inputs["relations"],
        operations=(*source_inputs["operations"], *aggregate_operations),
        bound=bound,
        aggregate_operation_id=aggregate_operations[-1].id,
    )


def _bound_source_with_row_population_basis(
    bound: BoundSource,
    *,
    row_population_basis: RowPopulationBasis,
) -> BoundSource:
    if bound.source is None or bound.source.kind != SourceKind.API_READ:
        return bound
    row_source_id = row_population_basis.row_source_id
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


def _compile_grouped_aggregate_answer(
    *,
    index: int,
    answer: GroupedAggregateAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
    input_context: CompilerInputContext,
    requested_fact: RequestedFact,
) -> CompiledPattern:
    relation_id = _pattern_relation_id(index)
    output_relation_id = _pattern_output_relation_id(index)
    ordering = CompiledOrdering.from_requested_fact(
        requested_fact,
        input_context=input_context,
    )
    aggregate_relation_id = (
        f"{output_relation_id}_aggregate"
        if ordering is not None
        else output_relation_id
    )
    selection = selected_grouped_aggregate_operation(
        answer,
        bound_sources=bound_sources,
        requested_fact=requested_fact,
    )
    address = PatternAddress(
        requested_fact_id=answer.requested_fact_id,
        answer_output_ids=selection.fulfills_answer_output_ids,
        plan_shape=answer.pattern,
        source_binding_id=selection.source_binding_id,
    )
    bound = bound_sources[selection.source_binding_id]
    metric = selection.metric
    row_population_basis = metric.row_population_basis
    if row_population_basis and bound.source is not None:
        bound = _bound_source_with_row_population_basis(
            bound,
            row_population_basis=row_population_basis,
        )
        bound_sources = {**bound_sources, bound.id: bound}
    source_group_fields = _grouped_aggregate_group_fields(selection)
    derived_group = _derived_group_key(
        selection=selection,
        requested_fact=requested_fact,
        input_context=input_context,
    )
    result_selection = (
        _selection_with_derived_group(selection, field_id=derived_group[0])
        if derived_group is not None
        else selection
    )
    group_fields = _grouped_aggregate_group_fields(result_selection)
    relation_fields = _grouped_aggregate_relation_fields(
        group_fields=source_group_fields,
        metric=metric,
    )
    _validate_relation_fields_for_bound(
        address=address,
        bound=bound,
        relation_fields=relation_fields,
        required_answer_evidence_ids_by_output=(
            _grouped_aggregate_answer_evidence_ids_by_output(result_selection)
        ),
        selected_metric=metric,
    )
    relation_outputs = _grouped_aggregate_relation_outputs(
        index=index,
        output_relation_id=output_relation_id,
        answer_outputs=result_selection.answer_outputs,
        metric=metric,
        namespace_result_outputs=namespace_result_outputs,
    )
    return _compiled_pattern(
        address=address,
        relation_id=relation_id,
        relation_fields=relation_fields,
        operations=(
            *(
                _derived_group_projection_operations(
                    input_relation_id=relation_id,
                    output_relation_id=f"{relation_id}_grouped",
                    field_id=derived_group[0],
                    expression=derived_group[1],
                    metric=metric,
                )
                if derived_group is not None
                else ()
            ),
            *(
                _ordered_aggregate_operations(
                    input_relation_id=(
                        f"{relation_id}_grouped"
                        if derived_group is not None
                        else relation_id
                    ),
                    aggregate_relation_id=aggregate_relation_id,
                    output_relation_id=output_relation_id,
                    order_operation_id=f"{output_relation_id}_order",
                    group_fields=group_fields,
                    metric=metric,
                    ordering=ordering,
                    ordering_field_id=(
                        answer.ordering_field.field_id
                        if answer.ordering_field is not None
                        else metric.output_field_id
                    ),
                    required_group_fields=_answer_result_group_fields(
                        selection=result_selection,
                        group_fields=group_fields,
                    ),
                )
                if ordering is not None
                else _aggregate_operations(
                    input_relation_id=(
                        f"{relation_id}_grouped"
                        if derived_group is not None
                        else relation_id
                    ),
                    output_relation_id=output_relation_id,
                    group_fields=group_fields,
                    metric=metric,
                    required_group_fields=_answer_result_group_fields(
                        selection=result_selection,
                        group_fields=group_fields,
                    ),
                )
            ),
        ),
        relation_outputs=relation_outputs,
        fulfillment_result_ids=_grouped_aggregate_fulfillment_result_ids(
            selection=result_selection,
            relation_outputs=relation_outputs,
        ),
        bound_sources=bound_sources,
        relation_builder=relation_builder,
        required_answer_evidence_ids_by_output=(
            _grouped_aggregate_answer_evidence_ids_by_output(result_selection)
        ),
        selected_metric=metric,
    )


def _derived_group_key(
    *,
    selection: GroupedAggregateSelection,
    requested_fact: RequestedFact,
    input_context: CompilerInputContext,
) -> tuple[str, FunctionExpression] | None:
    source_field_id = selection.group_key_source_field_id
    expression = requested_fact.answer_expression
    group_key = expression.group_key if expression is not None else None
    if not source_field_id:
        return None
    if (
        group_key is None
        or group_key.source_kind is not GroupKeySourceKind.TEMPORAL_BUCKET
        or not group_key.temporal_grain
    ):
        raise ValueError("derived group key requires a temporal bucket")
    return (
        group_key.id,
        FunctionExpression(
            function=ExpressionFunction.TEMPORAL_BUCKET,
            arguments=(
                FieldRef(source_field_id),
                input_context.expression_for_value(
                    group_key.temporal_grain_value_id(
                        requested_fact_id=requested_fact.id,
                    )
                ),
                EnvironmentRef(key=ANCHOR_TIMEZONE_REF),
            ),
        ),
    )


def _selection_with_derived_group(
    selection: GroupedAggregateSelection,
    *,
    field_id: str,
) -> GroupedAggregateSelection:
    return replace(
        selection,
        group_field_ids=(field_id,),
        group_entity_key_id="",
        group_entity_kind="",
        group_entity_components=(),
        answer_outputs=tuple(
            replace(
                output,
                field_ids=(field_id,),
                key_id="",
                entity_kind="",
                entity_components=(),
            )
            if output.role == "GROUP_KEY"
            else output
            for output in selection.answer_outputs
        ),
    )


def _derived_group_projection_operations(
    *,
    input_relation_id: str,
    output_relation_id: str,
    field_id: str,
    expression: FunctionExpression,
    metric: CompiledMetric,
) -> tuple[Operation, ...]:
    return (
        Operation(
            id=f"{output_relation_id}_project",
            spec=ProjectSpec(
                input_relation=input_relation_id,
                outputs=(
                    NamedExpression(output_field=field_id, expression=expression),
                    *(
                        (
                            NamedExpression(
                                output_field=metric.field_id,
                                expression=FieldRef(metric.field_id),
                            ),
                        )
                        if metric.field_id
                        else ()
                    ),
                ),
            ),
            output_relation=output_relation_id,
        ),
    )


def _grouped_aggregate_group_fields(
    selection: GroupedAggregateSelection,
) -> tuple[dict[str, str], ...]:
    return tuple(
        _field_spec({"field_id": field_id}) for field_id in selection.group_field_ids
    )


def _grouped_aggregate_relation_fields(
    *,
    group_fields: tuple[dict[str, str], ...],
    metric: CompiledMetric,
) -> tuple[RelationField, ...]:
    return (
        *_relation_fields(group_fields),
        *(
            (
                RelationField(
                    field_id=metric.field_id,
                    roles=(FieldBindingRole.OUTPUT,),
                ),
            )
            if metric.field_id
            else ()
        ),
    )


def _grouped_aggregate_fulfillment_result_ids(
    *,
    selection: GroupedAggregateSelection,
    relation_outputs: tuple[Any, ...],
) -> tuple[str, ...]:
    result_id_by_answer_output: dict[str, str] = {}
    for answer_output in selection.answer_outputs:
        render_id = next(
            (
                item.id
                for item in relation_outputs
                if (
                    answer_output.key_id
                    and item.entity_key is not None
                    and item.entity_key.key_id == answer_output.key_id
                    and item.entity_key.entity_kind == answer_output.entity_kind
                    and tuple(
                        (component.component_id, component.field_id)
                        for component in item.entity_key.components
                    )
                    == answer_output.entity_components
                )
                or (
                    not answer_output.key_id
                    and item.field_id in answer_output.field_ids
                )
            ),
            "",
        )
        if not render_id:
            raise ValueError("operation support missing result output field")
        result_id_by_answer_output.setdefault(answer_output.answer_output_id, render_id)
    output: list[str] = []
    for answer_output_id in selection.fulfills_answer_output_ids:
        render_id = result_id_by_answer_output.get(answer_output_id, "")
        if not render_id:
            raise ValueError("operation support missing answer output render")
        output.append(render_id)
    return tuple(output)


def _grouped_aggregate_answer_evidence_ids_by_output(
    selection: GroupedAggregateSelection,
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for answer_output in selection.answer_outputs:
        if not answer_output.evidence_ids:
            continue
        output.setdefault(answer_output.answer_output_id, ())
        output[answer_output.answer_output_id] = (
            *output[answer_output.answer_output_id],
            *answer_output.evidence_ids,
        )
    return output


def _answer_result_group_fields(
    *,
    selection: GroupedAggregateSelection,
    group_fields: tuple[dict[str, str], ...],
) -> tuple[str, ...]:
    group_field_ids = {item["field_id"] for item in group_fields}
    return tuple(
        dict.fromkeys(
            field_id
            for answer in selection.answer_outputs
            if answer.role == "GROUP_KEY"
            for field_id in answer.field_ids
            if field_id in group_field_ids
        )
    )


def _metric_result_ids_by_answer_output(
    *,
    relation_outputs: tuple[Any, ...],
    metric: CompiledMetric,
) -> dict[str, str]:
    answer_output_id = metric.answer_output_id
    if not answer_output_id:
        return {}
    for output in relation_outputs:
        if output.field_id == metric.output_field_id:
            return {answer_output_id: output.id}
    return {}


def _aggregate_fulfillment_result_ids(
    *,
    address: PatternAddress,
    bound: BoundSource,
    relation_outputs: tuple[Any, ...],
    metric: CompiledMetric,
) -> tuple[str, ...]:
    result_ids_by_answer_output = _aggregate_result_ids_by_answer_output(
        address=address,
        bound=bound,
        relation_outputs=relation_outputs,
        metric=metric,
    )
    output: list[str] = []
    for answer_output_id in address.answer_output_ids:
        result_output_id = result_ids_by_answer_output.get(answer_output_id, "")
        if not result_output_id:
            raise ValueError(
                "aggregate pattern missing result output for answer output"
            )
        output.append(result_output_id)
    return tuple(output)


def _aggregate_result_ids_by_answer_output(
    *,
    address: PatternAddress,
    bound: BoundSource,
    relation_outputs: tuple[Any, ...],
    metric: CompiledMetric,
) -> dict[str, str]:
    result_id_by_field_id = {
        item.field_id: item.id for item in relation_outputs if item.field_id
    }
    field_ids_by_answer_output = _result_value_field_ids_by_answer_output(
        address=address,
        bound=bound,
    )
    output: dict[str, str] = {}
    for answer_output_id, field_ids in field_ids_by_answer_output.items():
        result_output_id = _result_output_id_for_answer_output_field(
            field_ids=field_ids,
            result_id_by_field_id=result_id_by_field_id,
        )
        if result_output_id:
            output[answer_output_id] = result_output_id
    for answer_output_id, result_output_id in _metric_result_ids_by_answer_output(
        relation_outputs=relation_outputs,
        metric=metric,
    ).items():
        output.setdefault(answer_output_id, result_output_id)
    return output


def _result_output_id_for_answer_output_field(
    *,
    field_ids: tuple[str, ...],
    result_id_by_field_id: dict[str, str],
) -> str:
    for field_id in field_ids:
        result_output_id = result_id_by_field_id.get(field_id)
        if result_output_id:
            return result_output_id
    return ""
