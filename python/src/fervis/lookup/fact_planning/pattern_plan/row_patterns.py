"""Row and field-value pattern compilers."""

from __future__ import annotations

from fervis.lookup.answer_program.operations import (
    Operation,
    OrderSpec,
    NamedExpression,
    ProjectSpec,
    SortKey,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.result_projection import (
    EntityKeyProjection,
    EntityKeyProjectionComponent,
    RelationResultOutput,
)
from fervis.lookup.source_binding import (
    BoundSource,
    SourceFulfillment,
    entity_evidence_entity_kind,
    entity_evidence_key_id,
)
from fervis.lookup.fact_planning.provider_contract import (
    DirectFieldValueAnswerOutput,
    GroupedRowsAnswerOutput,
    ListRowsAnswerOutput,
)

from .shared import (
    RelationBuilder,
    _answer_value_field_ids_by_answer_output,
    _bound_source,
    _compiled_pattern,
    _field_spec,
    _pattern_output_relation_id,
    _pattern_relation_id,
    _relation_fields,
)
from fervis.lookup.fact_planning.compiled_patterns import (
    CompiledOrdering,
    CompiledPattern,
    PatternAddress,
)
from fervis.lookup.question_contract import RequestedFact
from .result_ids import _result_output_id


def _compile_project_pattern_answer(
    *,
    index: int,
    answer: ListRowsAnswerOutput | GroupedRowsAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
    input_context: CompilerInputContext,
    requested_fact: RequestedFact,
) -> CompiledPattern:
    address = PatternAddress(
        requested_fact_id=answer.requested_fact_id,
        answer_output_ids=answer.answer_output_ids,
        plan_shape=answer.pattern,
        source_binding_id=answer.source_binding_id,
    )
    order_field = None
    ordering = CompiledOrdering.from_requested_fact(
        requested_fact,
        input_context=input_context,
    )
    match answer:
        case GroupedRowsAnswerOutput():
            group_fields = tuple(
                _field_spec({"field_id": field.field_id})
                for field in answer.group_fields
            )
        case ListRowsAnswerOutput():
            group_fields = ()
    if ordering is not None:
        ordering_field = getattr(answer, "ordering_field", None)
        if ordering_field is None:
            raise ValueError("ordered row answer requires ordering field")
        order_field = _field_spec({"field_id": ordering_field.field_id})
    output_fields = tuple(
        _field_spec({"field_id": field.field_id}) for field in answer.output_fields
    )
    return _compile_project_fields(
        index=index,
        address=address,
        group_fields=group_fields,
        output_fields=output_fields,
        namespace_result_outputs=namespace_result_outputs,
        bound_sources=bound_sources,
        relation_builder=relation_builder,
        order_field=order_field,
        ordering=ordering,
        input_context=input_context,
    )


def _compile_project_fields(
    *,
    index: int,
    address: PatternAddress,
    group_fields: tuple[dict[str, str], ...],
    output_fields: tuple[dict[str, str], ...],
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
    order_field: dict[str, str] | None = None,
    ordering: CompiledOrdering | None = None,
    input_context: CompilerInputContext,
) -> CompiledPattern:
    relation_id = _pattern_relation_id(index)
    output_relation_id = _pattern_output_relation_id(index)
    bound = _bound_source(address.source_binding_id, bound_sources=bound_sources)
    output_fields = _without_existing_fields(
        output_fields,
        existing_field_ids={item["field_id"] for item in group_fields},
    )
    if not output_fields and not _group_fields_cover_answer_value_fields(
        address=address,
        bound=bound,
        group_fields=group_fields,
    ):
        raise ValueError("row pattern requires output_fields")
    output_field_result_pairs = _field_result_output_pairs(
        index=index,
        answer_output_ids=address.answer_output_ids,
        fields=output_fields,
        namespace_result_outputs=namespace_result_outputs,
    )
    group_field_result_pairs = _field_result_output_pairs(
        index=index,
        answer_output_ids=address.answer_output_ids,
        fields=group_fields,
        namespace_result_outputs=namespace_result_outputs,
        offset=len(output_fields),
    )
    project_fields = tuple(
        NamedExpression(
            output_field=item["output_field_id"],
            expression=FieldRef(item["field_id"]),
        )
        for item in (*group_fields, *output_fields)
    )
    result_pairs = (*output_field_result_pairs, *group_field_result_pairs)
    field_ids_by_answer_output = _answer_value_field_ids_by_answer_output(
        address=address,
        bound=bound,
    )
    result_outputs = _relation_result_outputs(
        index=index,
        requested_fact_id=address.requested_fact_id,
        answer_output_ids=address.answer_output_ids,
        field_ids_by_answer_output=field_ids_by_answer_output,
        bound=bound,
        output_relation_id=output_relation_id,
        result_pairs=result_pairs,
        namespace_result_outputs=namespace_result_outputs,
    )
    fulfillment_result_ids = _fulfillment_result_ids(
        index=index,
        answer_output_ids=address.answer_output_ids,
        namespace_result_outputs=namespace_result_outputs,
    )
    operations = _row_operations(
        relation_id=relation_id,
        output_relation_id=output_relation_id,
        project_fields=project_fields,
        order_field=order_field,
        ordering=ordering,
    )
    support_fields = tuple(
        item
        for item in ((order_field,) if order_field is not None else ())
        if item not in (*group_fields, *output_fields)
    )
    compiled = _compiled_pattern(
        address=address,
        relation_id=relation_id,
        relation_fields=(
            *_relation_fields(group_fields, bound_source=bound),
            *_relation_fields(output_fields, identity=False, bound_source=bound),
            *_relation_fields(support_fields, bound_source=bound),
        ),
        operations=operations,
        relation_outputs=result_outputs,
        fulfillment_result_ids=fulfillment_result_ids,
        bound_sources=bound_sources,
        relation_builder=relation_builder,
    )
    return compiled


def _row_operations(
    *,
    relation_id: str,
    output_relation_id: str,
    project_fields: tuple[NamedExpression, ...],
    order_field: dict[str, str] | None,
    ordering: CompiledOrdering | None,
) -> tuple[Operation, ...]:
    project_input_relation = relation_id
    operations: list[Operation] = []
    if ordering is not None and order_field is not None:
        project_input_relation = f"{output_relation_id}_ordered"
        operations.append(
            Operation(
                id=f"{output_relation_id}_order",
                spec=OrderSpec(
                    input_relation=relation_id,
                    order_by=(
                        SortKey(
                            field=order_field["field_id"],
                            direction=ordering.direction,
                        ),
                    ),
                    selection=ordering.selection,
                ),
                output_relation=project_input_relation,
            )
        )
    operations.append(
        Operation(
            id=f"{output_relation_id}_project",
            spec=ProjectSpec(
                input_relation=project_input_relation,
                outputs=project_fields,
            ),
            output_relation=output_relation_id,
        )
    )
    return tuple(operations)


def _relation_result_outputs(
    *,
    index: int,
    requested_fact_id: str,
    answer_output_ids: tuple[str, ...],
    field_ids_by_answer_output: dict[str, tuple[str, ...]],
    bound: BoundSource,
    output_relation_id: str,
    result_pairs: tuple[tuple[dict[str, str], str], ...],
    namespace_result_outputs: bool,
) -> tuple[RelationResultOutput, ...]:
    fulfillment_by_answer_output = {
        fulfillment.answer_output_id: fulfillment
        for fulfillment in bound.fulfillments
        if fulfillment.requested_fact_id == requested_fact_id
    }
    output_field_id_by_source = {
        item["field_id"]: item["output_field_id"] for item, _ in result_pairs
    }
    answer_outputs, consumed_field_ids = _answer_result_outputs(
        index=index,
        answer_output_ids=answer_output_ids,
        fulfillment_by_answer_output=fulfillment_by_answer_output,
        field_ids_by_answer_output=field_ids_by_answer_output,
        output_field_id_by_source=output_field_id_by_source,
        output_relation_id=output_relation_id,
        namespace_result_outputs=namespace_result_outputs,
    )
    support_outputs = _support_result_outputs(
        index=index,
        result_pairs=result_pairs,
        consumed_field_ids=consumed_field_ids,
        reserved_output_ids={output.id for output in answer_outputs},
        output_relation_id=output_relation_id,
        namespace_result_outputs=namespace_result_outputs,
    )
    return (*answer_outputs, *support_outputs)


def _answer_result_outputs(
    *,
    index: int,
    answer_output_ids: tuple[str, ...],
    fulfillment_by_answer_output: dict[str, SourceFulfillment],
    field_ids_by_answer_output: dict[str, tuple[str, ...]],
    output_field_id_by_source: dict[str, str],
    output_relation_id: str,
    namespace_result_outputs: bool,
) -> tuple[tuple[RelationResultOutput, ...], frozenset[str]]:
    outputs: list[RelationResultOutput] = []
    consumed_field_ids: set[str] = set()
    for answer_output_id in answer_output_ids:
        fulfillment = fulfillment_by_answer_output.get(answer_output_id)
        source_field_ids = field_ids_by_answer_output.get(answer_output_id, ())
        result_output = _answer_result_output(
            index=index,
            answer_output_id=answer_output_id,
            fulfillment=fulfillment,
            source_field_ids=source_field_ids,
            output_field_id_by_source=output_field_id_by_source,
            output_relation_id=output_relation_id,
            namespace_result_outputs=namespace_result_outputs,
        )
        outputs.append(result_output)
        consumed_field_ids.update(source_field_ids)
    return tuple(outputs), frozenset(consumed_field_ids)


def _answer_result_output(
    *,
    index: int,
    answer_output_id: str,
    fulfillment: SourceFulfillment | None,
    source_field_ids: tuple[str, ...],
    output_field_id_by_source: dict[str, str],
    output_relation_id: str,
    namespace_result_outputs: bool,
) -> RelationResultOutput:
    projected_field_ids = _projected_field_ids(
        source_field_ids,
        output_field_id_by_source=output_field_id_by_source,
    )
    if fulfillment is None or not projected_field_ids:
        raise ValueError("row pattern missing selected answer evidence")
    result_output_id = _result_output_id(
        index,
        answer_output_id,
        namespace_result_outputs=namespace_result_outputs,
    )
    label = " ".join(projected_field_ids) if namespace_result_outputs else ""
    entity_key = _answer_entity_key(
        fulfillment,
        output_field_id_by_source=output_field_id_by_source,
    )
    field_id = projected_field_ids[0] if entity_key is None else ""
    return RelationResultOutput(
        id=result_output_id,
        relation_id=output_relation_id,
        field_id=field_id,
        entity_key=entity_key,
        label=label,
        role="answer_value",
    )


def _answer_entity_key(
    fulfillment: SourceFulfillment,
    *,
    output_field_id_by_source: dict[str, str],
) -> EntityKeyProjection | None:
    evidence = fulfillment.entity_evidence
    if evidence is None:
        return None
    components = tuple(
        EntityKeyProjectionComponent(
            component_id=component.component_id,
            field_id=output_field_id_by_source[component.field_id],
        )
        for component in evidence.components
    )
    return EntityKeyProjection(
        entity_kind=entity_evidence_entity_kind(evidence),
        key_id=entity_evidence_key_id(evidence),
        components=components,
    )


def _support_result_outputs(
    *,
    index: int,
    result_pairs: tuple[tuple[dict[str, str], str], ...],
    consumed_field_ids: frozenset[str],
    reserved_output_ids: set[str],
    output_relation_id: str,
    namespace_result_outputs: bool,
) -> tuple[RelationResultOutput, ...]:
    outputs: list[RelationResultOutput] = []
    for item, fallback_output_id in result_pairs:
        if item["field_id"] in consumed_field_ids:
            continue
        output_id = _support_result_output_id(
            index=index,
            item=item,
            fallback_output_id=fallback_output_id,
            reserved_output_ids=reserved_output_ids,
            namespace_result_outputs=namespace_result_outputs,
        )
        label = item["label"] if namespace_result_outputs else ""
        output = RelationResultOutput(
            id=output_id,
            relation_id=output_relation_id,
            field_id=item["output_field_id"],
            label=label,
            role="support",
        )
        outputs.append(output)
        reserved_output_ids.add(output_id)
    return tuple(outputs)


def _support_result_output_id(
    *,
    index: int,
    item: dict[str, str],
    fallback_output_id: str,
    reserved_output_ids: set[str],
    namespace_result_outputs: bool,
) -> str:
    if fallback_output_id not in reserved_output_ids:
        return fallback_output_id
    return _result_output_id(
        index,
        f"support_{item['output_field_id']}",
        namespace_result_outputs=namespace_result_outputs,
    )


def _fulfillment_result_ids(
    *,
    index: int,
    answer_output_ids: tuple[str, ...],
    namespace_result_outputs: bool,
) -> tuple[str, ...]:
    return tuple(
        _result_output_id(
            index,
            answer_output_id,
            namespace_result_outputs=namespace_result_outputs,
        )
        for answer_output_id in answer_output_ids
    )


def _projected_field_ids(
    source_field_ids: tuple[str, ...],
    *,
    output_field_id_by_source: dict[str, str],
) -> tuple[str, ...]:
    missing = tuple(
        field_id
        for field_id in source_field_ids
        if field_id not in output_field_id_by_source
    )
    if missing:
        raise ValueError("row pattern does not project selected answer evidence")
    return tuple(output_field_id_by_source[field_id] for field_id in source_field_ids)


def _compile_direct_field_value_answer(
    *,
    index: int,
    answer: DirectFieldValueAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
    input_context: CompilerInputContext,
) -> CompiledPattern:
    field = _field_spec({"field_id": answer.output_field.field_id})
    address = PatternAddress(
        requested_fact_id=answer.requested_fact_id,
        answer_output_ids=answer.answer_output_ids,
        plan_shape=answer.pattern,
        source_binding_id=answer.source_binding_id,
    )
    return _compile_project_fields(
        index=index,
        address=address,
        group_fields=(),
        output_fields=(field,),
        namespace_result_outputs=namespace_result_outputs,
        bound_sources=bound_sources,
        relation_builder=relation_builder,
        input_context=input_context,
    )


def _field_result_output_pairs(
    *,
    index: int,
    answer_output_ids: tuple[str, ...],
    fields: tuple[dict[str, str], ...],
    namespace_result_outputs: bool,
    offset: int = 0,
) -> tuple[tuple[dict[str, str], str], ...]:
    output: list[tuple[dict[str, str], str]] = []
    for field_index, item in enumerate(fields):
        answer_index = offset + field_index
        output_id = (
            answer_output_ids[answer_index]
            if answer_index < len(answer_output_ids)
            else item["output_field_id"]
        )
        output.append(
            (
                item,
                _result_output_id(
                    index,
                    output_id,
                    namespace_result_outputs=namespace_result_outputs,
                ),
            )
        )
    return tuple(output)


def _group_fields_cover_answer_value_fields(
    *,
    address: PatternAddress,
    bound: BoundSource,
    group_fields: tuple[dict[str, str], ...],
) -> bool:
    group_field_ids = {item["field_id"] for item in group_fields}
    required_field_ids = {
        field_id
        for field_ids in _answer_value_field_ids_by_answer_output(
            address=address,
            bound=bound,
        ).values()
        for field_id in field_ids
    }
    return bool(required_field_ids and required_field_ids <= group_field_ids)


def _without_existing_fields(
    fields: tuple[dict[str, str], ...],
    *,
    existing_field_ids: set[str],
) -> tuple[dict[str, str], ...]:
    output: list[dict[str, str]] = []
    seen = set(existing_field_ids)
    for field in fields:
        field_id = field["field_id"]
        if field_id in seen:
            continue
        seen.add(field_id)
        output.append(field)
    return tuple(output)
