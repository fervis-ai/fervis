"""Canonical result-output builders for aggregate pattern compilers."""

from fervis.lookup.answer_program.result_projection import (
    EntityKeyProjection,
    EntityKeyProjectionComponent,
    RelationResultOutput,
)
from fervis.lookup.fact_planning.compiled_patterns import CompiledMetric
from fervis.lookup.fact_planning.grouped_ranked_choices import (
    GroupedRankedAnswerOutput,
)

from .result_ids import _result_output_id


def _aggregate_relation_outputs(
    *,
    index: int,
    output_relation_id: str,
    group_fields: tuple[dict[str, str], ...],
    metric: CompiledMetric,
    namespace_result_outputs: bool,
) -> tuple[RelationResultOutput, ...]:
    return tuple(
        RelationResultOutput(
            id=_result_output_id(
                index,
                item["output_field_id"],
                namespace_result_outputs=namespace_result_outputs,
            ),
            relation_id=output_relation_id,
            field_id=item.get("result_field_id", item["output_field_id"]),
            label=item["label"] if namespace_result_outputs else "",
            role=item["role"],
        )
        for item in (
            *(
                {
                    **item,
                    "result_field_id": item["field_id"],
                    "role": "support",
                }
                for item in group_fields
            ),
            {
                "output_field_id": metric.output_field_id,
                "label": metric.label,
                "role": "answer_value",
            },
        )
    )


def _grouped_ranked_relation_outputs(
    *,
    index: int,
    output_relation_id: str,
    answer_outputs: tuple[GroupedRankedAnswerOutput, ...],
    metric: CompiledMetric,
    namespace_result_outputs: bool,
) -> tuple[RelationResultOutput, ...]:
    relation_answer_outputs = _grouped_ranked_answer_result_outputs(
        index=index,
        output_relation_id=output_relation_id,
        answer_outputs=answer_outputs,
        namespace_result_outputs=namespace_result_outputs,
    )
    if not metric.output_field_id or any(
        item.field_id == metric.output_field_id for item in answer_outputs
    ):
        return relation_answer_outputs
    return (
        *relation_answer_outputs,
        _ranked_metric_result_output(
            index=index,
            output_relation_id=output_relation_id,
            metric=metric,
            metric_answer_output_id=(
                _grouped_ranked_metric_parent_output_id(
                    metric=metric,
                    answer_outputs=answer_outputs,
                )
            ),
            reserved_result_output_ids=tuple(
                item.id for item in relation_answer_outputs
            ),
            namespace_result_outputs=namespace_result_outputs,
        ),
    )


def _grouped_ranked_answer_result_outputs(
    *,
    index: int,
    output_relation_id: str,
    answer_outputs: tuple[GroupedRankedAnswerOutput, ...],
    namespace_result_outputs: bool,
) -> tuple[RelationResultOutput, ...]:
    reserved: set[str] = set()
    seen_answer_outputs: set[str] = set()
    output: list[RelationResultOutput] = []
    for item in answer_outputs:
        answer_output_id = item.answer_output_id
        field_ids = item.field_ids
        if answer_output_id in seen_answer_outputs:
            continue
        seen_answer_outputs.add(answer_output_id)
        result_output_id = _unique_result_output_id(
            index=index,
            output_id=answer_output_id,
            reserved=reserved,
            namespace_result_outputs=namespace_result_outputs,
        )
        reserved.add(result_output_id)
        result_output = _grouped_ranked_answer_result_output(
            item=item,
            result_output_id=result_output_id,
            output_relation_id=output_relation_id,
            field_ids=field_ids,
            namespace_result_outputs=namespace_result_outputs,
        )
        output.append(result_output)
    return tuple(output)


def _grouped_ranked_answer_result_output(
    *,
    item: GroupedRankedAnswerOutput,
    result_output_id: str,
    output_relation_id: str,
    field_ids: tuple[str, ...],
    namespace_result_outputs: bool,
) -> RelationResultOutput:
    key_id = item.key_id
    field_id = field_ids[0] if len(field_ids) == 1 and not key_id else ""
    entity_key = _grouped_ranked_entity_key(item, key_id=key_id)
    label = " ".join(field_ids) if namespace_result_outputs else ""
    return RelationResultOutput(
        id=result_output_id,
        relation_id=output_relation_id,
        field_id=field_id,
        entity_key=entity_key,
        label=label,
        role="answer_value",
    )


def _grouped_ranked_entity_key(
    item: GroupedRankedAnswerOutput,
    *,
    key_id: str,
) -> EntityKeyProjection | None:
    if not key_id:
        return None
    entity_components = item.entity_components
    components = tuple(
        EntityKeyProjectionComponent(
            component_id=component_id,
            field_id=field_id,
        )
        for component_id, field_id in entity_components
    )
    entity_kind = item.entity_kind
    return EntityKeyProjection(
        entity_kind=entity_kind,
        key_id=key_id,
        components=components,
    )


def _unique_result_output_id(
    *,
    index: int,
    output_id: str,
    reserved: set[str],
    namespace_result_outputs: bool,
) -> str:
    candidate = output_id
    result_output_id = _result_output_id(
        index,
        candidate,
        namespace_result_outputs=namespace_result_outputs,
    )
    suffix = 2
    while result_output_id in reserved:
        candidate = f"{output_id}_{suffix}"
        result_output_id = _result_output_id(
            index,
            candidate,
            namespace_result_outputs=namespace_result_outputs,
        )
        suffix += 1
    return result_output_id


def _grouped_ranked_metric_parent_output_id(
    *,
    metric: CompiledMetric,
    answer_outputs: tuple[GroupedRankedAnswerOutput, ...],
) -> str:
    return metric.answer_output_id or (
        answer_outputs[0].answer_output_id if answer_outputs else ""
    )


def _ranked_metric_result_output(
    *,
    index: int,
    output_relation_id: str,
    metric: CompiledMetric,
    metric_answer_output_id: str,
    reserved_result_output_ids: tuple[str, ...],
    namespace_result_outputs: bool,
) -> RelationResultOutput:
    output_id = metric_answer_output_id or "support_1"
    reserved = set(reserved_result_output_ids)
    result_output_id = _result_output_id(
        index,
        output_id,
        namespace_result_outputs=namespace_result_outputs,
    )
    if metric_answer_output_id and result_output_id in reserved:
        output_id = f"{metric_answer_output_id}.ranking_metric"
        result_output_id = _result_output_id(
            index,
            output_id,
            namespace_result_outputs=namespace_result_outputs,
        )
    suffix = 2
    while result_output_id in reserved:
        output_id = f"{metric_answer_output_id or 'support_1'}.ranking_metric_{suffix}"
        result_output_id = _result_output_id(
            index,
            output_id,
            namespace_result_outputs=namespace_result_outputs,
        )
        suffix += 1
    return RelationResultOutput(
        id=result_output_id,
        relation_id=output_relation_id,
        field_id=metric.output_field_id,
        label=metric.label if namespace_result_outputs else "",
        role=(
            "answer_value" if output_id == metric_answer_output_id else "ranking_metric"
        ),
    )
