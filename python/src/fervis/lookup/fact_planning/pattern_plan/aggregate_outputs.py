"""Render output builders for aggregate pattern compilers."""

from fervis.lookup.fact_plan.render_spec import RenderRelationOutput

from .render_ids import _render_output_id


def _aggregate_relation_outputs(
    *,
    index: int,
    output_relation_id: str,
    group_fields: tuple[dict[str, str], ...],
    carry_fields: tuple[dict[str, str], ...],
    metric: dict[str, str],
    namespace_render_outputs: bool,
) -> tuple[RenderRelationOutput, ...]:
    return tuple(
        RenderRelationOutput(
            id=_render_output_id(
                index,
                item["output_field_id"],
                namespace_render_outputs=namespace_render_outputs,
            ),
            relation_id=output_relation_id,
            field_id=item.get("render_field_id", item["output_field_id"]),
            label=item["label"] if namespace_render_outputs else "",
            role=item["role"],
        )
        for item in (
            *(
                {
                    **item,
                    "render_field_id": item["field_id"],
                    "role": "support",
                }
                for item in (*group_fields, *carry_fields)
            ),
            {
                "output_field_id": metric["output_field_id"],
                "label": metric["label"],
                "role": "answer_value",
            },
        )
    )


def _grouped_ranked_relation_outputs(
    *,
    index: int,
    output_relation_id: str,
    answer_outputs: tuple[object, ...],
    carry_fields: tuple[dict[str, str], ...],
    metric: dict[str, str],
    namespace_render_outputs: bool,
    answer_output_render_ids: bool = True,
) -> tuple[RenderRelationOutput, ...]:
    relation_answer_outputs = _grouped_ranked_answer_render_outputs(
        index=index,
        output_relation_id=output_relation_id,
        answer_outputs=answer_outputs,
        namespace_render_outputs=namespace_render_outputs,
        answer_output_render_ids=answer_output_render_ids,
    )
    carry_outputs = _ranked_carry_render_outputs(
        index=index,
        carry_fields=tuple(
            item
            for item in carry_fields
            if item["field_id"] not in {_field_id(output) for output in answer_outputs}
            and item["field_id"] != metric["output_field_id"]
        ),
        output_relation_id=output_relation_id,
        namespace_render_outputs=namespace_render_outputs,
    )
    if not metric["output_field_id"] or any(
        _field_id(item) == metric["output_field_id"] for item in answer_outputs
    ):
        return (*relation_answer_outputs, *carry_outputs)
    return (
        *relation_answer_outputs,
        *carry_outputs,
        _ranked_metric_render_output(
            index=index,
            output_relation_id=output_relation_id,
            metric=metric,
            metric_answer_output_id=(
                _grouped_ranked_metric_parent_output_id(
                    metric=metric,
                    answer_outputs=answer_outputs,
                )
                if answer_output_render_ids
                else metric["output_field_id"]
            ),
            reserved_render_output_ids=tuple(
                item.id for item in (*relation_answer_outputs, *carry_outputs)
            ),
            namespace_render_outputs=namespace_render_outputs,
        ),
    )


def _grouped_ranked_answer_render_outputs(
    *,
    index: int,
    output_relation_id: str,
    answer_outputs: tuple[object, ...],
    namespace_render_outputs: bool,
    answer_output_render_ids: bool,
) -> tuple[RenderRelationOutput, ...]:
    reserved: set[str] = set()
    seen_answer_fields: set[tuple[str, str]] = set()
    seen_fields: set[str] = set()
    output: list[RenderRelationOutput] = []
    for item in answer_outputs:
        answer_output_id = _answer_output_id(item)
        field_id = _field_id(item)
        answer_field = (answer_output_id, field_id)
        if field_id in seen_fields or answer_field in seen_answer_fields:
            continue
        seen_fields.add(field_id)
        seen_answer_fields.add(answer_field)
        render_output_id = _unique_render_output_id(
            index=index,
            output_id=answer_output_id if answer_output_render_ids else field_id,
            reserved=reserved,
            namespace_render_outputs=namespace_render_outputs,
        )
        reserved.add(render_output_id)
        output.append(
            RenderRelationOutput(
                id=render_output_id,
                relation_id=output_relation_id,
                field_id=field_id,
                label=_field_id(item) if namespace_render_outputs else "",
                role="answer_value",
            )
        )
    return tuple(output)


def _unique_render_output_id(
    *,
    index: int,
    output_id: str,
    reserved: set[str],
    namespace_render_outputs: bool,
) -> str:
    candidate = output_id
    render_output_id = _render_output_id(
        index,
        candidate,
        namespace_render_outputs=namespace_render_outputs,
    )
    suffix = 2
    while render_output_id in reserved:
        candidate = f"{output_id}_{suffix}"
        render_output_id = _render_output_id(
            index,
            candidate,
            namespace_render_outputs=namespace_render_outputs,
        )
        suffix += 1
    return render_output_id


def _answer_output_id(item: object) -> str:
    return str(getattr(item, "answer_output_id", "") or "")


def _field_id(item: object) -> str:
    return str(getattr(item, "field_id", "") or "")


def _grouped_ranked_metric_parent_output_id(
    *,
    metric: dict[str, str],
    answer_outputs: tuple[object, ...],
) -> str:
    return str(metric.get("answer_output_id") or "") or (
        _answer_output_id(answer_outputs[0]) if answer_outputs else ""
    )


def _ranked_metric_render_output(
    *,
    index: int,
    output_relation_id: str,
    metric: dict[str, str],
    metric_answer_output_id: str,
    reserved_render_output_ids: tuple[str, ...],
    namespace_render_outputs: bool,
) -> RenderRelationOutput:
    output_id = metric_answer_output_id or "support_1"
    reserved = set(reserved_render_output_ids)
    render_output_id = _render_output_id(
        index,
        output_id,
        namespace_render_outputs=namespace_render_outputs,
    )
    if metric_answer_output_id and render_output_id in reserved:
        output_id = f"{metric_answer_output_id}.ranking_metric"
        render_output_id = _render_output_id(
            index,
            output_id,
            namespace_render_outputs=namespace_render_outputs,
        )
    suffix = 2
    while render_output_id in reserved:
        output_id = f"{metric_answer_output_id or 'support_1'}.ranking_metric_{suffix}"
        render_output_id = _render_output_id(
            index,
            output_id,
            namespace_render_outputs=namespace_render_outputs,
        )
        suffix += 1
    return RenderRelationOutput(
        id=render_output_id,
        relation_id=output_relation_id,
        field_id=metric["output_field_id"],
        label=metric["label"] if namespace_render_outputs else "",
        role=(
            "answer_value" if output_id == metric_answer_output_id else "ranking_metric"
        ),
    )


def _ranked_carry_render_outputs(
    *,
    index: int,
    carry_fields: tuple[dict[str, str], ...],
    output_relation_id: str,
    namespace_render_outputs: bool,
) -> tuple[RenderRelationOutput, ...]:
    return tuple(
        RenderRelationOutput(
            id=_render_output_id(
                index,
                item["output_field_id"],
                namespace_render_outputs=namespace_render_outputs,
            ),
            relation_id=output_relation_id,
            field_id=item["field_id"],
            label=item["label"] if namespace_render_outputs else "",
            role="support",
        )
        for item in carry_fields
    )
