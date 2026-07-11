"""Row and field-value pattern compilers."""

from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.operations import (
    Operation,
    ProjectField,
    ProjectSpec,
)
from fervis.lookup.answer_program.render_spec import RenderRelationOutput
from fervis.lookup.source_binding import BoundSource

from .shared import (
    RelationBuilder,
    _answer_value_field_ids_by_answer_output,
    _bound_source,
    _compiled_pattern,
    _dict,
    _field_spec,
    _field_specs,
    _pattern_output_relation_id,
    _pattern_relation_id,
    _relation_fields,
    _required_strings,
)
from .render_ids import _render_output_id


def _compile_project_pattern_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
) -> dict[str, Any]:
    relation_id = _pattern_relation_id(index)
    output_relation_id = _pattern_output_relation_id(index)
    bound = _bound_source(payload, bound_sources=bound_sources)
    group_fields = _field_specs(payload.get("group_fields"))
    output_fields = _field_specs(payload.get("output_fields"))
    output_fields = _without_existing_fields(
        output_fields,
        existing_field_ids={item["field_id"] for item in group_fields},
    )
    if not output_fields and not _group_fields_cover_answer_value_fields(
        payload=payload,
        bound=bound,
        group_fields=group_fields,
    ):
        raise ValueError("row pattern requires output_fields")
    output_field_render_pairs = _field_render_output_pairs(
        index=index,
        payload=payload,
        fields=output_fields,
        namespace_render_outputs=namespace_render_outputs,
    )
    group_field_render_pairs = _field_render_output_pairs(
        index=index,
        payload=payload,
        fields=group_fields,
        namespace_render_outputs=namespace_render_outputs,
        offset=len(output_fields),
    )
    project_fields = tuple(
        ProjectField(source=item["field_id"], output=item["output_field_id"])
        for item in (*group_fields, *output_fields)
    )
    render_outputs = tuple(
        RenderRelationOutput(
            id=render_output_id,
            relation_id=output_relation_id,
            field_id=item["output_field_id"],
            label=item["label"] if namespace_render_outputs else "",
            role="answer_value",
        )
        for item, render_output_id in output_field_render_pairs
    ) + tuple(
        RenderRelationOutput(
            id=render_output_id,
            relation_id=output_relation_id,
            field_id=item["output_field_id"],
            label=item["label"] if namespace_render_outputs else "",
            role="support",
        )
        for item, render_output_id in group_field_render_pairs
    )
    render_pairs = (*output_field_render_pairs, *group_field_render_pairs)
    return _compiled_pattern(
        payload=payload,
        relation_id=relation_id,
        relation_fields=(
            *_relation_fields(group_fields, bound_source=bound),
            *_relation_fields(output_fields, identity=False, bound_source=bound),
        ),
        operations=(
            Operation(
                id=f"{output_relation_id}_project",
                spec=ProjectSpec(input_relation=relation_id, fields=project_fields),
                output_relation=output_relation_id,
            ),
        ),
        relation_outputs=render_outputs,
        fulfillment_render_ids=_fulfillment_render_ids_by_answer_value_field(
            payload=payload,
            bound=bound,
            render_pairs=render_pairs,
        ),
        bound_sources=bound_sources,
        relation_builder=relation_builder,
    )


def _compile_direct_field_value_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
    relation_builder: RelationBuilder,
) -> dict[str, Any]:
    field = _field_spec(_dict(payload.get("output_field"), "output_field"))
    return _compile_project_pattern_answer(
        index=index,
        namespace_render_outputs=namespace_render_outputs,
        payload={
            **payload,
            "output_fields": [
                {
                    "field_id": field["field_id"],
                }
            ],
        },
        bound_sources=bound_sources,
        relation_builder=relation_builder,
    )


def _field_render_output_pairs(
    *,
    index: int,
    payload: dict[str, Any],
    fields: tuple[dict[str, str], ...],
    namespace_render_outputs: bool,
    offset: int = 0,
) -> tuple[tuple[dict[str, str], str], ...]:
    answer_output_ids = _required_strings(
        payload.get("answer_output_ids"),
        "answer_output_ids",
    )
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
                _render_output_id(
                    index,
                    output_id,
                    namespace_render_outputs=namespace_render_outputs,
                ),
            )
        )
    return tuple(output)


def _group_fields_cover_answer_value_fields(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    group_fields: tuple[dict[str, str], ...],
) -> bool:
    group_field_ids = {item["field_id"] for item in group_fields}
    required_field_ids = {
        field_id
        for field_ids in _answer_value_field_ids_by_answer_output(
            payload=payload,
            bound=bound,
        ).values()
        for field_id in field_ids
    }
    return bool(required_field_ids and required_field_ids <= group_field_ids)


def _fulfillment_render_ids_by_answer_value_field(
    *,
    payload: dict[str, Any],
    bound: BoundSource,
    render_pairs: tuple[tuple[dict[str, str], str], ...],
) -> tuple[str, ...]:
    render_id_by_field_id = {
        item["field_id"]: render_output_id for item, render_output_id in render_pairs
    }
    field_ids_by_answer_output = _answer_value_field_ids_by_answer_output(
        payload=payload,
        bound=bound,
    )
    ordered_render_ids = tuple(render_output_id for _, render_output_id in render_pairs)
    output: list[str] = []
    for index, answer_output_id in enumerate(
        _required_strings(payload.get("answer_output_ids"), "answer_output_ids")
    ):
        render_output_id = _render_output_id_for_answer_value_field(
            answer_output_id=answer_output_id,
            field_ids_by_answer_output=field_ids_by_answer_output,
            render_id_by_field_id=render_id_by_field_id,
        )
        if not render_output_id and index < len(ordered_render_ids):
            render_output_id = ordered_render_ids[index]
        if not render_output_id:
            raise ValueError(
                "row pattern missing render output for answer value evidence"
            )
        output.append(render_output_id)
    return tuple(output)


def _render_output_id_for_answer_value_field(
    *,
    answer_output_id: str,
    field_ids_by_answer_output: dict[str, tuple[str, ...]],
    render_id_by_field_id: dict[str, str],
) -> str:
    for field_id in field_ids_by_answer_output.get(answer_output_id, ()):
        render_output_id = render_id_by_field_id.get(field_id)
        if render_output_id:
            return render_output_id
    return ""


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
