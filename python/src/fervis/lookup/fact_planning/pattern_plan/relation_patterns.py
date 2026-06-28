"""Multi-relation pattern compilers."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from fervis.lookup.fact_plan.fact_plan import FactFulfillment
from fervis.lookup.fact_plan.operations import (
    AntiJoinSpec,
    JoinKey,
    JoinSpec,
    Operation,
    ProjectField,
    ProjectSpec,
    RelationRole,
    RelationRoleRef,
)
from fervis.lookup.fact_plan.relations import Relation
from fervis.lookup.fact_plan.render_spec import RenderRelationOutput
from fervis.lookup.source_binding import BoundSource

from .shared import (
    _dict,
    _field_specs,
    _identity_relation_fields,
    _join_key_specs,
    _joined_output_fields,
    _pattern_output_relation_id,
    _relation_fields,
    _relation_for_bound_source,
    _relation_operand,
    _required_strings,
    _text,
)
from .render_ids import _render_output_id


def _compile_set_difference_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
    allowed_source_binding_ids: tuple[str, ...],
    allowed_source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    candidate = _relation_operand(_dict(payload.get("candidate"), "candidate"))
    observed = _relation_operand(_dict(payload.get("observed"), "observed"))
    output_fields = _field_specs(candidate.get("output_fields"))
    candidate_identity_fields = _required_strings(
        candidate.get("identity_fields"), "candidate.identity_fields"
    )
    observed_identity_fields = _required_strings(
        observed.get("identity_fields"), "observed.identity_fields"
    )
    _validate_relation_operand_sources_selected(
        (candidate, observed),
        allowed_source_binding_ids=allowed_source_binding_ids,
    )
    _validate_relation_operand_source_role(
        candidate,
        requirement_id="candidate_set",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    _validate_relation_operand_source_role(
        observed,
        requirement_id="observed_set",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    if len(candidate_identity_fields) != len(observed_identity_fields):
        raise ValueError("set_difference identity field counts must match")
    candidate_relation_id = f"answer_{index}_candidate"
    observed_relation_id = f"answer_{index}_observed"
    output_relation_id = _pattern_output_relation_id(index)
    candidate_relation_fields = (
        *_identity_relation_fields(candidate_identity_fields),
        *_relation_fields(output_fields),
    )
    observed_relation_fields = _identity_relation_fields(observed_identity_fields)
    return _compiled_multi_relation_pattern(
        payload=payload,
        relations=(
            _relation_for_bound_source(
                relation_id=candidate_relation_id,
                payload=_operand_payload_with_parent_context(
                    payload,
                    operand=candidate,
                ),
                relation_fields=candidate_relation_fields,
                bound_sources=bound_sources,
            ),
            _relation_for_bound_source(
                relation_id=observed_relation_id,
                payload=_operand_payload_with_parent_context(
                    payload,
                    operand=observed,
                ),
                relation_fields=observed_relation_fields,
                bound_sources=bound_sources,
            ),
        ),
        operations=(
            Operation(
                id=f"{output_relation_id}_anti_join",
                spec=AntiJoinSpec(
                    candidate=RelationRoleRef(
                        relation_id=candidate_relation_id,
                        role=RelationRole.ANTI_JOIN_CANDIDATE,
                        required_identity_fields=candidate_identity_fields,
                    ),
                    observed=RelationRoleRef(
                        relation_id=observed_relation_id,
                        role=RelationRole.ANTI_JOIN_OBSERVED,
                        required_identity_fields=observed_identity_fields,
                    ),
                    join_keys=tuple(
                        JoinKey(left=left, right=right)
                        for left, right in zip(
                            candidate_identity_fields,
                            observed_identity_fields,
                            strict=True,
                        )
                    ),
                    output_fields=tuple(
                        ProjectField(
                            source=item["field_id"],
                            output=item["output_field_id"],
                        )
                        for item in output_fields
                    ),
                ),
                output_relation=output_relation_id,
            ),
        ),
        output_relation_id=output_relation_id,
        output_fields=output_fields,
        answer_index=index,
        namespace_render_outputs=namespace_render_outputs,
    )


def _compile_joined_rows_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
    allowed_source_binding_ids: tuple[str, ...],
    allowed_source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    left = _relation_operand(_dict(payload.get("left"), "left"))
    right = _relation_operand(_dict(payload.get("right"), "right"))
    left_fields = _field_specs(left.get("fields"))
    right_fields = _field_specs(right.get("fields"))
    join_keys = _join_key_specs(payload.get("join_keys"))
    output_fields = _joined_output_fields(payload.get("output_fields"))
    _validate_relation_operand_sources_selected(
        (left, right),
        allowed_source_binding_ids=allowed_source_binding_ids,
    )
    _validate_relation_operand_source_role(
        left,
        requirement_id="left",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    _validate_relation_operand_source_role(
        right,
        requirement_id="right",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    left_relation_id = f"answer_{index}_left"
    right_relation_id = f"answer_{index}_right"
    joined_relation_id = f"{_pattern_output_relation_id(index)}_joined"
    output_relation_id = _pattern_output_relation_id(index)
    left_relation_fields = _relation_fields(left_fields)
    right_relation_fields = _relation_fields(right_fields)
    return _compiled_multi_relation_pattern(
        payload=payload,
        relations=(
            _relation_for_bound_source(
                relation_id=left_relation_id,
                payload=_operand_payload_with_parent_context(
                    payload,
                    operand=left,
                ),
                relation_fields=left_relation_fields,
                bound_sources=bound_sources,
            ),
            _relation_for_bound_source(
                relation_id=right_relation_id,
                payload=_operand_payload_with_parent_context(
                    payload,
                    operand=right,
                ),
                relation_fields=right_relation_fields,
                bound_sources=bound_sources,
            ),
        ),
        operations=(
            Operation(
                id=f"{joined_relation_id}_join",
                spec=JoinSpec(
                    left=left_relation_id,
                    right=right_relation_id,
                    join_keys=tuple(
                        JoinKey(
                            left=item["left_field_id"], right=item["right_field_id"]
                        )
                        for item in join_keys
                    ),
                ),
                output_relation=joined_relation_id,
            ),
            Operation(
                id=f"{output_relation_id}_project",
                spec=ProjectSpec(
                    input_relation=joined_relation_id,
                    fields=tuple(
                        ProjectField(
                            source=item["field_id"],
                            output=item["output_field_id"],
                        )
                        for item in output_fields
                    ),
                ),
                output_relation=output_relation_id,
            ),
        ),
        output_relation_id=output_relation_id,
        output_fields=output_fields,
        answer_index=index,
        namespace_render_outputs=namespace_render_outputs,
    )


def _compiled_multi_relation_pattern(
    *,
    payload: dict[str, Any],
    relations: tuple[Relation, ...],
    operations: tuple[Operation, ...],
    output_relation_id: str,
    output_fields: tuple[dict[str, str], ...],
    answer_index: int,
    namespace_render_outputs: bool,
) -> dict[str, Any]:
    answer_output_ids = _required_strings(
        payload.get("answer_output_ids"), "answer_output_ids"
    )
    return {
        "fulfillment": tuple(
            FactFulfillment(
                requested_fact_id=_text(payload.get("requested_fact_id")),
                answer_output_id=answer_output_id,
                render_output_id=_render_output_id(
                    answer_index,
                    output_fields[min(output_index, len(output_fields) - 1)][
                        "output_field_id"
                    ],
                    namespace_render_outputs=namespace_render_outputs,
                ),
            )
            for output_index, answer_output_id in enumerate(answer_output_ids)
        ),
        "values": (),
        "value_uses": (),
        "relations": relations,
        "operations": operations,
        "relation_outputs": tuple(
            RenderRelationOutput(
                id=_render_output_id(
                    answer_index,
                    item["output_field_id"],
                    namespace_render_outputs=namespace_render_outputs,
                ),
                relation_id=output_relation_id,
                field_id=item["output_field_id"],
                label=item["label"] if namespace_render_outputs else "",
                role="answer_value",
            )
            for item in output_fields
        ),
        "scalar_outputs": (),
    }


def _operand_payload_with_parent_context(
    parent: dict[str, Any],
    *,
    operand: dict[str, Any],
) -> dict[str, Any]:
    return {
        **operand,
        "requested_fact_id": parent.get("requested_fact_id"),
        "answer_output_ids": parent.get("answer_output_ids"),
        "pattern": parent.get("pattern"),
    }


def _validate_relation_operand_sources_selected(
    operands: tuple[dict[str, Any], ...],
    *,
    allowed_source_binding_ids: tuple[str, ...],
) -> None:
    allowed = set(allowed_source_binding_ids)
    for operand in operands:
        if _text(operand.get("source_binding_id")) not in allowed:
            raise ValueError("fact plan references source outside selected plan shape")


def _validate_relation_operand_source_role(
    operand: dict[str, Any],
    *,
    requirement_id: str,
    allowed_source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
) -> None:
    allowed = set(allowed_source_binding_ids_by_requirement.get(requirement_id, ()))
    if not allowed:
        return
    if _text(operand.get("source_binding_id")) not in allowed:
        raise ValueError("fact plan references source outside selected operand role")
