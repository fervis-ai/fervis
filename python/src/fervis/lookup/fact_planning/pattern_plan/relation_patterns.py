"""Multi-relation pattern compilers."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.answer_program.operations import (
    AntiJoinSpec,
    JoinKey,
    JoinSpec,
    Operation,
    ProjectField,
    ProjectSpec,
    RelationRole,
    RelationRoleRef,
)
from fervis.lookup.answer_program.result_projection import RelationResultOutput
from fervis.lookup.answer_program.result_projection import (
    EntityKeyProjection,
    EntityKeyProjectionComponent,
)
from fervis.lookup.source_binding import (
    BoundSource,
    SourceFulfillment,
    entity_evidence_entity_kind,
    entity_evidence_key_id,
)
from fervis.lookup.fact_planning.provider_contract import (
    JoinedRowsAnswerOutput,
    SetDifferenceAnswerOutput,
)

from .shared import (
    RelationBuilder,
    _field_spec,
    _identity_relation_fields,
    _pattern_output_relation_id,
    _relation_fields,
    _relation_for_bound_source,
)
from .result_ids import _result_output_id
from .parameterization import ParameterizedRelation
from fervis.lookup.fact_planning.compiled_patterns import (
    CompiledPattern,
    PatternAddress,
)


def _compile_set_difference_answer(
    *,
    index: int,
    answer: SetDifferenceAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    allowed_source_binding_ids: tuple[str, ...],
    allowed_source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
    relation_builder: RelationBuilder,
) -> CompiledPattern:
    candidate = answer.candidate
    observed = answer.observed
    candidate_identity_fields = candidate.identity_fields
    observed_identity_fields = observed.identity_fields
    _validate_relation_operand_sources_selected(
        (candidate.source_binding_id, observed.source_binding_id),
        allowed_source_binding_ids=allowed_source_binding_ids,
    )
    _validate_relation_operand_source_role(
        candidate.source_binding_id,
        requirement_id="candidate_set",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    _validate_relation_operand_source_role(
        observed.source_binding_id,
        requirement_id="observed_set",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    if len(candidate_identity_fields) != len(observed_identity_fields):
        raise ValueError("set_difference identity field counts must match")
    requested_fact_id = answer.requested_fact_id
    answer_output_ids = answer.answer_output_ids
    candidate_source = bound_sources[candidate.source_binding_id]
    fulfillments = _set_difference_fulfillments(
        candidate_source,
        requested_fact_id=requested_fact_id,
        answer_output_ids=answer_output_ids,
        identity_field_ids=candidate_identity_fields,
    )
    candidate_relation_id = f"answer_{index}_candidate"
    observed_relation_id = f"answer_{index}_observed"
    output_relation_id = _pattern_output_relation_id(index)
    candidate_relation_fields = _identity_relation_fields(candidate_identity_fields)
    observed_relation_fields = _identity_relation_fields(observed_identity_fields)
    built_relations = (
        _relation_for_bound_source(
            relation_id=candidate_relation_id,
            address=PatternAddress(
                requested_fact_id=requested_fact_id,
                answer_output_ids=answer_output_ids,
                plan_shape=answer.pattern,
                source_binding_id=candidate.source_binding_id,
            ),
            relation_fields=candidate_relation_fields,
            bound_sources=bound_sources,
            relation_builder=relation_builder,
        ),
        _relation_for_bound_source(
            relation_id=observed_relation_id,
            address=PatternAddress(
                requested_fact_id=requested_fact_id,
                answer_output_ids=answer_output_ids,
                plan_shape=answer.pattern,
                source_binding_id=observed.source_binding_id,
            ),
            relation_fields=observed_relation_fields,
            bound_sources=bound_sources,
            relation_builder=relation_builder,
        ),
    )
    operations = (
        *(operation for item in built_relations for operation in item.operations),
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
                    ProjectField(source=field_id, output=field_id)
                    for field_id in candidate_identity_fields
                ),
            ),
            output_relation=output_relation_id,
        ),
    )
    fact_fulfillment = _set_difference_fact_fulfillment(
        fulfillments,
        requested_fact_id=requested_fact_id,
        answer_index=index,
        namespace_result_outputs=namespace_result_outputs,
    )
    result_outputs = _set_difference_result_outputs(
        fulfillments,
        output_relation_id=output_relation_id,
        answer_index=index,
        namespace_result_outputs=namespace_result_outputs,
    )
    return CompiledPattern(
        fulfillment=fact_fulfillment,
        relations=tuple(item.relation for item in built_relations),
        operations=operations,
        relation_outputs=result_outputs,
        scalar_outputs=(),
    )


def _set_difference_fulfillments(
    source: BoundSource,
    *,
    requested_fact_id: str,
    answer_output_ids: tuple[str, ...],
    identity_field_ids: tuple[str, ...],
) -> tuple[SourceFulfillment, ...]:
    fulfillments_by_output = {
        fulfillment.answer_output_id: fulfillment
        for fulfillment in source.fulfillments
        if fulfillment.requested_fact_id == requested_fact_id
    }
    fulfillments: list[SourceFulfillment] = []
    for answer_output_id in answer_output_ids:
        fulfillment = fulfillments_by_output.get(answer_output_id)
        if fulfillment is None or fulfillment.entity_evidence is None:
            raise ValueError("set_difference requires candidate-key answer evidence")
        evidence_field_ids = tuple(
            component.field_id for component in fulfillment.entity_evidence.components
        )
        if evidence_field_ids != identity_field_ids:
            raise ValueError(
                "set_difference identity fields must match candidate-key evidence"
            )
        fulfillments.append(fulfillment)
    return tuple(fulfillments)


def _set_difference_fact_fulfillment(
    fulfillments: tuple[SourceFulfillment, ...],
    *,
    requested_fact_id: str,
    answer_index: int,
    namespace_result_outputs: bool,
) -> tuple[FactFulfillment, ...]:
    return tuple(
        FactFulfillment(
            requested_fact_id=requested_fact_id,
            answer_output_id=fulfillment.answer_output_id,
            result_output_id=_result_output_id(
                answer_index,
                fulfillment.answer_output_id,
                namespace_result_outputs=namespace_result_outputs,
            ),
        )
        for fulfillment in fulfillments
    )


def _set_difference_result_outputs(
    fulfillments: tuple[SourceFulfillment, ...],
    *,
    output_relation_id: str,
    answer_index: int,
    namespace_result_outputs: bool,
) -> tuple[RelationResultOutput, ...]:
    outputs: list[RelationResultOutput] = []
    for fulfillment in fulfillments:
        evidence = fulfillment.entity_evidence
        if evidence is None:
            raise ValueError("set_difference requires candidate-key answer evidence")
        components = tuple(
            EntityKeyProjectionComponent(
                component_id=component.component_id,
                field_id=component.field_id,
            )
            for component in evidence.components
        )
        entity_key = EntityKeyProjection(
            entity_kind=entity_evidence_entity_kind(evidence),
            key_id=entity_evidence_key_id(evidence),
            components=components,
        )
        result_output_id = _result_output_id(
            answer_index,
            fulfillment.answer_output_id,
            namespace_result_outputs=namespace_result_outputs,
        )
        outputs.append(
            RelationResultOutput(
                id=result_output_id,
                relation_id=output_relation_id,
                entity_key=entity_key,
                role="answer_value",
            )
        )
    return tuple(outputs)


def _compile_joined_rows_answer(
    *,
    index: int,
    answer: JoinedRowsAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    allowed_source_binding_ids: tuple[str, ...],
    allowed_source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
    relation_builder: RelationBuilder,
) -> CompiledPattern:
    left = answer.left
    right = answer.right
    left_fields = tuple(
        _field_spec({"field_id": item.field_id}) for item in left.fields
    )
    right_fields = tuple(
        _field_spec({"field_id": item.field_id}) for item in right.fields
    )
    join_keys = tuple(
        {"left_field_id": item.left_field_id, "right_field_id": item.right_field_id}
        for item in answer.join_keys
    )
    output_fields = tuple(
        {
            "field_id": item.field_id,
            "output_field_id": item.field_id,
            "label": item.field_id,
            "side": item.side,
        }
        for item in answer.output_fields
    )
    _validate_relation_operand_sources_selected(
        (left.source_binding_id, right.source_binding_id),
        allowed_source_binding_ids=allowed_source_binding_ids,
    )
    _validate_relation_operand_source_role(
        left.source_binding_id,
        requirement_id="left",
        allowed_source_binding_ids_by_requirement=(
            allowed_source_binding_ids_by_requirement
        ),
    )
    _validate_relation_operand_source_role(
        right.source_binding_id,
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
        requested_fact_id=answer.requested_fact_id,
        answer_output_ids=answer.answer_output_ids,
        relations=(
            _relation_for_bound_source(
                relation_id=left_relation_id,
                address=PatternAddress(
                    requested_fact_id=answer.requested_fact_id,
                    answer_output_ids=answer.answer_output_ids,
                    plan_shape=answer.pattern,
                    source_binding_id=left.source_binding_id,
                ),
                relation_fields=left_relation_fields,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
            ),
            _relation_for_bound_source(
                relation_id=right_relation_id,
                address=PatternAddress(
                    requested_fact_id=answer.requested_fact_id,
                    answer_output_ids=answer.answer_output_ids,
                    plan_shape=answer.pattern,
                    source_binding_id=right.source_binding_id,
                ),
                relation_fields=right_relation_fields,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
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
        namespace_result_outputs=namespace_result_outputs,
    )


def _compiled_multi_relation_pattern(
    *,
    requested_fact_id: str,
    answer_output_ids: tuple[str, ...],
    relations: tuple[ParameterizedRelation, ...],
    operations: tuple[Operation, ...],
    output_relation_id: str,
    output_fields: tuple[dict[str, str], ...],
    answer_index: int,
    namespace_result_outputs: bool,
) -> CompiledPattern:
    fulfillment = tuple(
        FactFulfillment(
            requested_fact_id=requested_fact_id,
            answer_output_id=answer_output_id,
            result_output_id=_result_output_id(
                answer_index,
                output_fields[min(output_index, len(output_fields) - 1)][
                    "output_field_id"
                ],
                namespace_result_outputs=namespace_result_outputs,
            ),
        )
        for output_index, answer_output_id in enumerate(answer_output_ids)
    )
    relation_outputs = tuple(
        RelationResultOutput(
            id=_result_output_id(
                answer_index,
                item["output_field_id"],
                namespace_result_outputs=namespace_result_outputs,
            ),
            relation_id=output_relation_id,
            field_id=item["output_field_id"],
            label=item["label"] if namespace_result_outputs else "",
            role="answer_value",
        )
        for item in output_fields
    )
    return CompiledPattern(
        fulfillment=fulfillment,
        relations=tuple(item.relation for item in relations),
        operations=(
            *(operation for item in relations for operation in item.operations),
            *operations,
        ),
        relation_outputs=relation_outputs,
        scalar_outputs=(),
    )


def _validate_relation_operand_sources_selected(
    source_binding_ids: tuple[str, ...],
    *,
    allowed_source_binding_ids: tuple[str, ...],
) -> None:
    allowed = set(allowed_source_binding_ids)
    for source_binding_id in source_binding_ids:
        if source_binding_id not in allowed:
            raise ValueError("fact plan references source outside selected plan shape")


def _validate_relation_operand_source_role(
    source_binding_id: str,
    *,
    requirement_id: str,
    allowed_source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
) -> None:
    allowed = set(allowed_source_binding_ids_by_requirement.get(requirement_id, ()))
    if not allowed:
        return
    if source_binding_id not in allowed:
        raise ValueError("fact plan references source outside selected operand role")
