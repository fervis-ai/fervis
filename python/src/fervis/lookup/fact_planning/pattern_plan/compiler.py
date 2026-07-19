"""Compile model-facing fact plan patterns into executable answer plans."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.relations import RelationField
from fervis.lookup.answer_program.result_projection import ResultProjection
from fervis.lookup.fact_planning.provider_contract import (
    AggregateScalarAnswerOutput,
    ComputedScalarAnswerOutput,
    DirectFieldValueAnswerOutput,
    GroupedAggregateAnswerOutput,
    GroupedRowsAnswerOutput,
    JoinedRowsAnswerOutput,
    ListRowsAnswerOutput,
    RankedRowsAnswerOutput,
    PatternAnswerOutput,
    RankedAggregateAnswerOutput,
    SetDifferenceAnswerOutput,
)
from fervis.lookup.source_binding import BoundSource
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.source_binding.compiler_ir import DraftRelationSource

from .aggregate_patterns import (
    _compile_aggregate_pattern_answer,
    _compile_ranked_aggregate_answer,
)
from .relation_patterns import (
    _compile_joined_rows_answer,
    _compile_set_difference_answer,
)
from .row_patterns import (
    _compile_direct_field_value_answer,
    _compile_project_pattern_answer,
)
from .scalar_patterns import _compile_computed_scalar_answer
from .parameterization import (
    ParameterizedRelation,
    compiled_program_inputs,
    parameterize_relation,
)
from .shared import RelationBuilder
from fervis.lookup.fact_planning.compiled_patterns import CompiledPattern


def compile_pattern_answer_program(
    answers: tuple[PatternAnswerOutput, ...],
    *,
    bound_sources: tuple[BoundSource, ...],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    input_context: CompilerInputContext,
) -> tuple[AnswerProgram, BindingSet]:
    bound_sources_by_id = {item.id: item for item in bound_sources}
    namespace_result_outputs = len(answers) > 1
    parameters = {item.id: item for item in input_context.program_inputs.parameters}
    bindings = {
        item.parameter_id: item
        for item in input_context.program_inputs.bindings.bindings
    }

    def build_relation(
        relation_id: str,
        source: DraftRelationSource,
        fields: tuple[RelationField, ...],
    ) -> ParameterizedRelation:
        return parameterize_relation(
            relation_id=relation_id,
            source=source,
            fields=fields,
            input_context=input_context,
            parameters=parameters,
            bindings=bindings,
        )

    compiled = [
        _compile_pattern_answer(
            index=index + 1,
            answer=answer,
            namespace_result_outputs=namespace_result_outputs,
            bound_sources=bound_sources_by_id,
            source_binding_ids_by_requested_fact_id=(
                source_binding_ids_by_requested_fact_id
            ),
            source_binding_ids_by_requirement_by_requested_fact_id=(
                source_binding_ids_by_requirement_by_requested_fact_id
            ),
            input_context=input_context,
            relation_builder=build_relation,
        )
        for index, answer in enumerate(answers)
    ]
    compiled_inputs = compiled_program_inputs(
        parameters=parameters,
        bindings=bindings,
    )
    answer = AnswerProgram(
        fulfillment=tuple(
            item for compiled_item in compiled for item in compiled_item.fulfillment
        ),
        parameters=compiled_inputs.parameters,
        relations=tuple(
            item for compiled_item in compiled for item in compiled_item.relations
        ),
        operations=tuple(
            item for compiled_item in compiled for item in compiled_item.operations
        ),
        result_projection=ResultProjection(
            relation_outputs=tuple(
                item
                for compiled_item in compiled
                for item in compiled_item.relation_outputs
            ),
            scalar_outputs=tuple(
                item
                for compiled_item in compiled
                for item in compiled_item.scalar_outputs
            ),
        ),
    )
    return answer, compiled_inputs.bindings


def _compile_pattern_answer(
    *,
    index: int,
    answer: PatternAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    input_context: CompilerInputContext,
    relation_builder: RelationBuilder,
) -> CompiledPattern:
    match answer:
        case (
            ListRowsAnswerOutput()
            | RankedRowsAnswerOutput()
            | GroupedRowsAnswerOutput()
        ):
            _require_source_binding_selected(
                answer.requested_fact_id,
                answer.source_binding_id,
                source_binding_ids_by_requested_fact_id,
            )
            return _compile_project_pattern_answer(
                index=index,
                answer=answer,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
                input_context=input_context,
            )
        case DirectFieldValueAnswerOutput():
            _require_source_binding_selected(
                answer.requested_fact_id,
                answer.source_binding_id,
                source_binding_ids_by_requested_fact_id,
            )
            return _compile_direct_field_value_answer(
                index=index,
                answer=answer,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
                input_context=input_context,
            )
        case AggregateScalarAnswerOutput() | GroupedAggregateAnswerOutput():
            _require_source_binding_selected(
                answer.requested_fact_id,
                answer.source_binding_id,
                source_binding_ids_by_requested_fact_id,
            )
            return _compile_aggregate_pattern_answer(
                index=index,
                answer=answer,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
            )
        case RankedAggregateAnswerOutput():
            _require_source_binding_selected(
                answer.requested_fact_id,
                answer.source_binding_id,
                source_binding_ids_by_requested_fact_id,
            )
            return _compile_ranked_aggregate_answer(
                answer=answer,
                index=index,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                input_context=input_context,
                relation_builder=relation_builder,
            )
        case ComputedScalarAnswerOutput():
            return _compile_computed_scalar_answer(
                answer=answer,
                index=index,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                input_context=input_context,
                relation_builder=relation_builder,
            )
        case SetDifferenceAnswerOutput():
            allowed_ids = _allowed_source_binding_ids(
                answer.requested_fact_id,
                source_binding_ids_by_requested_fact_id,
            )
            allowed_by_requirement = _allowed_source_binding_ids_by_requirement(
                answer.requested_fact_id,
                source_binding_ids_by_requirement_by_requested_fact_id,
            )
            return _compile_set_difference_answer(
                answer=answer,
                index=index,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                allowed_source_binding_ids=allowed_ids,
                allowed_source_binding_ids_by_requirement=allowed_by_requirement,
                relation_builder=relation_builder,
            )
        case JoinedRowsAnswerOutput():
            allowed_ids = _allowed_source_binding_ids(
                answer.requested_fact_id,
                source_binding_ids_by_requested_fact_id,
            )
            allowed_by_requirement = _allowed_source_binding_ids_by_requirement(
                answer.requested_fact_id,
                source_binding_ids_by_requirement_by_requested_fact_id,
            )
            return _compile_joined_rows_answer(
                answer=answer,
                index=index,
                namespace_result_outputs=namespace_result_outputs,
                bound_sources=bound_sources,
                allowed_source_binding_ids=allowed_ids,
                allowed_source_binding_ids_by_requirement=allowed_by_requirement,
                relation_builder=relation_builder,
            )
    raise AssertionError("unreachable pattern answer")


def _allowed_source_binding_ids(
    requested_fact_id: str,
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    return source_binding_ids_by_requested_fact_id.get(
        requested_fact_id,
        (),
    )


def _require_source_binding_selected(
    requested_fact_id: str,
    source_binding_id: str,
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> None:
    selected = set(
        _allowed_source_binding_ids(
            requested_fact_id,
            source_binding_ids_by_requested_fact_id,
        )
    )
    if source_binding_id not in selected:
        raise ValueError("fact plan references source outside selected plan shape")


def _allowed_source_binding_ids_by_requirement(
    requested_fact_id: str,
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
) -> Mapping[str, tuple[str, ...]]:
    return source_binding_ids_by_requirement_by_requested_fact_id.get(
        requested_fact_id,
        {},
    )
