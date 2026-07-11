"""Compile model-facing fact plan patterns into executable answer plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.relations import Relation, RelationField
from fervis.lookup.answer_program.render_spec import RenderSpec
from fervis.lookup.fact_planning.plan_shapes import SOURCE_BOUND_PLAN_SHAPES
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
from .parameterization import compiled_program_inputs, parameterize_relation
from .shared import RelationBuilder, _required_dicts, _text


@dataclass(frozen=True)
class _PatternCompilerSpec:
    compile: Callable[..., dict[str, Any]]
    requires_selected_sources: bool = False
    requires_value_expressions: bool = False


def compile_pattern_answer_program(
    payload: dict[str, Any],
    *,
    bound_sources: tuple[BoundSource, ...],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    input_context: CompilerInputContext,
) -> tuple[AnswerProgram, BindingSet]:
    answer_payloads = _required_dicts(payload.get("answers"), "answers")
    bound_sources_by_id = {item.id: item for item in bound_sources}
    namespace_render_outputs = len(answer_payloads) > 1
    parameters = {
        item.id: item for item in input_context.program_inputs.parameters
    }
    bindings = {
        item.parameter_id: item
        for item in input_context.program_inputs.bindings.bindings
    }

    def build_relation(
        relation_id: str,
        source: DraftRelationSource,
        fields: tuple[RelationField, ...],
    ) -> Relation:
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
            payload=item,
            namespace_render_outputs=namespace_render_outputs,
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
        for index, item in enumerate(answer_payloads)
    ]
    compiled_inputs = compiled_program_inputs(
        parameters=parameters,
        bindings=bindings,
    )
    answer = AnswerProgram(
        fulfillment=tuple(
            item for compiled_item in compiled for item in compiled_item["fulfillment"]
        ),
        parameters=compiled_inputs.parameters,
        relations=tuple(
            item for compiled_item in compiled for item in compiled_item["relations"]
        ),
        operations=tuple(
            item for compiled_item in compiled for item in compiled_item["operations"]
        ),
        render_spec=RenderSpec(
            relation_outputs=tuple(
                item
                for compiled_item in compiled
                for item in compiled_item["relation_outputs"]
            ),
            scalar_outputs=tuple(
                item
                for compiled_item in compiled
                for item in compiled_item["scalar_outputs"]
            ),
        ),
    )
    return answer, compiled_inputs.bindings


def _compile_pattern_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    input_context: CompilerInputContext,
    relation_builder: RelationBuilder,
) -> dict[str, Any]:
    pattern = _text(payload.get("pattern"))
    compiler = _PATTERN_COMPILERS.get(pattern)
    if compiler is None:
        raise ValueError(f"unsupported fact plan pattern: {pattern}")
    kwargs: dict[str, Any] = {
        "index": index,
        "payload": payload,
        "namespace_render_outputs": namespace_render_outputs,
        "bound_sources": bound_sources,
        "relation_builder": relation_builder,
    }
    if pattern in SOURCE_BOUND_PLAN_SHAPES:
        _require_source_binding_selected(
            payload,
            source_binding_ids_by_requested_fact_id,
        )
    if compiler.requires_selected_sources:
        kwargs["allowed_source_binding_ids"] = _allowed_source_binding_ids(
            payload,
            source_binding_ids_by_requested_fact_id,
        )
        kwargs["allowed_source_binding_ids_by_requirement"] = (
            _allowed_source_binding_ids_by_requirement(
                payload,
                source_binding_ids_by_requirement_by_requested_fact_id,
            )
        )
    if compiler.requires_value_expressions:
        kwargs["input_context"] = input_context
    return compiler.compile(**kwargs)


_PATTERN_COMPILERS: dict[str, _PatternCompilerSpec] = {
    "list_rows": _PatternCompilerSpec(_compile_project_pattern_answer),
    "grouped_rows": _PatternCompilerSpec(_compile_project_pattern_answer),
    "direct_field_value": _PatternCompilerSpec(_compile_direct_field_value_answer),
    "aggregate_scalar": _PatternCompilerSpec(_compile_aggregate_pattern_answer),
    "aggregate_by_group": _PatternCompilerSpec(_compile_aggregate_pattern_answer),
    "ranked_aggregate": _PatternCompilerSpec(
        _compile_ranked_aggregate_answer,
        requires_value_expressions=True,
    ),
    "computed_scalar": _PatternCompilerSpec(
        _compile_computed_scalar_answer,
        requires_value_expressions=True,
    ),
    "set_difference": _PatternCompilerSpec(
        _compile_set_difference_answer,
        requires_selected_sources=True,
    ),
    "joined_rows": _PatternCompilerSpec(
        _compile_joined_rows_answer,
        requires_selected_sources=True,
    ),
}


def _allowed_source_binding_ids(
    payload: dict[str, Any],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    return source_binding_ids_by_requested_fact_id.get(
        _text(payload.get("requested_fact_id")),
        (),
    )


def _require_source_binding_selected(
    payload: dict[str, Any],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> None:
    selected = set(
        _allowed_source_binding_ids(payload, source_binding_ids_by_requested_fact_id)
    )
    if _text(payload.get("source_binding_id")) not in selected:
        raise ValueError("fact plan references source outside selected plan shape")


def _allowed_source_binding_ids_by_requirement(
    payload: dict[str, Any],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
) -> Mapping[str, tuple[str, ...]]:
    return source_binding_ids_by_requirement_by_requested_fact_id.get(
        _text(payload.get("requested_fact_id")),
        {},
    )
