"""Static compilation of one closed answer program."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from fervis.lookup.contract_codec import canonicalize_answer_program
from fervis.lookup.answer_program.compatibility import (
    build_program_compatibility,
)
from fervis.lookup.answer_program.inputs import compile_relation_program_inputs
from fervis.lookup.answer_program.model import AnswerProgram, RelationProgram
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    SourceKind,
)
from fervis.lookup.answer_program.values import (
    BindingSet,
    EnvironmentRef,
)
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.relation_catalog.row_sources.model import RowSourceCatalog
from fervis.lookup.relation_catalog.row_sources.builder import build_row_source_catalog
from fervis.lookup.relation_catalog.row_sources.lookup import row_source_for_relation


def compile_answer_program(
    program: AnswerProgram,
    *,
    question_contract: QuestionContract,
    catalog: RelationCatalog,
    bindings: BindingSet,
    memory_relations: tuple[RelationRows, ...] = (),
) -> tuple[AnswerProgram, BindingSet]:
    """Close, statically verify, pin, and canonicalize a reusable program."""

    row_sources = build_row_source_catalog(
        catalog,
        memory_relations=memory_relations,
    )
    from fervis.lookup.answer_program.request_projection import project_request_arguments
    closed_sources = close_catalog_defaults(project_request_arguments(program), row_sources=row_sources)
    compiled_inputs = compile_relation_program_inputs(
        closed_sources,
        bindings=bindings,
    )
    closed = replace(
        closed_sources,
        inputs=question_contract.inputs,
        input_denotations=question_contract.input_denotations,
        fact_template=question_contract.requested_facts,
        parameters=compiled_inputs.parameters,
        compatibility=build_program_compatibility(
            closed_sources,
            row_sources=row_sources,
        ),
    )
    from fervis.lookup.plan_execution.verification import (
        verify_answer_program_structure,
    )

    verified = verify_answer_program_structure(
        closed,
        compiled_inputs=compiled_inputs,
        catalog=catalog,
        memory_relations=memory_relations,
    )
    canonical = canonicalize_answer_program(verified)
    return canonical, compiled_inputs.bindings


_Program = TypeVar("_Program", bound=RelationProgram)

def close_catalog_defaults(
    program: _Program,
    *,
    row_sources: RowSourceCatalog,
) -> _Program:
    """Make every selected source default an explicit environment expression."""

    relations = []
    for relation in program.relations:
        source = relation.source
        if source.kind not in {SourceKind.API_READ, SourceKind.GENERATED_CALENDAR}:
            relations.append(relation)
            continue
        row_source = row_source_for_relation(relation, row_sources=row_sources)
        bound_param_ids = {binding.param_id for binding in source.param_bindings}
        defaults = tuple(
            EndpointParamBinding(
                param_id=param.id,
                value_expr=EnvironmentRef(
                    key="catalog_param_default",
                    source_ref=f"{row_source.id}:{param.id}",
                ),
            )
            for param in row_source.params
            if param.default is not None and param.id not in bound_param_ids
        )
        relations.append(
            replace(
                relation,
                source=replace(
                    source,
                    param_bindings=(*source.param_bindings, *defaults),
                ),
            )
        )
    return replace(program, relations=tuple(relations))
