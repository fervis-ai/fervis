"""Public answer-program verification boundary."""

from ._shared import (
    AnswerProgram,
    AuthorizedExecutionSources,
    CatalogSelectionResult,
    RelationCatalog,
    RelationRows,
)
from .relation_program import PreparedRelationProgram
from .answer_program import (
    prepare_answer_program,
    verify_prepared_answer_program,
)
from fervis.lookup.answer_program.inputs import CompiledProgramInputs


def verify_answer_program_structure(
    program: AnswerProgram,
    *,
    compiled_inputs: CompiledProgramInputs,
    catalog: RelationCatalog,
    memory_relations: tuple[RelationRows, ...] = (),
    catalog_selection: CatalogSelectionResult | None = None,
    authorized_sources: AuthorizedExecutionSources | None = None,
) -> AnswerProgram:
    execution_catalog = _execution_catalog(catalog, authorized_sources)
    structured = prepare_answer_program(
        program,
        compiled_inputs=compiled_inputs,
        catalog=execution_catalog,
        memory_relations=memory_relations,
        catalog_selection=catalog_selection,
        authorized_sources=authorized_sources,
    )
    return structured.program


def _execution_catalog(
    catalog: RelationCatalog | None,
    authorized_sources: AuthorizedExecutionSources | None,
) -> RelationCatalog | None:
    if authorized_sources is None:
        return catalog
    return authorized_sources.relation_catalog


__all__ = [
    "PreparedRelationProgram",
    "prepare_answer_program",
    "verify_answer_program_structure",
    "verify_prepared_answer_program",
]
