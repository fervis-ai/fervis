"""Execute physical access graphs without inventing a factual question template."""

from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.answer_program.compilation import close_catalog_defaults
from fervis.lookup.answer_program.request_projection import project_request_arguments
from fervis.lookup.answer_program.inputs import compile_relation_program_inputs
from fervis.lookup.answer_program.instantiation import materialize_relation_inputs
from fervis.lookup.answer_program.source_materialization import (
    execute_relation_inputs,
    RelationExecutionResult,
)
from fervis.lookup.plan_execution.verification.relation_program import (
    prepare_relation_program,
    verify_prepared_relation_program,
)
from fervis.lookup.plan_execution.authorized_sources import AuthorizedExecutionSources
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.memory.projection import LookupMemory


def execute_access_program(
    program: RelationProgram,
    *,
    catalog: RelationCatalog,
    read_session: ApiReadSession,
    bindings: BindingSet = BindingSet(),
    authority_ref: str = "",
) -> RelationExecutionResult:
    sources = build_row_source_catalog(catalog)
    program = close_catalog_defaults(
        project_request_arguments(program), row_sources=sources
    )
    inputs = compile_relation_program_inputs(program, bindings=bindings)
    prepared = prepare_relation_program(
        program,
        compiled_inputs=inputs,
        catalog=catalog,
        authorized_sources=AuthorizedExecutionSources(
            catalog, tuple(read.id for read in catalog.reads)
        ),
    )
    materialized = materialize_relation_inputs(
        prepared.program,
        bindings=prepared.bindings,
        catalog=catalog,
        row_sources=prepared.row_sources,
    )
    verify_prepared_relation_program(
        prepared, materialized=materialized, catalog=catalog
    )
    return execute_relation_inputs(
        prepared.program,
        materialized,
        catalog=catalog,
        row_sources=prepared.row_sources,
        memory=LookupMemory(),
        read_session=read_session,
        authority_ref=authority_ref,
    )
