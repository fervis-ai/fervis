"""Acquire current source structure before any saved or newly compiled execution."""

from fervis.lineage.enums import SourceInspectionPhase
from fervis.lookup.lineage.representation import RepresentationInspectionAudit
from fervis.lookup.source_reads.representation import inspect_selected_representations


def prepare_execution_catalog(
    *,
    run_id,
    catalog,
    program,
    data_access_port,
    lineage_step_sink,
    allowed_read_ids=None,
    inspection_phase=SourceInspectionPhase.EXECUTION,
):
    from fervis.lookup.plan_execution.authorized_sources import require_read_in_scope
    from fervis.lookup.answer_program.relations import SourceKind

    read_ids = tuple(
        relation.source.read_id
        for relation in program.relations
        if relation.source.kind is SourceKind.API_READ
    )
    for read_id in read_ids:
        require_read_in_scope(read_id, allowed_read_ids)
    audit = RepresentationInspectionAudit(
        run_id=run_id, sink=lineage_step_sink, phase=inspection_phase
    )
    try:
        return inspect_selected_representations(
            catalog,
            read_ids=read_ids,
            data_access_port=data_access_port,
            on_response=audit.observe,
        )
    finally:
        audit.flush()


def execution_catalog_with_observed_sources(full_catalog, selected_catalog):
    """Preserve discovered definitions without dropping operational prerequisites."""
    from dataclasses import replace
    from fervis.lookup.plan_execution.errors import VerificationError

    full_ids = {read.id for read in full_catalog.reads}
    if any(read.id not in full_ids for read in selected_catalog.reads):
        raise VerificationError("selected source is absent from the current catalog")
    observed = {read.id: read for read in selected_catalog.reads}
    facts = {fact.ref: fact for fact in (*full_catalog.facts, *selected_catalog.facts)}
    return replace(full_catalog,
        reads=tuple(observed.get(read.id, read) for read in full_catalog.reads),
        facts=tuple(facts.values()),
    )
