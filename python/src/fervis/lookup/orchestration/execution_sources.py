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
    bindings=None,
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
    inspection_args = _bound_inspection_arguments(
        catalog=catalog, program=program, bindings=bindings,
    )
    try:
        return inspect_selected_representations(
            catalog,
            read_ids=read_ids,
            data_access_port=data_access_port,
            on_response=audit.observe,
            inspection_args_by_read=inspection_args,
        )
    finally:
        audit.flush()


def _bound_inspection_arguments(
    *, catalog, program, bindings, include_observed_read_ids=frozenset(),
):
    """Recover only direct, original question addresses from validated bindings."""
    if bindings is None:
        return {}
    from fervis.lookup.answer_program.inputs import (
        compile_relation_program_inputs, resolve_value_expression,
    )
    from fervis.lookup.answer_program.relations import SourceKind
    from fervis.lookup.canonical_data import canonical_runtime_json
    from fervis.lookup.relation_catalog.model import requires_caller_supplied_input
    from fervis.lookup.relation_catalog.parameter_values import parse_catalog_parameter_value

    compiled = compile_relation_program_inputs(program, bindings=bindings)
    by_read = {}
    conflicted = set()
    for relation in program.relations:
        source = relation.source
        if source.kind is not SourceKind.API_READ or source.argument_relation_id:
            continue
        read = catalog.read(source.read_id)
        if (read.fields and read.id not in include_observed_read_ids) or (read.source_metadata or {}).get("representation_status") in {
            "unavailable", "read_failed"
        }:
            continue
        required = {
            param.ref: param for param in read.params
            if requires_caller_supplied_input(param)
        }
        if not required:
            continue
        bound = {binding.param_id: binding for binding in source.param_bindings}
        if not set(required) <= set(bound):
            continue
        args = {}
        for ref, param in required.items():
            resolved = resolve_value_expression(
                bound[ref].value_expr, bindings=compiled.bindings
            )
            input_ref = resolved.fact_value.known_input_id
            if not input_ref or f"question_input:{input_ref}" not in resolved.proof_refs:
                break
            args[ref] = parse_catalog_parameter_value(
                resolved.value, type_name=param.type, choices=param.choices
            )
        if set(args) != set(required):
            continue
        previous = by_read.get(read.id)
        if previous is not None and canonical_runtime_json(previous) != canonical_runtime_json(args):
            conflicted.add(read.id)
            continue
        by_read[read.id] = args
    return {read_id: args for read_id, args in by_read.items()
            if read_id not in conflicted}


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
