"""Source ownership for observed-record result representations."""

from fervis.lookup.question_contract.model import (
    FactLocalRef,
    FactTerm,
    AssociationTerm,
    Comparison,
    Aggregate,
    AggregateFunction,
)
from fervis.lookup.semantic_types import IdentifierType


def is_record_output_set(request, set_ref):
    ref = FactLocalRef.from_token(set_ref)
    for output in request.index.output_requirements:
        if output.value_ref == ref:
            return True
        term = request.index.term_by_ref.get(output.value_ref)
        if (
            isinstance(term, FactTerm)
            and isinstance(term.value_type, IdentifierType)
            and term.value_type.set_ref == ref.local_id
        ):
            return True
    return False


def validate_record_fields(request, *, set_ref, source_ref, identity_ref, fields):
    is_output = is_record_output_set(request, set_ref)
    if fields and (identity_ref is not None or not is_output):
        raise ValueError("Observed record fields require an anonymous output carrier")
    known = {
        field.field_ref for field in request.source_catalog.source(source_ref).fields
    }
    if (
        any(not name.strip() for name, _ in fields)
        or len({name for name, _ in fields}) != len(fields)
        or any(field_ref not in known for _, field_ref in fields)
    ):
        raise ValueError(
            "Observed record properties must name unique fields from their selected carrier"
        )
    if identity_ref is None and is_output and not fields:
        raise ValueError(
            "An anonymous output carrier requires selected observed record fields"
        )


def allows_row_occurrence_identity(request, fact_ref):
    ref = FactLocalRef.from_token(fact_ref)
    term = request.index.term_by_ref.get(ref)
    if not isinstance(term, FactTerm) or not isinstance(
        term.value_type, IdentifierType
    ):
        return False
    owner_ref = request.index.fact_local_ref_by_local_id.get(term.owner_ref)
    owner = request.index.term_by_ref.get(owner_ref)
    # An association-owned identity can denote its related row occurrence for
    # grouping/presentation, without asserting a globally unique entity key.
    if term.owner_ref != term.value_type.set_ref and not (
        isinstance(owner, AssociationTerm)
        and term.value_type.set_ref in {owner.from_set_ref, owner.to_set_ref}
    ):
        return False
    if ref in request.index.ordering_refs:
        return False
    consumers = [
        request.index.expression_by_ref.get(consumer)
        for consumer, dependencies in request.index.direct_dependencies_by_ref.items()
        if ref in dependencies
    ]
    from .reference_bindings import runtime_reference_uses
    reference_consumers = {use.expression_ref for use in runtime_reference_uses(request) if use.reference_fact_ref == ref}
    consumers = [node for node in consumers if not (isinstance(node, Comparison)
        and request.index.fact_local_ref_by_local_id[node.id] in reference_consumers)]
    if any(
        not isinstance(node, Aggregate)
        or node.function is not AggregateFunction.COUNT
        or node.argument_ref != ref.local_id
        for node in consumers
    ):
        return False
    return (
        bool(reference_consumers)
        or bool(consumers)
        or ref in request.index.grouping_refs
        or any(output.value_ref == ref for output in request.index.output_requirements)
    )
