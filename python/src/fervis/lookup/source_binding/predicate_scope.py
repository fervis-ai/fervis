"""When a predicate may narrow a read used by a relational expression."""

from fervis.lookup.question_contract import (
    FactLocalRef,
    Quantify,
    Quantifier,
    Aggregate,
    RelatedRow,
    Coverage,
    Singleton,
)
from fervis.lookup.question_contract.domains import (
    value_set_dependencies,
    value_population_dependencies,
)


def scoped_predicate_filter(index, requirement, *, affected_set_refs=None) -> bool:
    owner_ref = FactLocalRef.from_token(requirement.owner_expression_ref)
    owner = index.expression_by_ref[owner_ref]
    atom = FactLocalRef.from_token(requirement.atom_ref.value_ref)
    affected = (
        set(affected_set_refs)
        if affected_set_refs is not None
        else value_set_dependencies(index, atom)
    )
    if isinstance(owner, Quantify) and owner.quantifier is not Quantifier.FORALL:
        bound = index.fact_local_ref_by_local_id[owner.over_set_ref].token
        if value_set_dependencies(index, atom) != {bound}:
            return False
        condition = owner.condition_ref
        protected = frozenset((owner_ref,))
    elif isinstance(owner, Aggregate) and isinstance(index.result_grain, Singleton):
        if owner.filter_ref is None:
            return False
        condition = owner.filter_ref
        protected = frozenset(
            ref
            for ref, node in index.expression_by_ref.items()
            if isinstance(node, Aggregate) and node.filter_ref == condition
        )
    else:
        return False
    clauses = index.condition_qualification(condition).clauses
    if not clauses or not all(
        requirement.atom_ref in clause.atom_refs for clause in clauses
    ):
        return False
    for protected_ref in protected:
        for ref in index.transitive_dependencies_by_ref.get(protected_ref, ()):
            if ref in protected:
                continue
            node = index.expression_by_ref.get(ref)
            if isinstance(node, (Aggregate, Quantify, RelatedRow, Coverage)):
                closed_populations = value_population_dependencies(
                    index, ref
                ) - value_set_dependencies(index, ref)
                if affected & closed_populations:
                    return False

    fact = index.requested_fact
    roots = (
        *(
            (index.subject_obligation.subject_set_ref,)
            if not isinstance(index.result_grain, Singleton)
            else ()
        ),
        *index.grouping_refs,
        *index.ordering_refs,
        *(item.value_ref for item in index.output_requirements),
        *(index.fact_local_ref_by_local_id[ref] for ref in fact.distinct_by),
        *(
            (index.fact_local_ref_by_local_id[fact.qualification_ref],)
            if fact.qualification_ref is not None
            else ()
        ),
    )
    return not any(
        affected & value_population_dependencies(index, root, excluding=protected)
        for root in roots
    )
