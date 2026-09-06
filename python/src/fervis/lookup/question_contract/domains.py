"""Free logical row sets of a verified semantic expression."""

from fervis.lookup.question_contract.model import (
    SetTerm,
    FactTerm,
    AssociationTerm,
    Aggregate,
    Quantify,
    RelatedRow,
    Coverage,
    FactLocalRef,
)
from fervis.lookup.question_contract.analysis import RowDomain, Singleton
from fervis.lookup.semantic_types import IdentifierType


def value_set_dependencies(index, ref: FactLocalRef | str) -> set[str]:
    if isinstance(ref, str):
        return set()
    term = index.term_by_ref.get(ref)
    if isinstance(term, SetTerm):
        return {ref.token}
    if isinstance(term, FactTerm):
        owner = index.fact_local_ref_by_local_id[term.owner_ref]
        result = value_set_dependencies(index, owner)
        if isinstance(term.value_type, IdentifierType):
            result.add(index.fact_local_ref_by_local_id[term.value_type.set_ref].token)
        return result
    if isinstance(term, AssociationTerm):
        return {
            index.fact_local_ref_by_local_id[item].token
            for item in (term.from_set_ref, term.to_set_ref)
        }
    node = index.expression_by_ref.get(ref)
    if isinstance(node, Aggregate) and isinstance(
        domain := index.evaluation_domain_by_ref[ref], RowDomain
    ):
        return {domain.owner_ref.token}
    if isinstance(node, Aggregate) and isinstance(index.result_grain, Singleton):
        # Whole-population aggregates consume their own row domains.
        return set()
    if isinstance(node, (Quantify, RelatedRow, Coverage)):
        return relational_free_sets(index, node)
    return set().union(
        *(
            value_set_dependencies(index, child)
            for child in index.direct_dependencies_by_ref.get(ref, ())
        )
    )


def value_population_dependencies(index, ref: FactLocalRef | str) -> set[str]:
    """All consumed populations, including those bound inside scalar operators."""
    if isinstance(ref, str):
        return set()
    if ref in index.term_by_ref:
        return value_set_dependencies(index, ref)
    node = index.expression_by_ref.get(ref)
    populations = set()
    if isinstance(node, (Quantify, RelatedRow)):
        populations.add(index.fact_local_ref_by_local_id[
            node.over_set_ref if isinstance(node, Quantify) else node.set_ref
        ].token)
        for association in node.association_refs:
            populations.update(value_set_dependencies(index, index.fact_local_ref_by_local_id[association]))
    if isinstance(node, Coverage):
        populations.update(index.fact_local_ref_by_local_id[item].token for item in (
            node.candidate_set_ref, node.required_dimension_set_ref, node.observation_set_ref
        ))
    return populations | set().union(*(
        value_population_dependencies(index, child)
        for child in index.direct_dependencies_by_ref.get(ref, ())
    ))


def relational_free_sets(index, node: Quantify | RelatedRow | Coverage) -> set[str]:
    if isinstance(node, Coverage):
        return {index.fact_local_ref_by_local_id[node.candidate_set_ref].token}
    over = node.over_set_ref if isinstance(node, Quantify) else node.set_ref
    bound = {index.fact_local_ref_by_local_id[over].token}
    starts, ends = set(), set()
    for ref in node.association_refs:
        association = index.term_by_ref[index.fact_local_ref_by_local_id[ref]]
        assert isinstance(association, AssociationTerm)
        starts.add(index.fact_local_ref_by_local_id[association.from_set_ref].token)
        ends.add(index.fact_local_ref_by_local_id[association.to_set_ref].token)
    bound.update(ends)
    free = (
        set()
        if node.condition_ref is None
        else value_set_dependencies(
            index, index.fact_local_ref_by_local_id[node.condition_ref]
        )
    )
    return (free - bound) | (starts - ends)
