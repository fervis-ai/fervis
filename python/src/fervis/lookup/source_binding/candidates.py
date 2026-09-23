"""Mechanical availability of source scopes, without semantic alignment guesses."""

from dataclasses import replace

from fervis.lookup.available_sources import AvailableSourceCatalog
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.question_contract import RequestedFactSemanticIndex, SetTerm
from fervis.lookup.source_binding.model import (
    CandidateSourceStrategy,
    SourceStrategyBranch,
    SemanticSourceBindingRequest,
)


def candidate_source_strategy(
    index: RequestedFactSemanticIndex,
    catalog: AvailableSourceCatalog,
    canonical_values: tuple[CanonicalInputValue, ...],
) -> CandidateSourceStrategy:
    strategy = CandidateSourceStrategy(
        index.requested_fact_id,
        (
            SourceStrategyBranch(
                branch_id=f"{index.requested_fact_id}:source_branch:1",
                source_refs=tuple(source.id for source in catalog.sources),
                relation_evidence_refs=tuple(
                    edge.evidence_ref for edge in catalog.relation_evidence
                ),
                qualification_clause_refs=tuple(
                    clause.clause_ref for clause in index.qualification.clauses
                ),
            ),
        ),
    )
    request = SemanticSourceBindingRequest(index, strategy, catalog, canonical_values)
    feasible = bool(catalog.sources) and all(
        request.row_references_for_set(ref.token)
        for ref in index.source_requirement_refs
        if isinstance(index.term_by_ref[ref], SetTerm)
    )
    return strategy if feasible else replace(strategy, branches=())
