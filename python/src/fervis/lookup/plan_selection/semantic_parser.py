"""Parse source assessments and derive the current-run source strategy."""

from __future__ import annotations

from fervis.lookup.plan_selection import semantic_provider_contract as output
from fervis.lookup.plan_selection.semantic import (
    CandidateSourceStrategy,
    SemanticPlanSelectionRequest,
    SourceAlignment,
    SourceAlignmentAssessment,
)
from fervis.lookup.plan_selection.strategy_derivation import (
    derive_source_strategy_branches,
    strategy_basis,
)


def parse_semantic_plan_selection(
    payload: dict[str, object],
    *,
    request: SemanticPlanSelectionRequest,
) -> tuple[CandidateSourceStrategy, ...]:
    parsed = output.SemanticPlanSelectionOutput.parse(payload)
    indexes = {index.requested_fact_id: index for index in request.indexes}
    reviews = parsed.source_assessments_by_requested_fact
    if set(reviews) != set(indexes):
        raise ValueError("plan selection must cover every requested fact")

    strategies: list[CandidateSourceStrategy] = []
    for requested_fact_id, index in indexes.items():
        assessments = _parse_assessments(
            reviews[requested_fact_id],
            requested_fact_id=requested_fact_id,
            request=request,
        )
        branches = derive_source_strategy_branches(
            requested_fact_id=requested_fact_id,
            assessments=assessments,
            source_catalog=request.source_catalog,
            qualification_clause_refs=tuple(
                clause.clause_ref for clause in index.qualification.clauses
            ),
        )
        strategies.append(
            CandidateSourceStrategy(
                requested_fact_id=requested_fact_id,
                source_assessments=assessments,
                strategy_basis=strategy_basis(assessments, branches),
                branches=branches,
            )
        )
    return tuple(strategies)


def _parse_assessments(
    values: dict[str, output.SourceAlignmentAssessmentOutput],
    *,
    requested_fact_id: str,
    request: SemanticPlanSelectionRequest,
) -> tuple[SourceAlignmentAssessment, ...]:
    source_refs = tuple(source.id for source in request.source_catalog.sources)
    if set(values) != set(source_refs):
        raise ValueError("plan selection must assess every shown source once")
    assessments: list[SourceAlignmentAssessment] = []
    for source_ref in source_refs:
        value = values[source_ref]
        alignment = SourceAlignment(value.alignment)
        if alignment not in request.allowed_alignments(
            requested_fact_id=requested_fact_id,
            source_ref=source_ref,
        ):
            raise ValueError("source alignment contradicts certified identity authority")
        assessments.append(
            SourceAlignmentAssessment(
                source_ref=source_ref,
                basis=_required_text(value.basis),
                alignment=alignment,
            )
        )
    return tuple(assessments)


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("plan selection requires non-empty assessment text")
    return text


__all__ = ["parse_semantic_plan_selection"]
