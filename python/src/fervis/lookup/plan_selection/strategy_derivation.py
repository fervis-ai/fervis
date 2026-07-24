"""Deterministically derive executable source branches from alignment reviews."""

from __future__ import annotations

from fervis.lookup.available_sources import AvailableSourceCatalog
from fervis.lookup.plan_selection.semantic import (
    SourceAlignment,
    SourceAlignmentAssessment,
    SourceStrategyBranch,
)


def derive_source_strategy_branches(
    *,
    requested_fact_id: str,
    assessments: tuple[SourceAlignmentAssessment, ...],
    source_catalog: AvailableSourceCatalog,
    qualification_clause_refs: tuple[str, ...],
) -> tuple[SourceStrategyBranch, ...]:
    """Lower independent assessments without adding semantic judgments."""

    direct_ref = next(
        (
            assessment.source_ref
            for assessment in assessments
            if assessment.alignment is SourceAlignment.DIRECT
        ),
        None,
    )
    if direct_ref is not None:
        return (
            _branch(
                requested_fact_id=requested_fact_id,
                branch_number=1,
                source_refs=(direct_ref,),
                relation_evidence_refs=(),
                qualification_clause_refs=qualification_clause_refs,
            ),
        )

    partial_refs = tuple(
        assessment.source_ref
        for assessment in assessments
        if assessment.alignment is SourceAlignment.PARTIAL
    )
    components = _connected_components(
        partial_refs,
        source_catalog=source_catalog,
    )
    return tuple(
        _branch(
            requested_fact_id=requested_fact_id,
            branch_number=branch_number,
            source_refs=component,
            relation_evidence_refs=tuple(
                evidence.evidence_ref
                for evidence in source_catalog.relation_evidence
                if {
                    evidence.left_source_ref,
                    evidence.right_source_ref,
                }
                <= set(component)
            ),
            qualification_clause_refs=qualification_clause_refs,
        )
        for branch_number, component in enumerate(components, start=1)
        if len(component) > 1
    )


def strategy_basis(
    assessments: tuple[SourceAlignmentAssessment, ...],
    branches: tuple[SourceStrategyBranch, ...],
) -> str:
    selected_refs = {
        source_ref for branch in branches for source_ref in branch.source_refs
    }
    selected_bases = tuple(
        assessment.basis
        for assessment in assessments
        if assessment.source_ref in selected_refs
    )
    if selected_bases:
        return " | ".join(selected_bases)
    return "No assessed source forms a complete connected strategy."


def _connected_components(
    source_refs: tuple[str, ...],
    *,
    source_catalog: AvailableSourceCatalog,
) -> tuple[tuple[str, ...], ...]:
    allowed = set(source_refs)
    neighbors: dict[str, set[str]] = {
        source_ref: set() for source_ref in source_refs
    }
    for evidence in source_catalog.relation_evidence:
        left = evidence.left_source_ref
        right = evidence.right_source_ref
        if left in allowed and right in allowed:
            neighbors[left].add(right)
            neighbors[right].add(left)

    pending = set(source_refs)
    components: list[tuple[str, ...]] = []
    for source_ref in source_refs:
        if source_ref not in pending:
            continue
        reached = {source_ref}
        frontier = [source_ref]
        while frontier:
            current = frontier.pop()
            discovered = neighbors[current] - reached
            reached.update(discovered)
            frontier.extend(discovered)
        pending.difference_update(reached)
        components.append(tuple(item for item in source_refs if item in reached))
    return tuple(components)


def _branch(
    *,
    requested_fact_id: str,
    branch_number: int,
    source_refs: tuple[str, ...],
    relation_evidence_refs: tuple[str, ...],
    qualification_clause_refs: tuple[str, ...],
) -> SourceStrategyBranch:
    return SourceStrategyBranch(
        branch_id=f"{requested_fact_id}:source_branch:{branch_number}",
        source_refs=source_refs,
        relation_evidence_refs=relation_evidence_refs,
        qualification_clause_refs=qualification_clause_refs,
    )


__all__ = ["derive_source_strategy_branches", "strategy_basis"]
