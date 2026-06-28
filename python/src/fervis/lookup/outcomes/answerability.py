"""Classify requested facts that are structurally unanswerable before execution."""

from __future__ import annotations

from fervis.lookup.outcomes.model import (
    BlockedRequirement,
    BlockedRequirementField,
    BlockedRequirementKind,
    FactResult,
    Impossible,
)
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    PlanImpossible,
)
from fervis.lookup.fact_plan.row_sources import (
    read_evidence_ref,
    read_field_evidence_ref,
)
from fervis.lookup.question_contract import QuestionContract, RequestedFact


def classify_plan_impossible(
    outcome: PlanImpossible,
    *,
    question_contract: QuestionContract,
) -> FactResult | None:
    if not outcome.blocked_facts:
        return None
    requested = {fact.id: fact for fact in question_contract.requested_facts}
    blocked = [
        _blocked_requirement(
            item,
            requested[item.requested_fact_id],
        )
        for item in outcome.blocked_facts
        if item.requested_fact_id in requested
    ]
    return FactResult(
        outcome=Impossible(
            blocked_requirements=tuple(blocked),
            proof_refs=tuple(
                dict.fromkeys(
                    ref for requirement in blocked for ref in requirement.proof_refs
                )
            ),
        )
    )


def _blocked_requirement(
    item: BlockedFact,
    requested: RequestedFact,
) -> BlockedRequirement:
    return BlockedRequirement(
        id=f"blocked:{requested.id}",
        kind=_blocked_requirement_kind(item),
        requested_fact_id=requested.id,
        fact_ref=requested.id,
        required_for=requested.required_for or requested.description,
        reviewed_read_ids=item.reviewed_read_ids,
        nearest_fields=tuple(
            BlockedRequirementField(
                read_id=field.read_id,
                field_id=field.field_id,
            )
            for field in item.nearest_fields
        ),
        proof_refs=_blocked_proof_refs(
            item,
        ),
    )


def _blocked_requirement_kind(
    item: BlockedFact,
) -> BlockedRequirementKind:
    if item.basis == BlockedFactBasis.POLICY_ACCESS:
        return BlockedRequirementKind.POLICY
    if not item.reviewed_read_ids and not item.nearest_fields:
        return BlockedRequirementKind.OPERATION_NOT_SUPPORTED_BY_CATALOG
    return BlockedRequirementKind.FIELD


def _blocked_proof_refs(
    item: BlockedFact,
) -> tuple[str, ...]:
    refs: list[str] = []
    refs.extend(item.evidence_refs)
    for read_id in item.reviewed_read_ids:
        refs.append(read_evidence_ref(read_id))
    for field in item.nearest_fields:
        refs.append(
            read_field_evidence_ref(
                read_id=field.read_id,
                field_id=field.field_id,
            )
        )
    return tuple(dict.fromkeys(refs))
