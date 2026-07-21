"""Parse and validate fact-local plan selections."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from fervis.lookup.relation_catalog.selection import (
    catalog_selection_evidence_ref,
)
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    PlanImpossible,
)
from fervis.lookup.fact_plan.row_sources import read_evidence_ref
from fervis.lookup.plan_selection.source_strategies import (
    source_alignment_candidates_by_fact,
    source_strategies_by_fact,
)
from fervis.lookup.operation_families.plan_selection_registry import (
    plan_selection_shape_specs_for_family,
)
from fervis.lookup.plan_selection.model import (
    SourceAlignment,
    SelectedSourceStrategy,
    PlanSelectionSet,
    PlanSelectionRequest,
    PlanSelectionResult,
    SourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.source_binding.candidates.model import SourceCandidateRegistry
from fervis.lookup.plan_selection import provider_contract as provider_output


def parse_plan_selection(
    payload: dict[str, object],
    *,
    request: PlanSelectionRequest,
) -> PlanSelectionResult:
    output = provider_output.PlanSelectionOutput.parse(payload)
    outcome = output.outcome
    kind = _required_text(outcome.kind)
    if kind != "source_alignment_reviews":
        raise ValueError(f"unsupported plan selection outcome: {kind}")
    reviews_by_fact = outcome.reviews_by_requested_fact
    fact_ids = {fact.id for fact in request.requested_facts}
    if set(reviews_by_fact) != fact_ids:
        raise ValueError("source alignment reviews must cover every requested fact")
    strategies_by_fact = source_strategies_by_fact(
        request.source_candidates,
        requested_facts=request.requested_facts,
        relation_catalog=request.relation_catalog,
        shape_specs_for_family=plan_selection_shape_specs_for_family,
    )
    candidates_by_fact = source_alignment_candidates_by_fact(strategies_by_fact)
    plans: list[SelectedSourceStrategy] = []
    blocked_facts: list[BlockedFact] = []
    for fact in request.requested_facts:
        source_candidate_ids = tuple(
            candidate.source_candidate_id
            for candidate in candidates_by_fact.get(fact.id, ())
        )
        if not source_candidate_ids:
            raw_read_ids = _raw_source_candidate_read_ids_for_fact(
                request.source_candidates,
                requested_fact_id=fact.id,
            )
            if raw_read_ids:
                blocked_facts.append(
                    _blocked_fact_from_unexecutable_candidates(
                        requested_fact_id=fact.id,
                        read_ids=raw_read_ids,
                    )
                )
                continue
        source_alignment_by_id, basis_by_source_id = _source_alignment_reviews(
            reviews_by_fact[fact.id].named(provider_output.SourceAlignmentReviewOutput),
            source_candidate_ids=source_candidate_ids,
        )
        aligned_strategies = tuple(
            source_strategy
            for source_strategy in strategies_by_fact.get(fact.id, ())
            if _source_strategy_is_supported(
                source_strategy,
                source_alignment_by_id=source_alignment_by_id,
            )
        )
        aligned_strategies = _minimal_supported_strategies(aligned_strategies)
        if not aligned_strategies:
            blocked_facts.append(
                _blocked_fact_from_unaligned_reviews(
                    requested_fact_id=fact.id,
                    source_candidates=candidates_by_fact.get(fact.id, ()),
                    basis_by_source_id=basis_by_source_id,
                )
            )
            continue
        for source_strategy in aligned_strategies:
            plans.append(
                SelectedSourceStrategy(
                    plan_selection_id=(
                        f"source_alignment.{fact.id}.{source_strategy.source_strategy_id}"
                    ),
                    requested_fact_id=fact.id,
                    source_strategy_id=source_strategy.source_strategy_id,
                    plan_shape=source_strategy.plan_shape,
                    required_answer_output_ids=(
                        source_strategy.required_answer_output_ids
                    ),
                    source_members=source_strategy.source_members,
                    basis=" | ".join(
                        basis_by_source_id.get(member.source_candidate_id, "")
                        for member in source_strategy.source_members
                    ),
                )
            )
    if blocked_facts:
        if plans:
            raise ValueError("source alignment cannot mix aligned and blocked facts")
        return PlanSelectionResult(
            outcome=PlanImpossible(blocked_facts=tuple(blocked_facts))
        )
    return PlanSelectionResult(outcome=PlanSelectionSet(plan_selections=tuple(plans)))


def _source_alignment_reviews(
    reviews: dict[str, provider_output.SourceAlignmentReviewOutput],
    *,
    source_candidate_ids: tuple[str, ...],
) -> tuple[dict[str, SourceAlignment], dict[str, str]]:
    if set(reviews) != set(source_candidate_ids):
        raise ValueError("source alignment must review every shown source candidate")
    source_alignment_by_id: dict[str, SourceAlignment] = {}
    basis_by_source_id: dict[str, str] = {}
    for source_candidate_id in source_candidate_ids:
        review = reviews[source_candidate_id]
        reviewed_source_candidate_id = _required_text(review.source_candidate_id)
        if reviewed_source_candidate_id != source_candidate_id:
            raise ValueError("source alignment source_candidate_id must match key")
        basis = _required_text(review.basis)
        basis_by_source_id[source_candidate_id] = basis
        alignment = _enum(
            SourceAlignment,
            _required_text(review.source_alignment),
            "source_alignment",
        )
        source_alignment_by_id[source_candidate_id] = alignment
    return source_alignment_by_id, basis_by_source_id


def _source_strategy_is_supported(
    source_strategy: SourceStrategy,
    *,
    source_alignment_by_id: dict[str, SourceAlignment],
) -> bool:
    member_alignments = tuple(
        source_alignment_by_id[member.source_candidate_id]
        for member in source_strategy.source_members
    )
    if SourceAlignment.NOT_ALIGNED in member_alignments:
        return False
    if len(member_alignments) == 1:
        return member_alignments[0] is SourceAlignment.DIRECT
    return True


def _minimal_supported_strategies(
    strategies: tuple[SourceStrategy, ...],
) -> tuple[SourceStrategy, ...]:
    standalone = tuple(
        strategy for strategy in strategies if len(strategy.source_members) == 1
    )
    return standalone or strategies


def _blocked_fact_from_unaligned_reviews(
    *,
    requested_fact_id: str,
    source_candidates: tuple[SourceStrategyMember, ...],
    basis_by_source_id: dict[str, str],
) -> BlockedFact:
    reviewed_read_ids = tuple(
        dict.fromkeys(
            candidate.read_id for candidate in source_candidates if candidate.read_id
        )
    )
    evidence_refs = (
        tuple(read_evidence_ref(read_id) for read_id in reviewed_read_ids)
        if reviewed_read_ids
        else (catalog_selection_evidence_ref(requested_fact_id=requested_fact_id),)
    )
    return BlockedFact(
        requested_fact_id=requested_fact_id,
        basis=BlockedFactBasis.CATALOG_ACCESS,
        evidence_refs=evidence_refs,
        reviewed_read_ids=reviewed_read_ids,
        nearest_fields=(),
        explanation=" | ".join(
            basis_by_source_id.get(candidate.source_candidate_id, "")
            for candidate in source_candidates
            if basis_by_source_id.get(candidate.source_candidate_id, "")
        ),
    )


def _raw_source_candidate_read_ids_for_fact(
    catalog: SourceCandidateRegistry,
    *,
    requested_fact_id: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            candidate.read_id
            for candidate in catalog.candidates_for(requested_fact_id)
            if candidate.read_id
        )
    )


def _blocked_fact_from_unexecutable_candidates(
    *,
    requested_fact_id: str,
    read_ids: tuple[str, ...],
) -> BlockedFact:
    return BlockedFact(
        requested_fact_id=requested_fact_id,
        basis=BlockedFactBasis.CATALOG_ACCESS,
        evidence_refs=tuple(read_evidence_ref(read_id) for read_id in read_ids),
        reviewed_read_ids=read_ids,
        nearest_fields=(),
        explanation="Selected reads expose no executable evidence for the answer.",
    )


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("plan selection requires non-empty text")
    return text


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum(enum_type: type[_EnumT], value: str, path: str) -> _EnumT:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{path} has invalid value") from exc
