"""Parse and validate fact-local plan selections."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

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
)


def parse_plan_selection(
    payload: dict[str, Any],
    *,
    request: PlanSelectionRequest,
) -> PlanSelectionResult:
    outcome = _dict(payload.get("outcome"), "outcome")
    kind = _text(outcome.get("kind"))
    if kind != "source_alignment_reviews":
        raise ValueError(f"unsupported plan selection outcome: {kind}")
    raw_reviews = _dict(
        outcome.get("reviews_by_requested_fact"),
        "reviews_by_requested_fact",
    )
    fact_ids = {fact.id for fact in request.requested_facts}
    if set(raw_reviews) != fact_ids:
        raise ValueError("source alignment reviews must cover every requested fact")
    strategies_by_fact = source_strategies_by_fact(
        request.source_candidate_payload,
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
        if not source_candidate_ids and _raw_source_candidate_ids_for_fact(
            request.source_candidate_payload,
            requested_fact_id=fact.id,
        ):
            raise ValueError("source alignment produced no executable source strategy")
        aligned_source_ids, basis_by_source_id = _aligned_source_reviews(
            _dict(raw_reviews.get(fact.id), f"reviews_by_requested_fact.{fact.id}"),
            source_candidate_ids=source_candidate_ids,
        )
        if not aligned_source_ids:
            blocked_facts.append(
                _blocked_fact_from_unaligned_reviews(
                    requested_fact_id=fact.id,
                    source_candidates=candidates_by_fact.get(fact.id, ()),
                    basis_by_source_id=basis_by_source_id,
                )
            )
            continue
        aligned_strategies = tuple(
            source_strategy
            for source_strategy in strategies_by_fact.get(fact.id, ())
            if all(
                member.source_candidate_id in aligned_source_ids
                for member in source_strategy.source_members
            )
        )
        if not aligned_strategies:
            raise ValueError("source alignment produced no executable source strategy")
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
                    source_members=tuple(
                        replace(member, fulfillment_support_set_ids=())
                        for member in source_strategy.source_members
                    ),
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


def _aligned_source_reviews(
    raw_reviews: dict[str, Any],
    *,
    source_candidate_ids: tuple[str, ...],
) -> tuple[set[str], dict[str, str]]:
    if set(raw_reviews) != set(source_candidate_ids):
        raise ValueError("source alignment must review every shown source candidate")
    aligned_source_ids: set[str] = set()
    basis_by_source_id: dict[str, str] = {}
    for source_candidate_id in source_candidate_ids:
        raw = _dict(raw_reviews.get(source_candidate_id), source_candidate_id)
        reviewed_source_candidate_id = _text(raw.get("source_candidate_id"))
        if reviewed_source_candidate_id != source_candidate_id:
            raise ValueError("source alignment source_candidate_id must match key")
        basis = _text(raw.get("basis"))
        basis_by_source_id[source_candidate_id] = basis
        alignment = _enum(
            SourceAlignment,
            raw.get("source_alignment"),
            "source_alignment",
        )
        if alignment in {
            SourceAlignment.DIRECT,
            SourceAlignment.PARTIAL,
        }:
            aligned_source_ids.add(source_candidate_id)
    return aligned_source_ids, basis_by_source_id


def _blocked_fact_from_unaligned_reviews(
    *,
    requested_fact_id: str,
    source_candidates: tuple[Any, ...],
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


def _raw_source_candidate_ids_for_fact(
    payload: dict[str, Any],
    *,
    requested_fact_id: str,
) -> tuple[str, ...]:
    output: list[str] = []
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        if str(fact_sources.get("requested_fact_id") or "") != requested_fact_id:
            continue
        for context in fact_sources.get("source_contexts") or ():
            if not isinstance(context, dict):
                continue
            output.extend(
                str(candidate.get("source_candidate_id") or "").strip()
                for candidate in context.get("source_options") or ()
                if isinstance(candidate, dict)
                and str(candidate.get("source_candidate_id") or "").strip()
            )
    return tuple(dict.fromkeys(output))


def _dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("expected array of objects")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("expected array of objects")
    return tuple(value)


def _strings(value: Any, path: str = "value") -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    output = tuple(_text(item) for item in value)
    if not all(output):
        raise ValueError(f"{path} must contain non-empty strings")
    return output


def _enum(enum_type: Any, value: Any, path: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{path} has invalid value") from exc


def _text(value: Any) -> str:
    return str(value or "").strip()
