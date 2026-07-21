"""Source fulfillment parsing and evidence selection."""

from __future__ import annotations

from typing import Literal
from typing_extensions import assert_never

from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.candidates.contracts import (
    EvidenceItem,
    EntityEvidence,
    FulfillmentSlot,
    FulfillmentSupportSet,
)
from fervis.lookup.source_binding.model import (
    SourceEvidenceItem,
    SourceFulfillment,
)
from fervis.lookup.source_binding.parser.candidate_access import (
    candidate_evidence_ids,
    candidate_metric_measure_evidence_ids,
    candidate_row_count_basis_evidence_ids,
)
from fervis.lookup.source_binding.parser.metric_fit import (
    candidate_fitting_metric_measure_evidence_ids,
    candidate_fitting_row_count_basis_evidence_ids,
    fitting_metric_measure_evidence_ids,
    fitting_row_count_basis_evidence_ids,
    plan_shape_uses_row_count_as_metric,
    source_metric_fit_bases,
)
from fervis.lookup.source_binding.parser_common import _text


__all__ = [
    "fulfillment_row_source_id",
    "parse_source_fulfillments",
]

MetricFitReviews = dict[str, dict[str, dict[str, str]]]


def fulfillment_row_source_id(
    fulfillments: tuple[SourceFulfillment, ...],
    *,
    evidence_items: tuple[SourceEvidenceItem, ...],
) -> str:
    field_evidence_ids = {
        evidence_id
        for fulfillment in fulfillments
        for evidence_id in fulfillment.field_evidence_ids()
    }
    row_source_ids = {
        item.row_source_id
        for item in evidence_items
        if item.evidence_id in field_evidence_ids and item.row_source_id
    }
    row_source_ids.update(
        fulfillment.entity_evidence.row_source_id
        for fulfillment in fulfillments
        if fulfillment.entity_evidence is not None
    )
    if len(row_source_ids) != 1:
        return ""
    return next(iter(row_source_ids))


def parse_source_fulfillments(
    raw_fulfillment_decisions: dict[str, provider_output.FulfillmentDecisionOutput],
    *,
    requested_fact_id: str,
    answer_output_ids: set[str],
    required_answer_output_ids: set[str],
    metric_answer_output_ids: set[str],
    candidate: SourceCandidate,
    plan_shape: str,
    metric_fit_reviews_by_requested_output: MetricFitReviews,
) -> tuple[SourceFulfillment, ...]:
    output: list[SourceFulfillment] = []
    seen_support_set_ids: set[str] = set()
    raw_decisions = raw_fulfillment_decisions
    if not answer_output_ids:
        if raw_decisions:
            raise ValueError("binding target does not allow answer fulfillment")
        return ()
    if set(raw_decisions) - answer_output_ids:
        raise ValueError("source fulfillment references unknown answer output")
    for answer_output_id, raw_value in raw_decisions.items():
        output.append(
            _parse_source_fulfillment_decision(
                raw_value,
                requested_fact_id=requested_fact_id,
                answer_output_id=answer_output_id,
                candidate=candidate,
                plan_shape=plan_shape,
                metric_fit_reviews_by_requested_output=(
                    metric_fit_reviews_by_requested_output
                ),
            )
        )
    for fulfillment in output:
        if fulfillment.fulfillment_support_set_id in seen_support_set_ids:
            raise ValueError("duplicate source fulfillment support set")
        seen_support_set_ids.add(fulfillment.fulfillment_support_set_id)
    output.extend(
        _derived_metric_fulfillments(
            requested_fact_id=requested_fact_id,
            missing_answer_output_ids=(
                required_answer_output_ids
                - {fulfillment.answer_output_id for fulfillment in output}
            ),
            metric_answer_output_ids=metric_answer_output_ids,
            candidate=candidate,
            plan_shape=plan_shape,
            metric_fit_reviews_by_requested_output=metric_fit_reviews_by_requested_output,
        )
    )
    missing = required_answer_output_ids - {
        fulfillment.answer_output_id for fulfillment in output
    }
    if missing:
        raise ValueError("fulfillment_decisions must cover required answer outputs")
    return tuple(output)


def _parse_source_fulfillment_decision(
    raw: provider_output.FulfillmentDecisionOutput,
    *,
    requested_fact_id: str,
    answer_output_id: str,
    candidate: SourceCandidate,
    plan_shape: str,
    metric_fit_reviews_by_requested_output: MetricFitReviews,
) -> SourceFulfillment:
    support_set_id = _source_fulfillment_support_set_id(
        _text(raw.fulfillment_choice_id),
        answer_output_id=answer_output_id,
        candidate=candidate,
    )
    slots = _source_fulfillment_support_set_slots(
        support_set_id,
        answer_output_id=answer_output_id,
        candidate=candidate,
    )
    selected_metric_measure_evidence_ids = tuple(
        dict.fromkeys(_slot_evidence_ids(slots, key="metric_measure_evidence"))
    )
    selected_value_evidence_ids = tuple(
        dict.fromkeys(_slot_evidence_ids(slots, key="value_evidence"))
    )
    selected_row_count_basis_evidence_ids = tuple(
        dict.fromkeys(_slot_evidence_ids(slots, key="row_count_basis_evidence"))
    )
    selected_metric_measure_evidence_ids = fitting_metric_measure_evidence_ids(
        requested_fact_id=requested_fact_id,
        answer_output_id=answer_output_id,
        selected_metric_measure_evidence_ids=selected_metric_measure_evidence_ids,
        metric_fit_reviews_by_requested_output=metric_fit_reviews_by_requested_output,
    )
    if plan_shape_uses_row_count_as_metric(plan_shape):
        selected_row_count_basis_evidence_ids = fitting_row_count_basis_evidence_ids(
            requested_fact_id=requested_fact_id,
            answer_output_id=answer_output_id,
            selected_row_count_basis_evidence_ids=selected_row_count_basis_evidence_ids,
            metric_fit_reviews_by_requested_output=metric_fit_reviews_by_requested_output,
        )
    if plan_shape == "aggregate_by_group":
        selected_metric_measure_evidence_ids = tuple(
            dict.fromkeys(
                (
                    *selected_metric_measure_evidence_ids,
                    *candidate_fitting_metric_measure_evidence_ids(
                        requested_fact_id=requested_fact_id,
                        answer_output_id=answer_output_id,
                        candidate_metric_measure_evidence_ids=(
                            candidate_metric_measure_evidence_ids(candidate)
                        ),
                        metric_fit_reviews_by_requested_output=(
                            metric_fit_reviews_by_requested_output
                        ),
                    ),
                )
            )
        )
        selected_row_count_basis_evidence_ids = tuple(
            dict.fromkeys(
                (
                    *selected_row_count_basis_evidence_ids,
                    *candidate_fitting_row_count_basis_evidence_ids(
                        requested_fact_id=requested_fact_id,
                        answer_output_id=answer_output_id,
                        candidate_row_count_basis_evidence_ids=(
                            candidate_row_count_basis_evidence_ids(candidate)
                        ),
                        metric_fit_reviews_by_requested_output=(
                            metric_fit_reviews_by_requested_output
                        ),
                    ),
                )
            )
        )
    entity_evidence = _slot_entity_evidence(slots)
    return SourceFulfillment(
        requested_fact_id=requested_fact_id,
        answer_output_id=answer_output_id,
        match_basis_explanation=_text(raw.match_basis_explanation),
        fulfillment_support_set_id=support_set_id,
        entity_evidence=entity_evidence,
        value_evidence_ids=selected_value_evidence_ids,
        metric_measure_evidence_ids=selected_metric_measure_evidence_ids,
        row_count_basis_evidence_ids=selected_row_count_basis_evidence_ids,
        metric_fit_bases=source_metric_fit_bases(
            requested_fact_id=requested_fact_id,
            answer_output_id=answer_output_id,
            evidence_ids=(
                *selected_metric_measure_evidence_ids,
                *selected_row_count_basis_evidence_ids,
            ),
            metric_fit_reviews_by_requested_output=metric_fit_reviews_by_requested_output,
        ),
    )


def _derived_metric_fulfillments(
    *,
    requested_fact_id: str,
    missing_answer_output_ids: set[str],
    metric_answer_output_ids: set[str],
    candidate: SourceCandidate,
    plan_shape: str,
    metric_fit_reviews_by_requested_output: MetricFitReviews,
) -> tuple[SourceFulfillment, ...]:
    return tuple(
        fulfillment
        for answer_output_id in sorted(
            missing_answer_output_ids & metric_answer_output_ids
        )
        for fulfillment in (
            _derived_metric_fulfillment(
                requested_fact_id=requested_fact_id,
                answer_output_id=answer_output_id,
                candidate=candidate,
                plan_shape=plan_shape,
                metric_fit_reviews_by_requested_output=(
                    metric_fit_reviews_by_requested_output
                ),
            ),
        )
        if fulfillment is not None
    )


def _derived_metric_fulfillment(
    *,
    requested_fact_id: str,
    answer_output_id: str,
    candidate: SourceCandidate,
    plan_shape: str,
    metric_fit_reviews_by_requested_output: MetricFitReviews,
) -> SourceFulfillment | None:
    if _candidate_has_model_selectable_fulfillment(
        candidate,
        answer_output_id=answer_output_id,
    ):
        return None
    selected_metric_measure_evidence_ids = candidate_fitting_metric_measure_evidence_ids(
        requested_fact_id=requested_fact_id,
        answer_output_id=answer_output_id,
        candidate_metric_measure_evidence_ids=(
            candidate_metric_measure_evidence_ids(candidate)
        ),
        metric_fit_reviews_by_requested_output=metric_fit_reviews_by_requested_output,
    )
    selected_row_count_basis_evidence_ids: tuple[str, ...] = ()
    if plan_shape_uses_row_count_as_metric(plan_shape):
        selected_row_count_basis_evidence_ids = (
            candidate_fitting_row_count_basis_evidence_ids(
                requested_fact_id=requested_fact_id,
                answer_output_id=answer_output_id,
                candidate_row_count_basis_evidence_ids=(
                    candidate_row_count_basis_evidence_ids(candidate)
                ),
                metric_fit_reviews_by_requested_output=(
                    metric_fit_reviews_by_requested_output
                ),
            )
        )
    evidence_ids = (
        *selected_metric_measure_evidence_ids,
        *selected_row_count_basis_evidence_ids,
    )
    if not evidence_ids:
        return None
    return SourceFulfillment(
        requested_fact_id=requested_fact_id,
        answer_output_id=answer_output_id,
        match_basis_explanation=(
            "Required metric output is satisfied by fitting source metric evidence."
        ),
        metric_measure_evidence_ids=selected_metric_measure_evidence_ids,
        row_count_basis_evidence_ids=selected_row_count_basis_evidence_ids,
        metric_fit_bases=source_metric_fit_bases(
            requested_fact_id=requested_fact_id,
            answer_output_id=answer_output_id,
            evidence_ids=evidence_ids,
            metric_fit_reviews_by_requested_output=metric_fit_reviews_by_requested_output,
        ),
    )


def _candidate_has_model_selectable_fulfillment(
    candidate: SourceCandidate,
    *,
    answer_output_id: str,
) -> bool:
    return any(
        support_set.fulfillment_choice_id
        for support_set in _candidate_fulfillment_support_sets_by_id(candidate).values()
        if support_set.answer_output_id == answer_output_id
    )


def _source_fulfillment_support_set_slots(
    support_set_id: str,
    *,
    answer_output_id: str,
    candidate: SourceCandidate,
) -> tuple[FulfillmentSlot, ...]:
    support_set = _candidate_fulfillment_support_sets_by_id(candidate).get(
        support_set_id
    )
    if support_set is None:
        raise ValueError("source fulfillment references unknown support set")
    if support_set.answer_output_id != answer_output_id:
        raise ValueError("source fulfillment support set mismatches answer output")
    selected_slots = support_set.fulfillment_slots
    evidence_ids = tuple(
        evidence_id
        for key in _EVIDENCE_SLOT_KEYS
        for evidence_id in _slot_evidence_ids(selected_slots, key=key)
    )
    if not evidence_ids:
        raise ValueError("source fulfillment slot requires evidence")
    available = candidate_evidence_ids(candidate)
    all_slot_evidence = {
        evidence_id
        for key in _EVIDENCE_SLOT_KEYS
        for evidence_id in _slot_evidence_ids(selected_slots, key=key)
    }
    if all_slot_evidence - available:
        raise ValueError("source fulfillment slot references unknown evidence")
    return selected_slots


def _source_fulfillment_support_set_id(
    choice_id: str,
    *,
    answer_output_id: str,
    candidate: SourceCandidate,
) -> str:
    support_set = _candidate_fulfillment_support_sets_by_choice_id(candidate).get(
        (answer_output_id, choice_id)
    )
    if support_set is None:
        raise ValueError("source fulfillment references unknown choice")
    support_set_id = support_set.fulfillment_support_set_id
    if not support_set_id:
        raise ValueError("source fulfillment choice is missing internal support set")
    return support_set_id


_EvidenceSlotKey = Literal[
    "metric_measure_evidence",
    "value_evidence",
    "row_count_basis_evidence",
    "entity_evidence",
]
_EVIDENCE_SLOT_KEYS: tuple[_EvidenceSlotKey, ...] = (
    "metric_measure_evidence",
    "value_evidence",
    "row_count_basis_evidence",
    "entity_evidence",
)


def _slot_evidence_ids(
    slots: tuple[FulfillmentSlot, ...],
    *,
    key: _EvidenceSlotKey,
) -> tuple[str, ...]:
    return tuple(
        item.evidence_id
        for slot in slots
        for item in _slot_evidence_items(slot, key=key)
        if item.evidence_id
    )


def _slot_evidence_items(
    slot: FulfillmentSlot,
    *,
    key: _EvidenceSlotKey,
) -> tuple[EvidenceItem, ...]:
    if key == "metric_measure_evidence":
        return slot.metric_measure_evidence
    if key == "value_evidence":
        return slot.value_evidence
    if key == "row_count_basis_evidence":
        return slot.row_count_basis_evidence
    if key == "entity_evidence":
        return slot.entity_evidence
    assert_never(key)


def _slot_entity_evidence(
    slots: tuple[FulfillmentSlot, ...],
) -> EntityEvidence | None:
    items = tuple(item for slot in slots for item in slot.entity_evidence)
    if not items:
        return None
    if len(items) != 1:
        raise ValueError("source fulfillment requires one entity evidence choice")
    item = items[0]
    return item


def _candidate_fulfillment_support_sets_by_id(
    candidate: SourceCandidate,
) -> dict[str, FulfillmentSupportSet]:
    return {
        support_set_id: item
        for item in candidate.fulfillment_support_sets
        for support_set_id in (item.fulfillment_support_set_id,)
        if support_set_id
    }


def _candidate_fulfillment_support_sets_by_choice_id(
    candidate: SourceCandidate,
) -> dict[tuple[str, str], FulfillmentSupportSet]:
    return {
        (answer_output_id, choice_id): item
        for item in candidate.fulfillment_support_sets
        for answer_output_id in (item.answer_output_id,)
        for choice_id in (item.fulfillment_choice_id,)
        if answer_output_id and choice_id
    }
