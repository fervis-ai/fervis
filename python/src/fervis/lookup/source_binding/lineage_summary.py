"""Lineage summaries for source-binding model output."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.plan_targets import (
    source_binding_fact_id_from_field,
)

from fervis.lineage.step_summary import (
    StepSummaryDetail,
    StepSummaryItem,
    step_summary_json,
)


def source_binding_step_summary(payload: dict[str, Any]) -> dict[str, object]:
    outcome = _source_binding_outcome(payload)
    return step_summary_json(
        *_metric_review_items(outcome),
        *_decision_basis_items(outcome),
    )


def _source_binding_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    outcome = _dict_or_empty(payload.get("outcome"))
    return outcome or payload


def _metric_review_items(payload: dict[str, Any]) -> tuple[StepSummaryItem, ...]:
    bases_by_fact = _dict_or_empty(payload.get("metric_fit_bases"))
    interpretations_by_fact = _dict_or_empty(payload.get("fit_basis_interpretations"))
    items: list[StepSummaryItem] = []
    for requested_fact_id, raw_bases in bases_by_fact.items():
        bases = _dict_or_empty(raw_bases)
        interpretations = _dict_or_empty(interpretations_by_fact.get(requested_fact_id))
        items.extend(
            _metric_review_items_for_fact(
                requested_fact_id=str(requested_fact_id),
                bases=bases,
                interpretations=interpretations,
            )
        )
    return tuple(items)


def _metric_review_items_for_fact(
    *,
    requested_fact_id: str,
    bases: dict[str, Any],
    interpretations: dict[str, Any],
) -> tuple[StepSummaryItem, ...]:
    items: list[StepSummaryItem] = []
    for evidence_id, raw_basis in bases.items():
        evidence_id = str(evidence_id)
        basis = _dict_or_empty(raw_basis)
        interpretation = _dict_or_empty(interpretations.get(evidence_id))
        fit_basis = str(basis.get("fit_basis") or "")
        decision = str(interpretation.get("interpretation") or "")
        if not evidence_id or not fit_basis or not decision:
            continue
        items.append(
            StepSummaryItem(
                text=f"{evidence_id}: {fit_basis} -> {decision}",
                is_explanation=True,
                path=(
                    "outcome",
                    "metric_fit_bases",
                    requested_fact_id,
                    evidence_id,
                    "fit_basis",
                ),
            )
        )
    return tuple(items)


def _decision_basis_items(payload: dict[str, Any]) -> tuple[StepSummaryItem, ...]:
    return tuple(
        item
        for field_id, raw_fact_binding in payload.items()
        if source_binding_fact_id_from_field(field_id) is not None
        for role_id, invocation in _dict_or_empty(raw_fact_binding).items()
        if role_id != "plan_shape"
        if isinstance(invocation, dict)
        for item in _invocation_decision_items(invocation)
    )


def _invocation_decision_items(
    invocation: dict[str, Any],
) -> tuple[StepSummaryItem, ...]:
    items: list[StepSummaryItem] = []
    binding_target = str(invocation.get("binding_target_id") or "")
    if binding_target:
        items.append(
            StepSummaryItem(
                text=f"Source binding {binding_target}",
                detail=StepSummaryDetail.VERBOSE,
            )
        )
    if item := _population_basis_item(invocation.get("answer_population")):
        items.append(item)
    items.extend(_fulfillment_basis_items(invocation.get("fulfillment_decisions")))
    items.extend(_param_basis_items(invocation.get("param_decisions")))
    return tuple(items)


def _population_basis_item(value: object) -> StepSummaryItem | None:
    population = _dict_or_empty(value)
    basis = str(population.get("match_basis_explanation") or "")
    if not basis:
        return None
    return StepSummaryItem(
        text=f"Population basis: {basis}",
        detail=StepSummaryDetail.VERBOSE,
        is_explanation=True,
        path=("answer_population", "match_basis_explanation"),
    )


def _fulfillment_basis_items(value: object) -> tuple[StepSummaryItem, ...]:
    return tuple(
        item
        for answer_output_id, raw in _dict_or_empty(value).items()
        if (
            item := _basis_item(
                label="Fulfillment basis",
                target_parts=(
                    str(answer_output_id),
                    str(_dict_or_empty(raw).get("fulfillment_choice_id") or ""),
                ),
                basis=str(_dict_or_empty(raw).get("match_basis_explanation") or ""),
                path=(
                    "fulfillment_decisions",
                    str(answer_output_id),
                    "match_basis_explanation",
                ),
            )
        )
    )


def _param_basis_items(value: object) -> tuple[StepSummaryItem, ...]:
    return tuple(
        item
        for param_id, raw in _dict_or_empty(value).items()
        if (
            item := _basis_item(
                label="Param basis",
                target_parts=(
                    str(param_id),
                    str(_dict_or_empty(raw).get("param_decision_id") or ""),
                ),
                basis=str(_dict_or_empty(raw).get("match_basis_explanation") or ""),
                path=("param_decisions", str(param_id), "match_basis_explanation"),
            )
        )
    )


def _basis_item(
    *,
    label: str,
    target_parts: tuple[str, str],
    basis: str,
    path: tuple[str, ...],
) -> StepSummaryItem | None:
    if not basis:
        return None
    target = _joined_target(*target_parts)
    return StepSummaryItem(
        text=f"{label} {target}: {basis}" if target else f"{label}: {basis}",
        detail=StepSummaryDetail.VERBOSE,
        is_explanation=True,
        path=path,
    )


def _joined_target(first: str, second: str) -> str:
    if first and second:
        return f"{first}/{second}"
    return first or second


def _dict_or_empty(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _list_of_dicts(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))
