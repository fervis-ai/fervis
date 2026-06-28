"""Shared fulfillment-evidence role rules for fact planning."""

from __future__ import annotations

from typing import Any, Callable

from fervis.lookup.source_binding import BoundSource, SourceFulfillment


def value_evidence_ids_for_plan(
    fulfillment: SourceFulfillment | Any,
) -> tuple[str, ...]:
    """Evidence that can satisfy the answer value/measure for a plan.

    Metric evidence is preferred because aggregate and ranked plans must not
    substitute identity evidence for a measured value.
    """

    metric_ids = tuple(getattr(fulfillment, "metric_measure_evidence_ids", ()) or ())
    if metric_ids:
        return metric_ids
    count_ids = tuple(getattr(fulfillment, "row_count_basis_evidence_ids", ()) or ())
    if count_ids:
        return count_ids
    return tuple(getattr(fulfillment, "group_key_evidence_ids", ()) or ())


def required_fulfillment_evidence_ids(
    fulfillment: SourceFulfillment | Any,
    *,
    plan_shape: str,
) -> tuple[str, ...]:
    """Evidence the fact plan must use for the selected operation shape."""

    metric_ids = tuple(getattr(fulfillment, "metric_measure_evidence_ids", ()) or ())
    count_ids = tuple(getattr(fulfillment, "row_count_basis_evidence_ids", ()) or ())
    group_ids = group_key_evidence_ids(fulfillment)
    if plan_shape in {"aggregate_by_group", "ranked_aggregate"}:
        value_ids = metric_ids or count_ids
    elif plan_shape == "aggregate_scalar":
        value_ids = metric_ids or count_ids
    else:
        value_ids = metric_ids or group_ids
    return tuple(
        dict.fromkeys(
            (
                *value_ids,
                *(
                    group_ids
                    if plan_shape in {"aggregate_by_group", "ranked_aggregate"}
                    else ()
                ),
            )
        )
    )


def group_key_evidence_ids(fulfillment: SourceFulfillment | Any) -> tuple[str, ...]:
    return tuple(getattr(fulfillment, "group_key_evidence_ids", ()) or ())


def row_count_basis_evidence_ids(
    fulfillment: SourceFulfillment | Any,
) -> tuple[str, ...]:
    return tuple(getattr(fulfillment, "row_count_basis_evidence_ids", ()) or ())


def field_ids_by_answer_output_from_evidence(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
    evidence_ids_by_fulfillment: Callable[[SourceFulfillment], tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    field_id_by_evidence_id = source_field_id_by_evidence_id(source)
    cardinality_by_evidence_id = source_cardinality_by_evidence_id(source)
    available_field_ids = set(source.available_field_ids)
    output: dict[str, tuple[str, ...]] = {}
    for fulfillment in source.fulfillments:
        if fulfillment.requested_fact_id != requested_fact_id:
            continue
        for evidence_id in evidence_ids_by_fulfillment(fulfillment):
            if not evidence_is_compatible_with_plan_shape(
                cardinality_by_evidence_id.get(evidence_id, ""),
                plan_shape=plan_shape,
            ):
                continue
            field_id = field_id_for_fulfillment_evidence(
                evidence_id,
                field_id_by_evidence_id=field_id_by_evidence_id,
                available_field_ids=available_field_ids,
            )
            if not field_id:
                continue
            output.setdefault(fulfillment.answer_output_id, ())
            output[fulfillment.answer_output_id] = (
                *output[fulfillment.answer_output_id],
                field_id,
            )
    return output


def source_field_id_by_evidence_id(source: BoundSource) -> dict[str, str]:
    return {
        item.evidence_id: item.field_id
        for item in source.evidence_items
        if item.field_id
    }


def source_cardinality_by_evidence_id(source: BoundSource) -> dict[str, str]:
    return {
        item.evidence_id: item.row_cardinality
        for item in source.evidence_items
        if item.row_cardinality
    }


def field_id_for_fulfillment_evidence(
    evidence_id: str,
    *,
    field_id_by_evidence_id: dict[str, str],
    available_field_ids: set[str],
) -> str:
    field_id = field_id_by_evidence_id.get(evidence_id)
    if field_id:
        return field_id
    if evidence_id in available_field_ids:
        return evidence_id
    return ""


def evidence_is_compatible_with_plan_shape(
    row_cardinality: str,
    *,
    plan_shape: str,
) -> bool:
    if not row_cardinality:
        return True
    if plan_shape == "aggregate_scalar":
        return row_cardinality in {"one", "many"}
    if plan_shape in {
        "list_rows",
        "grouped_rows",
        "aggregate_by_group",
        "ranked_aggregate",
    }:
        return row_cardinality == "many"
    if plan_shape == "direct_field_value":
        return row_cardinality == "one"
    return True
