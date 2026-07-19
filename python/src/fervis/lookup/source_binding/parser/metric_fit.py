"""Metric-fit review parsing and evidence filtering."""

from __future__ import annotations

from fervis.lookup.operation_families.source_binding_registry import (
    source_binding_metric_evidence_ids_by_requested_fact,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.parser.model import ParsedSourceBindingPlan
from fervis.lookup.source_binding.metric_fit import (
    METRIC_FIT_DECISION,
    METRIC_FIT_DECISIONS,
)
from fervis.lookup.source_binding.model import (
    SourceBindingRequest,
    SourceMetricFitBasis,
)
from fervis.lookup.source_binding.parser_common import _dict, _text


__all__ = [
    "candidate_fitting_metric_measure_evidence_ids",
    "candidate_fitting_row_count_basis_evidence_ids",
    "fitting_metric_measure_evidence_ids",
    "fitting_row_count_basis_evidence_ids",
    "metric_fit_interpretations_by_requested_fact",
    "plan_shape_uses_row_count_as_metric",
    "source_metric_fit_bases",
]


def metric_fit_interpretations_by_requested_fact(
    payload: ParsedSourceBindingPlan,
    *,
    request: SourceBindingRequest,
) -> dict[str, dict[str, dict[str, str]]]:
    bases_by_fact = _dict(payload.metric_fit_bases, "metric_fit_bases")
    interpretations_by_fact = _dict(
        payload.fit_basis_interpretations,
        "fit_basis_interpretations",
    )
    expected_by_fact = source_binding_metric_evidence_ids_by_requested_fact(request)
    unexpected_fact_ids = (set(bases_by_fact) | set(interpretations_by_fact)) - set(
        expected_by_fact
    )
    if unexpected_fact_ids:
        raise ValueError("metric fit output references unknown requested fact")

    output: dict[str, dict[str, dict[str, str]]] = {}
    for requested_fact_id, expected_metric_ids in expected_by_fact.items():
        raw_fact_bases = _dict(
            bases_by_fact.get(requested_fact_id),
            f"metric_fit_bases.{requested_fact_id}",
        )
        raw_fact_interpretations = _dict(
            interpretations_by_fact.get(requested_fact_id),
            f"fit_basis_interpretations.{requested_fact_id}",
        )
        expected = set(expected_metric_ids)
        actual_bases = set(raw_fact_bases)
        actual_interpretations = set(raw_fact_interpretations)
        if (actual_bases | actual_interpretations) - expected:
            raise ValueError("metric fit output references unknown metric evidence")
        if expected - actual_bases:
            raise ValueError("metric_fit_bases must include every metric")
        if expected - actual_interpretations:
            raise ValueError("fit_basis_interpretations must interpret every metric")
        if actual_bases != actual_interpretations:
            raise ValueError("fit_basis_interpretations must match metric_fit_bases")
        fact_reviews: dict[str, dict[str, str]] = {}
        for metric_evidence_id, raw_basis in raw_fact_bases.items():
            basis = provider_output.MetricFitBasisOutput.parse(raw_basis)
            metric_meaning = _text(basis.metric_meaning)
            fit_basis = _text(basis.fit_basis)
            raw_interpretation = provider_output.FitBasisInterpretationOutput.parse(
                raw_fact_interpretations.get(metric_evidence_id),
            )
            decision = _text(raw_interpretation.interpretation)
            if decision not in METRIC_FIT_DECISIONS:
                raise ValueError("unknown fit_basis interpretation")
            fact_reviews[str(metric_evidence_id)] = {
                "interpretation": decision,
                "metric_meaning": metric_meaning,
                "fit_basis": fit_basis,
            }
        output[requested_fact_id] = fact_reviews
    if not expected_by_fact and (bases_by_fact or interpretations_by_fact):
        raise ValueError("metric fit output must be empty without metric candidates")
    missing_fact_ids = set(expected_by_fact) - (
        set(bases_by_fact) & set(interpretations_by_fact)
    )
    if missing_fact_ids:
        raise ValueError("metric fit output must include every requested fact")
    return output


def fitting_metric_measure_evidence_ids(
    *,
    requested_fact_id: str,
    answer_output_id: str,
    selected_metric_measure_evidence_ids: tuple[str, ...],
    metric_fit_reviews_by_requested_output: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, ...]:
    if not selected_metric_measure_evidence_ids:
        return ()
    reviews_by_metric = metric_fit_reviews_by_requested_output.get(
        requested_fact_id, {}
    )
    fitting_metric_ids: list[str] = []
    for evidence_id in selected_metric_measure_evidence_ids:
        review = reviews_by_metric.get(evidence_id)
        if review is None:
            raise ValueError("fit_basis_interpretations missing selected metric")
        if _metric_fit_review_interpretation(review) != METRIC_FIT_DECISION:
            raise ValueError("selected support set metric does not fit")
        fitting_metric_ids.append(evidence_id)
    return tuple(fitting_metric_ids)


def candidate_fitting_metric_measure_evidence_ids(
    *,
    requested_fact_id: str,
    answer_output_id: str,
    candidate_metric_measure_evidence_ids: tuple[str, ...],
    metric_fit_reviews_by_requested_output: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, ...]:
    if not candidate_metric_measure_evidence_ids:
        return ()
    reviews_by_metric = metric_fit_reviews_by_requested_output.get(
        requested_fact_id, {}
    )
    fitting_metric_ids: list[str] = []
    for evidence_id in candidate_metric_measure_evidence_ids:
        review = reviews_by_metric.get(evidence_id)
        if review is None:
            continue
        if _metric_fit_review_interpretation(review) == METRIC_FIT_DECISION:
            fitting_metric_ids.append(evidence_id)
    return tuple(fitting_metric_ids)


def fitting_row_count_basis_evidence_ids(
    *,
    requested_fact_id: str,
    answer_output_id: str,
    selected_row_count_basis_evidence_ids: tuple[str, ...],
    metric_fit_reviews_by_requested_output: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, ...]:
    if not selected_row_count_basis_evidence_ids:
        return ()
    reviews_by_metric = metric_fit_reviews_by_requested_output.get(
        requested_fact_id, {}
    )
    fitting_metric_ids: list[str] = []
    for evidence_id in selected_row_count_basis_evidence_ids:
        review = reviews_by_metric.get(evidence_id)
        if review is None:
            raise ValueError(
                "fit_basis_interpretations missing selected row count basis"
            )
        if _metric_fit_review_interpretation(review) != METRIC_FIT_DECISION:
            raise ValueError("selected row count basis does not fit")
        fitting_metric_ids.append(evidence_id)
    return tuple(fitting_metric_ids)


def candidate_fitting_row_count_basis_evidence_ids(
    *,
    requested_fact_id: str,
    answer_output_id: str,
    candidate_row_count_basis_evidence_ids: tuple[str, ...],
    metric_fit_reviews_by_requested_output: dict[str, dict[str, dict[str, str]]],
) -> tuple[str, ...]:
    if not candidate_row_count_basis_evidence_ids:
        return ()
    reviews_by_metric = metric_fit_reviews_by_requested_output.get(
        requested_fact_id, {}
    )
    fitting_metric_ids: list[str] = []
    for evidence_id in candidate_row_count_basis_evidence_ids:
        review = reviews_by_metric.get(evidence_id)
        if review is None:
            continue
        if _metric_fit_review_interpretation(review) == METRIC_FIT_DECISION:
            fitting_metric_ids.append(evidence_id)
    return tuple(fitting_metric_ids)


def source_metric_fit_bases(
    *,
    requested_fact_id: str,
    answer_output_id: str,
    evidence_ids: tuple[str, ...],
    metric_fit_reviews_by_requested_output: dict[str, dict[str, dict[str, str]]],
) -> tuple[SourceMetricFitBasis, ...]:
    reviews_by_metric = metric_fit_reviews_by_requested_output.get(
        requested_fact_id, {}
    )
    output: list[SourceMetricFitBasis] = []
    for evidence_id in dict.fromkeys(evidence_ids):
        review = reviews_by_metric.get(evidence_id)
        if review is None:
            continue
        if _metric_fit_review_interpretation(review) != METRIC_FIT_DECISION:
            continue
        output.append(
            SourceMetricFitBasis(
                evidence_id=evidence_id,
                metric_meaning=_text(review.get("metric_meaning")),
                fit_basis=_text(review.get("fit_basis")),
            )
        )
    return tuple(output)


def _metric_fit_review_interpretation(review: dict[str, str]) -> str:
    return _text(review.get("interpretation"))


def plan_shape_uses_row_count_as_metric(plan_shape: str) -> bool:
    return plan_shape in {"aggregate_scalar", "aggregate_by_group"}
