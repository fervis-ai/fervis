"""Parse Read Eligibility decisions mechanically."""

from __future__ import annotations

from fervis.lookup.grounding import IdentityResolutionTask
from fervis.lookup.read_eligibility import semantic_provider_contract as output
from fervis.lookup.read_eligibility.semantic import (
    CanonicalOptionAssessment,
    IdentityRouteOutcome,
    IdentityRouteSelection,
    NoCanonicalInterpretation,
    NoResolverRoute,
    ReadRequirementAssessment,
    ResolverRouteAssessment,
    SemanticFitDecision,
    SemanticReadDecision,
    SemanticReadEligibilityRequest,
    SemanticReadEligibilityResult,
)


def parse_semantic_read_eligibility(
    payload: dict[str, object],
    *,
    request: SemanticReadEligibilityRequest,
) -> SemanticReadEligibilityResult:
    parsed = output.SemanticReadEligibilityOutput.parse(payload)
    candidates = {
        candidate.candidate_ref: candidate for candidate in request.read_candidates
    }
    fact_ids = tuple(index.requested_fact_id for index in request.contexts)
    if set(parsed.read_assessments_by_requested_fact) != set(fact_ids):
        raise ValueError("read eligibility must assess every requested fact once")

    read_assessments: list[ReadRequirementAssessment] = []
    for requested_fact_id in fact_ids:
        fact_assessments = parsed.read_assessments_by_requested_fact[
            requested_fact_id
        ]
        if set(fact_assessments) != set(candidates):
            raise ValueError(
                "read eligibility must assess every shown read for every fact"
            )
        for candidate_ref, assessment in fact_assessments.items():
            candidate = candidates[candidate_ref]
            relevant_field_refs = tuple(
                dict.fromkeys(assessment.relevant_field_refs)
            )
            allowed_fields = {field.field_ref for field in candidate.fields}
            if any(field_ref not in allowed_fields for field_ref in relevant_field_refs):
                raise ValueError("read retention references an unavailable field")
            decision = SemanticReadDecision(assessment.decision)
            if decision is SemanticReadDecision.DROP and relevant_field_refs:
                raise ValueError("dropped read retains fields")
            read_assessments.append(
                ReadRequirementAssessment(
                    requested_fact_id=requested_fact_id,
                    candidate_ref=candidate_ref,
                    source_refs=candidate.source_refs,
                    read_id=candidate.read_id,
                    relevant_field_refs=relevant_field_refs,
                    assessment_basis=_required_text(assessment.assessment_basis),
                    decision=decision,
                )
            )

    tasks = {task.task_ref: task for task in request.identity_tasks}
    if set(parsed.identity_outcomes) != set(tasks):
        raise ValueError("read eligibility must decide every identity task once")
    outcomes = tuple(
        _identity_outcome(task_ref, item, request=request)
        for task_ref, item in parsed.identity_outcomes.items()
    )
    return SemanticReadEligibilityResult(
        read_assessments=tuple(read_assessments),
        identity_outcomes=outcomes,
    )


def _identity_outcome(
    task_ref: str,
    item: output.IdentityRouteOutcomeOutput,
    *,
    request: SemanticReadEligibilityRequest,
) -> IdentityRouteOutcome:
    task = next(task for task in request.identity_tasks if task.task_ref == task_ref)
    canonical_ids = tuple(option.canonical_option_id for option in task.canonical_options)
    canonical_assessments = _canonical_assessments(
        item.canonical_option_assessments,
        expected_ids=canonical_ids,
    )
    canonical_basis = _required_text(item.canonical_option_basis)
    route_basis = _required_text(item.resolver_route_basis)
    route_assessments = _route_assessments(
        item.resolver_route_assessments,
        expected_ids=tuple(route.route_ref for route in task.resolver_routes),
    )
    evidence = _identity_evidence(item.evidence_refs, task=task, request=request)
    outcome = item.outcome
    if outcome == "NO_CANONICAL_INTERPRETATION":
        if item.canonical_option_id is not None or item.resolver_route_id is not None:
            raise ValueError("no-canonical outcome selects an identity or route")
        if any(
            assessment.decision != SemanticFitDecision.DOES_NOT_FIT.value
            for assessment in canonical_assessments
        ) or any(
            assessment.decision != SemanticFitDecision.DOES_NOT_FIT.value
            for assessment in route_assessments
        ):
            raise ValueError("no-canonical outcome contains a fitting option")
        return NoCanonicalInterpretation(
            task_ref=task_ref,
            canonical_option_assessments=canonical_assessments,
            canonical_option_basis=canonical_basis,
            evidence_refs=evidence,
        )

    canonical_option_id = _nullable_selection(item.canonical_option_id)
    option = next(
        (
            candidate
            for candidate in task.canonical_options
            if candidate.canonical_option_id == canonical_option_id
        ),
        None,
    )
    if option is None:
        raise ValueError("identity outcome references an unknown meaning")
    _require_selected_fit(
        canonical_assessments,
        selected_id=canonical_option_id,
        label="canonical option",
    )
    _require_other_meaning_routes_do_not_fit(
        route_assessments,
        selected_route_ids=option.resolver_route_refs,
    )
    if outcome == "NO_RESOLVER_ROUTE":
        if item.resolver_route_id is not None:
            raise ValueError("no-resolver outcome selects a route")
        if any(
            assessment.decision != SemanticFitDecision.DOES_NOT_FIT.value
            for assessment in route_assessments
        ):
            raise ValueError("no-resolver outcome contains a fitting route")
        return NoResolverRoute(
            task_ref=task_ref,
            canonical_option_assessments=canonical_assessments,
            canonical_option_basis=canonical_basis,
            canonical_option_id=canonical_option_id,
            resolver_route_assessments=route_assessments,
            resolver_route_basis=route_basis,
            evidence_refs=evidence,
        )
    if outcome != "SELECTED":
        raise ValueError("unknown identity outcome")
    route_id = _nullable_selection(item.resolver_route_id)
    if route_id not in option.resolver_route_refs:
        raise ValueError("identity selection references an unknown resolver route")
    _require_selected_fit(
        route_assessments,
        selected_id=route_id,
        label="resolver route",
    )
    return IdentityRouteSelection(
        task_ref=task_ref,
        canonical_option_assessments=canonical_assessments,
        canonical_option_basis=canonical_basis,
        canonical_option_id=canonical_option_id,
        resolver_route_assessments=route_assessments,
        resolver_route_basis=route_basis,
        resolver_route_id=route_id,
    )


def _require_other_meaning_routes_do_not_fit(
    assessments: tuple[ResolverRouteAssessment, ...],
    *,
    selected_route_ids: tuple[str, ...],
) -> None:
    selected = set(selected_route_ids)
    if any(
        assessment.resolver_route_id not in selected
        and assessment.decision != SemanticFitDecision.DOES_NOT_FIT.value
        for assessment in assessments
    ):
        raise ValueError("resolver route for an unselected meaning cannot fit")


def _canonical_assessments(
    values: tuple[output.CanonicalOptionAssessmentOutput, ...],
    *,
    expected_ids: tuple[str, ...],
) -> tuple[CanonicalOptionAssessment, ...]:
    _exact_assessment_coverage(
        tuple(item.canonical_option_id for item in values),
        expected_ids=expected_ids,
        label="canonical option",
    )
    return tuple(
        CanonicalOptionAssessment(
            canonical_option_id=item.canonical_option_id,
            assessment=_required_text(item.assessment),
            decision=SemanticFitDecision(item.decision).value,
        )
        for item in values
    )


def _route_assessments(
    values: tuple[output.ResolverRouteAssessmentOutput, ...],
    *,
    expected_ids: tuple[str, ...],
) -> tuple[ResolverRouteAssessment, ...]:
    _exact_assessment_coverage(
        tuple(item.resolver_route_id for item in values),
        expected_ids=expected_ids,
        label="resolver route",
    )
    return tuple(
        ResolverRouteAssessment(
            resolver_route_id=item.resolver_route_id,
            assessment=_required_text(item.assessment),
            decision=SemanticFitDecision(item.decision).value,
        )
        for item in values
    )


def _exact_assessment_coverage(
    actual_ids: tuple[str, ...],
    *,
    expected_ids: tuple[str, ...],
    label: str,
) -> None:
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError(f"{label} assessments require exact coverage")


def _require_selected_fit(
    assessments: tuple[CanonicalOptionAssessment, ...]
    | tuple[ResolverRouteAssessment, ...],
    *,
    selected_id: str,
    label: str,
) -> None:
    selected = next(
        (
            item
            for item in assessments
            if (
                getattr(item, "canonical_option_id", "") == selected_id
                or getattr(item, "resolver_route_id", "") == selected_id
            )
        ),
        None,
    )
    if selected is None or selected.decision != SemanticFitDecision.FITS.value:
        raise ValueError(f"selected {label} was not assessed as fitting")


def _identity_evidence(
    values: tuple[str, ...],
    *,
    task: IdentityResolutionTask,
    request: SemanticReadEligibilityRequest,
) -> tuple[str, ...]:
    allowed_evidence = {
        task.task_ref,
        *(option.canonical_option_id for option in task.canonical_options),
        *(candidate.candidate_ref for candidate in request.read_candidates),
        *(route.route_ref for route in task.resolver_routes),
    }
    evidence = _unique_subset(
        values,
        allowed=allowed_evidence,
        label="identity clarification evidence",
    )
    if not evidence:
        raise ValueError("identity clarification requires evidence")
    return evidence


def _unique_subset(
    values: tuple[str, ...],
    *,
    allowed: set[str],
    label: str,
) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(values))
    if any(value not in allowed for value in unique):
        raise ValueError(f"{label} selection references an unknown value")
    return unique


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("read eligibility requires non-empty assessment text")
    return text


def _nullable_selection(value: str | None) -> str:
    if value is None:
        raise ValueError("identity outcome requires a selected value")
    return _required_text(value)


__all__ = ["parse_semantic_read_eligibility"]
