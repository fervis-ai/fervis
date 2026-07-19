"""Top-level source-binding parser entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import PopulationCoverageClaim
from fervis.lookup.answer_program.relations import PopulationCoverageRole
from fervis.lookup.source_binding.candidates import (
    SourceCandidate,
    source_candidate_required_param_decision_ids,
)
from fervis.lookup.source_binding.closed_key_params import (
    ClosedKeyParamBindingIndex,
    closed_key_param_binding_index,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import SourceBindingRequest, SourceBindingResult
from fervis.lookup.source_binding.parser.context import source_binding_parse_context
from fervis.lookup.source_binding.parser.finite_choices import (
    derive_finite_choice_param_decisions,
)
from fervis.lookup.source_binding.parser.model import (
    ParsedRoleBinding,
    ParsedSourceBindingPlan,
)
from fervis.lookup.source_binding.parser.types import NormalizedParamDecision
from fervis.lookup.source_binding.input_applications import (
    ParsedResolvedInputApplications,
    ResolvedInputApplicationSurface,
    parse_resolved_input_applications,
)
from fervis.lookup.source_binding.parser.plan_builder import build_source_binding_plan
from fervis.lookup.source_binding.parser_common import _dict, _text
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
    SourceBindingTargetIndex,
    source_binding_fact_field_id,
)


from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.terminal_parser import (
    _plan_clarification,
    _plan_impossible,
)


@dataclass(frozen=True)
class _NormalizedBindingDecisions:
    param_decisions: dict[str, NormalizedParamDecision]
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...]
    population_coverage_claims: tuple[PopulationCoverageClaim, ...]
    input_applications: ParsedResolvedInputApplications


__all__ = [
    "parse_source_binding",
]


def parse_source_binding(
    payload: dict[str, Any],
    *,
    request: SourceBindingRequest,
) -> SourceBindingResult:
    outcome = _dict(payload.get("outcome"), "outcome")
    kind = _text(outcome.get("kind"))
    if kind == "source_bindings":
        context = source_binding_parse_context(request)
        normalized_plan = _normalize_source_binding_payload_with_derived_finite_choices(
            outcome,
            request,
            target_index=context.target_index,
            review_scope=context.review_scope,
            candidates=context.candidates,
            input_application_surfaces=context.input_application_surfaces,
        )
        return SourceBindingResult(
            outcome=build_source_binding_plan(
                normalized_plan,
                request,
                target_index=context.target_index,
                review_scope=context.review_scope,
                candidates=context.candidates,
            )
        )
    if kind == "needs_clarification":
        return SourceBindingResult(outcome=_plan_clarification(outcome))
    if kind == "impossible":
        return SourceBindingResult(outcome=_plan_impossible(outcome, request=request))
    raise ValueError(f"unsupported source binding outcome: {kind}")


def _normalize_source_binding_payload_with_derived_finite_choices(
    payload: dict[str, Any],
    request: SourceBindingRequest,
    *,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, SourceCandidate],
    input_application_surfaces: dict[str, ResolvedInputApplicationSurface],
) -> ParsedSourceBindingPlan:
    closed_key_bindings = closed_key_param_binding_index(
        request,
        targets=target_index.targets,
        candidates_by_id=candidates,
    )
    normalized_bindings: list[ParsedRoleBinding] = []
    for requested_fact_id, raw_fact_binding in _fact_binding_payloads(
        payload,
        request=request,
    ):
        normalized_bindings.extend(
            _normalize_requested_fact_binding(
                requested_fact_id,
                raw_fact_binding,
                request=request,
                target_index=target_index,
                review_scope=review_scope,
                candidates=candidates,
                closed_key_bindings=closed_key_bindings,
                input_application_surfaces=input_application_surfaces,
            )
        )
    return ParsedSourceBindingPlan(
        metric_fit_bases=payload.get("metric_fit_bases"),
        fit_basis_interpretations=payload.get("fit_basis_interpretations"),
        role_bindings=tuple(normalized_bindings),
    )


def _fact_binding_payloads(
    payload: dict[str, Any],
    *,
    request: SourceBindingRequest,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    fact_fields = tuple(
        (fact.id, source_binding_fact_field_id(fact.id))
        for fact in request.requested_facts
    )
    required_fields = {
        "kind",
        "metric_fit_bases",
        "fit_basis_interpretations",
        *(field_id for _, field_id in fact_fields),
    }
    unexpected_fields = set(payload) - required_fields
    if unexpected_fields:
        field_id = min(unexpected_fields)
        raise ValueError(f"source binding contains unexpected field: {field_id}")
    return tuple(
        (
            requested_fact_id,
            _dict(payload.get(field_id), field_id),
        )
        for requested_fact_id, field_id in fact_fields
    )


def _normalize_requested_fact_binding(
    requested_fact_id: str,
    raw_fact_binding: object,
    *,
    request: SourceBindingRequest,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, SourceCandidate],
    closed_key_bindings: ClosedKeyParamBindingIndex,
    input_application_surfaces: dict[str, ResolvedInputApplicationSurface],
) -> tuple[ParsedRoleBinding, ...]:
    fact_binding = _dict(
        raw_fact_binding,
        source_binding_fact_field_id(requested_fact_id),
    )
    plan_shape = _text(fact_binding.get("plan_shape"))
    role_bindings = {
        requirement_id: raw_invocation
        for requirement_id, raw_invocation in fact_binding.items()
        if requirement_id != "plan_shape"
    }
    return tuple(
        _normalize_role_binding(
            requested_fact_id=requested_fact_id,
            plan_shape=plan_shape,
            requirement_id=requirement_id,
            raw_invocation=raw_invocation,
            request=request,
            target_index=target_index,
            review_scope=review_scope,
            candidates=candidates,
            closed_key_bindings=closed_key_bindings,
            input_application_surfaces=input_application_surfaces,
        )
        for requirement_id, raw_invocation in role_bindings.items()
    )


def _normalize_role_binding(
    *,
    requested_fact_id: str,
    plan_shape: str,
    requirement_id: str,
    raw_invocation: object,
    request: SourceBindingRequest,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, SourceCandidate],
    closed_key_bindings: ClosedKeyParamBindingIndex,
    input_application_surfaces: dict[str, ResolvedInputApplicationSurface],
) -> ParsedRoleBinding:
    parsed_invocation = provider_output.SourceInvocationOutput.parse(raw_invocation)
    target = target_index.require(_text(parsed_invocation.binding_target_id))
    _require_enclosing_binding_address(
        target,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
        requirement_id=requirement_id,
    )
    candidate = _require_source_candidate(target, candidates=candidates)
    decisions = _normalize_binding_decisions(
        parsed_invocation,
        target=target,
        candidate=candidate,
        request=request,
        review_scope=review_scope,
        closed_key_bindings=closed_key_bindings,
        input_application_surface=input_application_surfaces.get(
            target.binding_target_id
        ),
    )
    effective_param_ids = _effective_param_ids(
        candidate,
        param_decisions=decisions.param_decisions,
        target=target,
        closed_key_bindings=closed_key_bindings,
        input_applications=decisions.input_applications,
    )
    return ParsedRoleBinding(
        target=target,
        invocation=parsed_invocation,
        param_decisions=decisions.param_decisions,
        effective_param_ids=effective_param_ids,
        population_choices=decisions.population_choices,
        population_coverage_claims=decisions.population_coverage_claims,
        input_applications=decisions.input_applications,
    )


def _require_enclosing_binding_address(
    target: SourceBindingTarget,
    *,
    requested_fact_id: str,
    plan_shape: str,
    requirement_id: str,
) -> None:
    address = (requested_fact_id, plan_shape, requirement_id)
    target_address = (
        target.requested_fact_id,
        target.plan_shape,
        target.requirement_id,
    )
    if target_address != address:
        raise ValueError(
            "source binding target does not match its fact, shape, and role"
        )


def _require_source_candidate(
    target: SourceBindingTarget,
    *,
    candidates: dict[str, SourceCandidate],
) -> SourceCandidate:
    candidate = candidates.get(target.source_candidate_id)
    if candidate is None:
        raise ValueError("source binding references unknown source candidate")
    return candidate


def _normalize_binding_decisions(
    invocation: provider_output.SourceInvocationOutput,
    *,
    target: SourceBindingTarget,
    candidate: SourceCandidate,
    request: SourceBindingRequest,
    review_scope: SourceBindingReviewScope,
    closed_key_bindings: ClosedKeyParamBindingIndex,
    input_application_surface: ResolvedInputApplicationSurface | None,
) -> _NormalizedBindingDecisions:
    if input_application_surface is None:
        raise ValueError("source binding target has no input application surface")
    input_applications = parse_resolved_input_applications(
        invocation.resolved_input_applications,
        surface=input_application_surface,
    )
    authored_decisions = {
        param_id: NormalizedParamDecision(
            population_intent=decision.population_intent,
            match_basis_explanation=decision.match_basis_explanation,
            param_decision_id=decision.param_decision_id or None,
        )
        for param_id, decision in invocation.param_decisions.items()
    }
    visible_decisions = closed_key_bindings.model_visible_param_map(
        target.binding_target_id,
        authored_decisions,
    )
    derived = derive_finite_choice_param_decisions(
        invocation.finite_choice_param_reviews,
        candidate=candidate,
        requested_fact_id=target.requested_fact_id,
        binding_target_id=target.binding_target_id,
        request=request,
        review_scope=review_scope,
        answer_population=invocation.answer_population,
        raw_param_decision_ids=tuple(visible_decisions),
        coverage_role=(
            PopulationCoverageRole.ROW_POPULATION
            if target.requires_answer_fulfillment
            else PopulationCoverageRole.OPERATION_CONDITION
        ),
    )
    decisions = {**visible_decisions, **derived.param_decisions}
    return _NormalizedBindingDecisions(
        param_decisions=decisions,
        population_choices=derived.population_choices,
        population_coverage_claims=derived.population_coverage_claims,
        input_applications=input_applications,
    )


def _effective_param_ids(
    candidate: SourceCandidate,
    *,
    param_decisions: dict[str, NormalizedParamDecision],
    target: SourceBindingTarget,
    closed_key_bindings: ClosedKeyParamBindingIndex,
    input_applications: ParsedResolvedInputApplications | None = None,
) -> tuple[str, ...]:
    effective_param_ids = tuple(
        dict.fromkeys(
            (
                *source_candidate_required_param_decision_ids(candidate),
                *param_decisions,
                *(
                    binding.param_id
                    for binding_set in (
                        input_applications.param_binding_sets
                        if input_applications is not None
                        else ()
                    )
                    for binding in binding_set
                ),
            )
        )
    )
    visible_param_ids = closed_key_bindings.model_visible_param_map(
        target.binding_target_id,
        dict.fromkeys(effective_param_ids),
    )
    return tuple(visible_param_ids)
