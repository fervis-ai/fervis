"""Top-level source-binding parser entrypoint."""

from __future__ import annotations

from typing import Any

from fervis.lookup.fact_plan.relations import RelationSourcePopulationChoice
from fervis.lookup.source_binding.candidates import source_candidate_required_param_decision_ids
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import SourceBindingRequest, SourceBindingResult
from fervis.lookup.source_binding.parser.context import source_binding_parse_context
from fervis.lookup.source_binding.parser.finite_choices import derive_finite_choice_param_decisions
from fervis.lookup.source_binding.parser.params import normalize_param_decisions
from fervis.lookup.source_binding.parser.plan_builder import build_source_binding_plan
from fervis.lookup.source_binding.parser_common import _dict, _required_dicts, _text
from fervis.lookup.source_binding.plan_targets import SourceBindingTargetIndex
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.terminal_parser import _plan_clarification, _plan_impossible


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
        plan_output = provider_output.SourceBindingPlanOutput.parse(outcome)
        context = source_binding_parse_context(request)
        (
            normalized_plan,
            effective_param_ids_by_index,
            population_choices_by_index,
        ) = _normalize_source_binding_payload_with_derived_finite_choices(
            plan_output,
            request,
            target_index=context.target_index,
            review_scope=context.review_scope,
            candidates=context.candidates,
        )
        return SourceBindingResult(
            outcome=build_source_binding_plan(
                normalized_plan,
                request,
                target_index=context.target_index,
                review_scope=context.review_scope,
                candidates=context.candidates,
                effective_param_ids_by_index=effective_param_ids_by_index,
                population_choices_by_index=population_choices_by_index,
            )
        )
    if kind == "needs_clarification":
        return SourceBindingResult(outcome=_plan_clarification(outcome))
    if kind == "impossible":
        return SourceBindingResult(outcome=_plan_impossible(outcome, request=request))
    raise ValueError(f"unsupported source binding outcome: {kind}")


def _normalize_source_binding_payload_with_derived_finite_choices(
    payload: provider_output.SourceBindingPlanOutput,
    request: SourceBindingRequest,
    *,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, Any],
) -> tuple[
    provider_output.SourceBindingPlanOutput,
    dict[int, tuple[str, ...]],
    dict[int, tuple[RelationSourcePopulationChoice, ...]],
]:
    normalized_invocations: list[provider_output.SourceInvocationOutput] = []
    effective_param_ids_by_index: dict[int, tuple[str, ...]] = {}
    population_choices_by_index: dict[
        int, tuple[RelationSourcePopulationChoice, ...]
    ] = {}
    for index, raw in enumerate(
        _required_dicts(payload.source_invocations, "source_invocations"),
        start=1,
    ):
        parsed_invocation = provider_output.SourceInvocationOutput.parse(raw)
        target = target_index.require(
            _text(parsed_invocation.binding_target_id)
        )
        candidate = candidates.get(target.source_candidate_id)
        if candidate is None:
            raise ValueError("source binding references unknown source candidate")
        raw_param_decisions = normalize_param_decisions(
            parsed_invocation.param_decisions,
            parse_provider_output=True,
        )
        derived = derive_finite_choice_param_decisions(
            parsed_invocation.finite_choice_param_reviews,
            candidate=candidate,
            requested_fact_id=target.requested_fact_id,
            binding_target_id=target.binding_target_id,
            request=request,
            review_scope=review_scope,
            answer_population=provider_output.AnswerPopulationOutput.parse(
                parsed_invocation.answer_population
            ),
            raw_param_decision_ids=tuple(raw_param_decisions),
        )
        combined_decisions = {**raw_param_decisions, **derived.param_decisions}
        normalized_invocations.append(
            provider_output.SourceInvocationOutput(
                binding_target_id=parsed_invocation.binding_target_id,
                answer_population=parsed_invocation.answer_population,
                fulfillment_decisions=parsed_invocation.fulfillment_decisions,
                param_decisions=combined_decisions,
                row_predicate_reviews=parsed_invocation.row_predicate_reviews,
                finite_choice_param_reviews=(
                    parsed_invocation.finite_choice_param_reviews
                ),
            )
        )
        population_choices_by_index[index] = derived.population_choices
        effective_param_ids_by_index[index] = tuple(
            dict.fromkeys(
                (
                    *source_candidate_required_param_decision_ids(candidate),
                    *combined_decisions.keys(),
                )
            )
        )
    return (
        provider_output.SourceBindingPlanOutput(
            kind="source_bindings",
            metric_fit_bases=payload.metric_fit_bases,
            fit_basis_interpretations=payload.fit_basis_interpretations,
            source_invocations=tuple(normalized_invocations),
        ),
        effective_param_ids_by_index,
        population_choices_by_index,
    )
