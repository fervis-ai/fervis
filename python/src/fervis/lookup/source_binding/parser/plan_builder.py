"""Build runtime source-binding plans from parsed provider output."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.lookup.fact_plan.relations import (
    RelationSourcePopulationChoice,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.closed_key_params import (
    closed_key_param_binding_index,
)
from fervis.lookup.source_binding.model import (
    BoundSource,
    SourceBindingPlan,
    SourceBindingRequest,
)
from fervis.lookup.source_binding.parser.candidate_access import (
    candidate_applied_filters,
    candidate_cardinality,
    candidate_source_evidence_items,
    candidate_source_fields,
    candidate_value_is_used_by_bound_source,
)
from fervis.lookup.source_binding.parser.fulfillment import parse_source_fulfillments
from fervis.lookup.source_binding.parser.metric_fit import (
    metric_fit_interpretations_by_requested_fact,
)
from fervis.lookup.source_binding.parser.params import (
    merged_param_bindings,
    parse_param_decision_binding_sets,
)
from fervis.lookup.source_binding.parser.population import (
    bound_relation_source,
    parse_answer_population,
)
from fervis.lookup.source_binding.parser.row_predicates import (
    parse_row_predicate_filters,
)
from fervis.lookup.source_binding.parser_common import _text
from fervis.lookup.source_binding.plan_targets import SourceBindingTargetIndex
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.role_selection import (
    bound_plan_selection_for_source_binding,
)


__all__ = [
    "build_source_binding_plan",
]


def build_source_binding_plan(
    payload: provider_output.SourceBindingPlanOutput,
    request: SourceBindingRequest,
    *,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, SourceCandidate],
    effective_param_ids_by_index: dict[int, tuple[str, ...]] | None = None,
    population_choices_by_index: (
        dict[int, tuple[RelationSourcePopulationChoice, ...]] | None
    ) = None,
) -> SourceBindingPlan:
    value_candidates_by_relation_id = _value_candidates_by_source_relation_id(
        candidates.values()
    )
    requested_fact_output_ids = {
        fact.id: {output.id for output in fact.support_answer_outputs}
        for fact in request.requested_facts
    }
    metric_answer_output_ids = _metric_answer_output_ids_by_requested_fact(request)
    metric_fit_reviews = metric_fit_interpretations_by_requested_fact(
        payload,
        request=request,
    )
    closed_key_bindings = closed_key_param_binding_index(
        request,
        targets=target_index.targets,
        candidates_by_id=candidates,
    )
    seen_binding_target_ids: set[str] = set()
    output: list[BoundSource] = []
    for index, parsed_invocation in enumerate(payload.source_invocations, start=1):
        target = target_index.require(_text(parsed_invocation.binding_target_id))
        if target.binding_target_id in seen_binding_target_ids:
            raise ValueError("duplicate source binding target")
        seen_binding_target_ids.add(target.binding_target_id)
        requested_fact_id = target.requested_fact_id
        if requested_fact_id not in requested_fact_output_ids:
            raise ValueError("source binding references unknown requested fact")
        candidate_id = target.source_candidate_id
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("source binding references unknown source candidate")
        if (
            candidate.applies_to_requested_fact_ids
            and requested_fact_id not in candidate.applies_to_requested_fact_ids
        ):
            raise ValueError("source candidate does not apply to requested fact")
        if (
            candidate.requested_fact_id
            and candidate.requested_fact_id != requested_fact_id
        ):
            raise ValueError("source candidate does not belong to requested fact")
        answer_population, population_binding = parse_answer_population(
            parsed_invocation.answer_population,
            request=request,
            requested_fact_id=requested_fact_id,
            candidate=candidate,
        )
        param_decisions = parse_param_decision_binding_sets(
            parsed_invocation.param_decisions,
            candidate=candidate,
            available_values=request.available_values,
            answer_population=answer_population,
            effective_param_ids=(effective_param_ids_by_index or {}).get(index),
        )
        row_predicates = parse_row_predicate_filters(
            parsed_invocation.row_predicate_reviews,
            candidate=candidate,
            request=request,
            requested_fact_id=requested_fact_id,
            binding_target_id=target.binding_target_id,
            review_scope=review_scope,
        )
        candidate_base_binding_sets = candidate.applied_param_binding_sets or (
            candidate.applied_param_bindings,
        )
        backend_param_binding_sets = closed_key_bindings.backend_param_binding_sets(
            target.binding_target_id,
        )
        param_binding_sets = tuple(
            merged_param_bindings(
                merged_param_bindings(
                    base_param_bindings,
                    backend_param_bindings,
                ),
                model_param_bindings,
            )
            for base_param_bindings in candidate_base_binding_sets
            for backend_param_bindings in backend_param_binding_sets
            for model_param_bindings in param_decisions.binding_sets
        )
        population_choices = (
            *((population_choices_by_index or {}).get(index, ())),
            *row_predicates.population_choices,
        )
        row_filters = row_predicates.filters
        fulfillments = parse_source_fulfillments(
            parsed_invocation.fulfillment_decisions,
            requested_fact_id=requested_fact_id,
            answer_output_ids=set(target.answer_output_ids),
            required_answer_output_ids=set(target.required_answer_output_ids),
            metric_answer_output_ids=metric_answer_output_ids.get(
                requested_fact_id,
                set(),
            ),
            candidate=candidate,
            plan_shape=target.plan_shape,
            metric_fit_reviews_by_requested_output=metric_fit_reviews,
        )
        closed_key_bindings.require_compatible_fulfillments(
            target.binding_target_id,
            candidate=candidate,
            fulfillments=fulfillments,
        )
        source, source_invocations = bound_relation_source(
            candidate=candidate,
            population_binding=population_binding,
            param_binding_sets=param_binding_sets,
            population_choices=population_choices,
        )
        if row_filters and source is not None:
            source = replace(source, row_filters=row_filters)
            source_invocations = tuple(
                replace(source_invocation, row_filters=row_filters)
                for source_invocation in source_invocations
            )
        evidence_items = candidate_source_evidence_items(candidate)
        available_fields = candidate_source_fields(
            candidate,
            evidence_items=evidence_items,
            fulfillments=fulfillments,
            row_filters=row_filters,
        )
        bound = BoundSource(
            id=f"sb_{len(output) + 1}",
            requested_fact_id=requested_fact_id,
            binding_target_id=target.binding_target_id,
            requirement_id=target.requirement_id,
            answer_population=answer_population,
            fulfillments=fulfillments,
            source=source,
            source_invocations=source_invocations,
            value_id=candidate.value_id,
            source_candidate_id=candidate.id,
            cardinality=candidate_cardinality(candidate),
            evidence_items=evidence_items,
            available_field_ids=tuple(
                sorted(field.field_id for field in available_fields)
            ),
            available_fields=available_fields,
            applied_filters=closed_key_bindings.source_level_applied_filters(
                target.binding_target_id,
                candidate_applied_filters(candidate),
            ),
        )
        output.append(bound)
        output.extend(
            _derived_value_bound_sources(
                bound,
                value_candidates_by_relation_id=value_candidates_by_relation_id,
                next_index=len(output) + 1,
            )
        )
    _require_answer_output_coverage(
        output,
        requested_fact_output_ids=requested_fact_output_ids,
    )
    plan = SourceBindingPlan(bound_sources=tuple(output))
    _require_complete_role_target_coverage(plan, request=request)
    return plan


def _metric_answer_output_ids_by_requested_fact(
    request: SourceBindingRequest,
) -> dict[str, set[str]]:
    return {
        fact.id: {
            output.id
            for output in fact.support_answer_outputs
            if output.role in {"MEASURED_VALUE", "ROW_POPULATION"}
        }
        for fact in request.requested_facts
    }


def _require_complete_role_target_coverage(
    plan: SourceBindingPlan,
    *,
    request: SourceBindingRequest,
) -> None:
    if (
        bound_plan_selection_for_source_binding(
            request.plan_selection,
            plan,
            requested_facts=request.requested_facts,
        )
        is None
    ):
        raise ValueError(
            "source binding must cover one complete source binding role set"
        )


def _require_answer_output_coverage(
    bound_sources: list[BoundSource],
    *,
    requested_fact_output_ids: dict[str, set[str]],
) -> None:
    covered: dict[str, set[str]] = {
        fact_id: set() for fact_id in requested_fact_output_ids
    }
    for source in bound_sources:
        if source.requested_fact_id not in covered:
            continue
        for fulfillment in source.fulfillments:
            covered[source.requested_fact_id].add(fulfillment.answer_output_id)
    for requested_fact_id, answer_output_ids in requested_fact_output_ids.items():
        missing = answer_output_ids - covered[requested_fact_id]
        if missing:
            raise ValueError("source binding does not cover requested answer outputs")


def _value_candidates_by_source_relation_id(
    candidates: Any,
) -> dict[str, tuple[Any, ...]]:
    output: dict[str, list[Any]] = {}
    for candidate in candidates:
        payload = getattr(candidate, "payload", None)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("kind") or "") != "value":
            continue
        relation_id = str(payload.get("source_relation_id") or "")
        if not relation_id:
            continue
        output.setdefault(relation_id, []).append(candidate)
    return {key: tuple(value) for key, value in output.items()}


def _derived_value_bound_sources(
    bound: BoundSource,
    *,
    value_candidates_by_relation_id: dict[str, tuple[Any, ...]],
    next_index: int,
) -> tuple[BoundSource, ...]:
    source = bound.source
    if source is None or not source.memory_relation_id:
        return ()
    candidates = value_candidates_by_relation_id.get(source.memory_relation_id, ())
    return tuple(
        BoundSource(
            id=f"sb_{next_index + index}",
            requested_fact_id=bound.requested_fact_id,
            answer_population=bound.answer_population,
            value_id=candidate.value_id,
            source_candidate_id=candidate.id,
            evidence_items=candidate_source_evidence_items(candidate),
        )
        for index, candidate in enumerate(candidates)
        if candidate.value_id
        and candidate_value_is_used_by_bound_source(
            candidate,
            bound,
        )
    )
