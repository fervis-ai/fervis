"""Build runtime source-binding plans from parsed provider output."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.closed_key_params import (
    ClosedKeyParamBindingIndex,
    closed_key_param_binding_index,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    build_row_source_catalog,
)
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.source_binding.model import (
    AnswerPopulation,
    BoundSource,
    SourceAppliedFilter,
    SourceBindingPlan,
    SourceBindingRequest,
    SourceEvidenceItem,
    SourceFulfillment,
)
from fervis.lookup.source_binding.parser.candidate_access import (
    candidate_applied_filters,
    candidate_cardinality,
    candidate_source_evidence_items,
    candidate_source_fields,
    candidate_value_is_used_by_bound_source,
)
from fervis.lookup.source_binding.parser.fulfillment import (
    fulfillment_row_source_id,
    parse_source_fulfillments,
)
from fervis.lookup.source_binding.parser.metric_fit import (
    metric_fit_interpretations_by_requested_fact,
)
from fervis.lookup.source_binding.parser.model import (
    ParsedRoleBinding,
    ParsedSourceBindingPlan,
)
from fervis.lookup.source_binding.parser.params import (
    merged_param_bindings,
    parse_param_decision_binding_sets,
)
from fervis.lookup.source_binding.parser.population import (
    bound_relation_source,
    parse_answer_population,
)
from fervis.lookup.source_binding.candidates.model import CandidatePopulationBinding
from fervis.lookup.source_binding.parser.row_predicates import (
    parse_row_predicate_filters,
)
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTarget,
    SourceBindingTargetIndex,
)
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.role_selection import (
    bound_plan_selection_for_source_binding,
)
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
)
from fervis.lookup.source_binding.membership_tests import membership_test_key


__all__ = [
    "build_source_binding_plan",
]


def build_source_binding_plan(
    payload: ParsedSourceBindingPlan,
    request: SourceBindingRequest,
    *,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, SourceCandidate],
) -> SourceBindingPlan:
    context = _plan_build_context(
        payload,
        request=request,
        target_index=target_index,
        review_scope=review_scope,
        candidates=candidates,
    )
    seen_binding_target_ids: set[str] = set()
    bound_sources: list[BoundSource] = []
    discharged_test_ids_by_fact: dict[str, set[str]] = {}
    for parsed_binding in payload.role_bindings:
        _require_new_binding_target(
            parsed_binding,
            seen_binding_target_ids=seen_binding_target_ids,
        )
        built = _build_bound_source(
            parsed_binding,
            context=context,
            source_index=len(bound_sources) + 1,
        )
        bound_source = built.source
        bound_sources.append(bound_source)
        discharged_test_ids_by_fact.setdefault(
            parsed_binding.target.requested_fact_id,
            set(),
        ).update(built.discharged_membership_test_ids)
        derived_sources = _derived_value_bound_sources(
            bound_source,
            value_candidates_by_relation_id=context.value_candidates_by_relation_id,
            next_index=len(bound_sources) + 1,
        )
        bound_sources.extend(derived_sources)
    plan = SourceBindingPlan(bound_sources=tuple(bound_sources))
    _require_complete_role_target_coverage(plan, request=request)
    _require_population_test_coverage(
        request,
        discharged_test_ids_by_fact=discharged_test_ids_by_fact,
    )
    return plan


@dataclass(frozen=True)
class _PlanBuildContext:
    request: SourceBindingRequest
    review_scope: SourceBindingReviewScope
    candidates: dict[str, SourceCandidate]
    requested_fact_output_ids: dict[str, set[str]]
    metric_answer_output_ids: dict[str, set[str]]
    metric_fit_reviews: dict[str, dict[str, dict[str, str]]]
    closed_key_bindings: ClosedKeyParamBindingIndex
    row_sources: RowSourceCatalog
    value_candidates_by_relation_id: dict[str, tuple[SourceCandidate, ...]]


@dataclass(frozen=True)
class _ParsedBindingDecisions:
    answer_population: AnswerPopulation
    population_binding: CandidatePopulationBinding
    param_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...]
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...]
    discharged_membership_test_ids: tuple[str, ...]


@dataclass(frozen=True)
class _BuiltBoundSource:
    source: BoundSource
    discharged_membership_test_ids: tuple[str, ...]


def _plan_build_context(
    payload: ParsedSourceBindingPlan,
    *,
    request: SourceBindingRequest,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, SourceCandidate],
) -> _PlanBuildContext:
    requested_fact_output_ids = {
        fact.id: {output.id for output in fact.support_answer_outputs}
        for fact in request.requested_facts
    }
    metric_fit_reviews = metric_fit_interpretations_by_requested_fact(
        payload,
        request=request,
    )
    closed_key_bindings = closed_key_param_binding_index(
        request,
        targets=target_index.targets,
        candidates_by_id=candidates,
    )
    return _PlanBuildContext(
        request=request,
        review_scope=review_scope,
        candidates=candidates,
        requested_fact_output_ids=requested_fact_output_ids,
        metric_answer_output_ids=_metric_answer_output_ids_by_requested_fact(request),
        metric_fit_reviews=metric_fit_reviews,
        closed_key_bindings=closed_key_bindings,
        row_sources=build_row_source_catalog(request.relation_catalog),
        value_candidates_by_relation_id=_value_candidates_by_source_relation_id(
            candidates.values()
        ),
    )


def _require_new_binding_target(
    binding: ParsedRoleBinding,
    *,
    seen_binding_target_ids: set[str],
) -> None:
    target_id = binding.target.binding_target_id
    if target_id in seen_binding_target_ids:
        raise ValueError("duplicate source binding target")
    seen_binding_target_ids.add(target_id)


def _build_bound_source(
    binding: ParsedRoleBinding,
    *,
    context: _PlanBuildContext,
    source_index: int,
) -> _BuiltBoundSource:
    candidate = _binding_candidate(binding, context=context)
    decisions = _parse_binding_decisions(
        binding,
        candidate=candidate,
        context=context,
    )
    fulfillments = _parse_binding_fulfillments(
        binding,
        candidate=candidate,
        context=context,
    )
    evidence_items = candidate_source_evidence_items(candidate)
    source = _materialize_bound_source(
        binding,
        candidate=candidate,
        decisions=decisions,
        fulfillments=fulfillments,
        evidence_items=evidence_items,
        context=context,
        source_index=source_index,
    )
    input_proved_test_ids = _input_proved_membership_test_ids(
        binding,
        source=source,
        param_binding_sets=decisions.param_binding_sets,
        request=context.request,
    )
    discharged_test_ids = tuple(
        dict.fromkeys(
            (
                *decisions.discharged_membership_test_ids,
                *input_proved_test_ids,
            )
        )
    )
    return _BuiltBoundSource(
        source=source,
        discharged_membership_test_ids=discharged_test_ids,
    )


def _input_proved_membership_test_ids(
    binding: ParsedRoleBinding,
    *,
    source: BoundSource,
    param_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...],
    request: SourceBindingRequest,
) -> tuple[str, ...]:
    bound_input_ids = {
        proof_ref.removeprefix("known_input:")
        for binding_set in param_binding_sets
        for param_binding in binding_set
        for proof_ref in param_binding.proof_refs
        if proof_ref.startswith("known_input:")
    }
    bound_input_ids.update(
        applied_filter.known_input_id
        for applied_filter in source.applied_filters
        if applied_filter.known_input_id
    )
    fact = next(
        item
        for item in request.requested_facts
        if item.id == binding.target.requested_fact_id
    )
    population = fact.answer_population
    if population is None:
        return ()
    return tuple(
        membership_test_key(test)
        for test in population.membership_tests
        if test.owned_question_input_refs
        and test.polarity is AnswerPopulationMembershipTestPolarity.MUST_PASS
        and set(test.owned_question_input_refs) <= bound_input_ids
    )


def _binding_candidate(
    binding: ParsedRoleBinding,
    *,
    context: _PlanBuildContext,
) -> SourceCandidate:
    target = binding.target
    requested_fact_id = target.requested_fact_id
    if requested_fact_id not in context.requested_fact_output_ids:
        raise ValueError("source binding references unknown requested fact")
    candidate = context.candidates.get(target.source_candidate_id)
    if candidate is None:
        raise ValueError("source binding references unknown source candidate")
    if requested_fact_id not in candidate.applies_to_requested_fact_ids:
        raise ValueError("source candidate does not apply to requested fact")
    return candidate


def _parse_binding_decisions(
    binding: ParsedRoleBinding,
    *,
    candidate: SourceCandidate,
    context: _PlanBuildContext,
) -> _ParsedBindingDecisions:
    target = binding.target
    invocation = binding.invocation
    answer_population, population_binding = parse_answer_population(
        invocation.answer_population,
        request=context.request,
        requested_fact_id=target.requested_fact_id,
        candidate=candidate,
    )
    param_decisions = parse_param_decision_binding_sets(
        binding.param_decisions,
        candidate=candidate,
        available_values=context.request.available_values,
        answer_population=answer_population,
        parameter_namespace=(
            f"semantic.{target.requested_fact_id}.{target.binding_target_id}"
        ),
        effective_param_ids=binding.effective_param_ids,
    )
    row_predicates = parse_row_predicate_filters(
        invocation.row_predicate_reviews,
        candidate=candidate,
        request=context.request,
        requested_fact_id=target.requested_fact_id,
        binding_target_id=target.binding_target_id,
        review_scope=context.review_scope,
    )
    param_binding_sets = _merged_binding_sets(
        candidate,
        backend_binding_sets=context.closed_key_bindings.backend_param_binding_sets(
            target.binding_target_id
        ),
        model_binding_sets=param_decisions.binding_sets,
    )
    population_choices = (
        *binding.population_choices,
        *row_predicates.population_choices,
    )
    return _ParsedBindingDecisions(
        answer_population=answer_population,
        population_binding=population_binding,
        param_binding_sets=param_binding_sets,
        population_choices=population_choices,
        discharged_membership_test_ids=tuple(
            dict.fromkeys(
                (
                    *binding.discharged_membership_test_ids,
                    *row_predicates.discharged_membership_test_ids,
                )
            )
        ),
    )


def _merged_binding_sets(
    candidate: SourceCandidate,
    *,
    backend_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...],
    model_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...],
) -> tuple[tuple[DraftEndpointParamBinding, ...], ...]:
    base_binding_sets = candidate.applied_param_binding_sets or (
        candidate.applied_param_bindings,
    )
    return tuple(
        merged_param_bindings(
            merged_param_bindings(base, backend),
            model,
        )
        for base in base_binding_sets
        for backend in backend_binding_sets
        for model in model_binding_sets
    )


def _parse_binding_fulfillments(
    binding: ParsedRoleBinding,
    *,
    candidate: SourceCandidate,
    context: _PlanBuildContext,
) -> tuple[SourceFulfillment, ...]:
    target = binding.target
    fulfillments = parse_source_fulfillments(
        binding.invocation.fulfillment_decisions,
        requested_fact_id=target.requested_fact_id,
        answer_output_ids=set(target.answer_output_ids),
        required_answer_output_ids=set(target.required_answer_output_ids),
        metric_answer_output_ids=context.metric_answer_output_ids.get(
            target.requested_fact_id,
            set(),
        ),
        candidate=candidate,
        plan_shape=target.plan_shape,
        metric_fit_reviews_by_requested_output=context.metric_fit_reviews,
    )
    context.closed_key_bindings.require_compatible_fulfillments(
        target.binding_target_id,
        candidate=candidate,
        fulfillments=fulfillments,
    )
    return fulfillments


def _materialize_bound_source(
    binding: ParsedRoleBinding,
    *,
    candidate: SourceCandidate,
    decisions: _ParsedBindingDecisions,
    fulfillments: tuple[SourceFulfillment, ...],
    evidence_items: tuple[SourceEvidenceItem, ...],
    context: _PlanBuildContext,
    source_index: int,
) -> BoundSource:
    target = binding.target
    selected_row_source_id = _selected_row_source_id(
        candidate,
        fulfillments,
        evidence_items=evidence_items,
    )
    source, source_invocations = bound_relation_source(
        candidate=candidate,
        population_binding=decisions.population_binding,
        param_binding_sets=decisions.param_binding_sets,
        population_choices=decisions.population_choices,
        row_source_id=selected_row_source_id,
    )
    required_field_ids = _required_target_field_ids(context.request, target=target)
    population_field_ids = tuple(
        choice.field_id for choice in decisions.population_choices if choice.field_id
    )
    available_fields = candidate_source_fields(
        candidate,
        row_source_id=selected_row_source_id,
        evidence_items=evidence_items,
        fulfillments=fulfillments,
        required_field_ids=(*required_field_ids, *population_field_ids),
        plan_shape=target.plan_shape,
    )
    available_field_ids = tuple(sorted(field.field_id for field in available_fields))
    applied_filters = context.closed_key_bindings.source_level_applied_filters(
        target.binding_target_id,
        candidate_applied_filters(candidate),
    )
    cardinality = candidate_cardinality(candidate)
    if selected_row_source_id:
        row_source = context.row_sources.find(selected_row_source_id)
        if row_source is not None:
            cardinality = _bound_cardinality(
                cardinality,
                row_source=row_source,
                applied_filters=applied_filters,
            )
    evidence_items = _bound_evidence_items(
        evidence_items,
        row_source_id=selected_row_source_id,
        cardinality=cardinality,
    )
    return BoundSource(
        id=f"sb_{source_index}",
        requested_fact_id=target.requested_fact_id,
        binding_target_id=target.binding_target_id,
        requirement_id=target.requirement_id,
        answer_population=decisions.answer_population,
        fulfillments=fulfillments,
        source=source,
        source_invocations=source_invocations,
        value_id=candidate.value_id,
        source_candidate_id=candidate.id,
        cardinality=cardinality,
        evidence_items=evidence_items,
        available_field_ids=available_field_ids,
        available_fields=available_fields,
        applied_filters=applied_filters,
    )


def _selected_row_source_id(
    candidate: SourceCandidate,
    fulfillments: tuple[SourceFulfillment, ...],
    *,
    evidence_items: tuple[SourceEvidenceItem, ...],
) -> str:
    fulfilled_row_source_id = fulfillment_row_source_id(
        fulfillments,
        evidence_items=evidence_items,
    )
    if fulfilled_row_source_id:
        return fulfilled_row_source_id
    if candidate.source is None:
        return ""
    return candidate.source.row_source_id


def _bound_cardinality(
    source_cardinality: str,
    *,
    row_source: RowSource,
    applied_filters: tuple[SourceAppliedFilter, ...],
) -> str:
    if source_cardinality != "many":
        return source_cardinality
    identity_field_ids = {
        field_id
        for applied_filter in applied_filters
        if applied_filter.value_kind == "identity"
        for field_id in applied_filter.predicate_field_ids
    }
    closes_candidate_key = any(
        key.stable
        and {component.field_id for component in key.components} <= identity_field_ids
        for key in row_source.candidate_keys
    )
    return "one" if closes_candidate_key else source_cardinality


def _bound_evidence_items(
    evidence_items: tuple[SourceEvidenceItem, ...],
    *,
    row_source_id: str,
    cardinality: str,
) -> tuple[SourceEvidenceItem, ...]:
    if cardinality != "one":
        return evidence_items
    return tuple(
        replace(item, row_cardinality="one")
        if item.row_source_id == row_source_id
        else item
        for item in evidence_items
    )


def _required_target_field_ids(
    request: SourceBindingRequest,
    *,
    target: SourceBindingTarget,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            field_id
            for plan in request.plan_selection.plan_selections
            if plan.requested_fact_id == target.requested_fact_id
            and plan.plan_shape == target.plan_shape
            for member in plan.source_members
            if member.source_candidate_id == target.source_candidate_id
            and (
                not member.requirement_ids
                or target.requirement_id in member.requirement_ids
            )
            for field_id in member.field_ids
        )
    )


def _metric_answer_output_ids_by_requested_fact(
    request: SourceBindingRequest,
) -> dict[str, set[str]]:
    return {
        fact.id: {
            output.id
            for output in fact.support_answer_outputs
            if output.role in {"MEASURED_VALUE", "ROW_COUNT"}
        }
        for fact in request.requested_facts
    }


def _require_complete_role_target_coverage(
    plan: SourceBindingPlan,
    *,
    request: SourceBindingRequest,
) -> None:
    bound_plan_selection = bound_plan_selection_for_source_binding(
        request.plan_selection,
        plan,
        requested_facts=request.requested_facts,
    )
    if bound_plan_selection is None:
        raise ValueError(
            "source binding must cover one complete source binding role set"
        )


def _require_population_test_coverage(
    request: SourceBindingRequest,
    *,
    discharged_test_ids_by_fact: dict[str, set[str]],
) -> None:
    for fact in request.requested_facts:
        population = fact.answer_population
        if population is None:
            continue
        discharged = discharged_test_ids_by_fact.get(fact.id, set())
        missing = tuple(
            membership_test_key(test)
            for test in population.membership_tests
            if test.kind == AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT
            and membership_test_key(test) not in discharged
        )
        if missing:
            raise ValueError(
                "source binding does not enforce answer population tests: "
                + ", ".join(missing)
            )


def _value_candidates_by_source_relation_id(
    candidates: Iterable[SourceCandidate],
) -> dict[str, tuple[SourceCandidate, ...]]:
    output: dict[str, list[SourceCandidate]] = {}
    for candidate in candidates:
        if candidate.kind != "value":
            continue
        relation_id = candidate.source_relation_id
        if not relation_id:
            continue
        output.setdefault(relation_id, []).append(candidate)
    return {key: tuple(value) for key, value in output.items()}


def _derived_value_bound_sources(
    bound: BoundSource,
    *,
    value_candidates_by_relation_id: dict[str, tuple[SourceCandidate, ...]],
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
