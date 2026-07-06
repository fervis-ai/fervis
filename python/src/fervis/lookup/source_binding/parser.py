"""Parse and validate source-binding model output."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

from fervis.lookup.relation_catalog import IdentityMetadata
from fervis.lookup.fact_plan.relations import (
    EndpointParamBinding,
    PopulationChoiceControllerKind,
    RelationSource,
    RelationSourcePopulationChoice,
    RelationSourceReviewScopeDecision,
    RelationSourceRowFilter,
    ReviewScopeDecisionKind as RelationReviewScopeDecisionKind,
    SourceKind,
)
from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.fact_plan.values import (
    FactValue,
    TimeComponent,
    ValueComponent,
    ValueKind,
)
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    NormalInstanceExplicitOverrideReason,
)
from fervis.lookup.source_binding.candidates import (
    source_candidate_required_param_decision_ids,
    source_candidates,
)
from fervis.lookup.source_binding.evidence_types import (
    evidence_item_can_measure,
)
from fervis.lookup.source_binding.model import (
    AnswerPopulation,
    BoundSource,
    SourceEvidenceItem,
    SourceField,
    SourceFulfillment,
    SourceMetricFitBasis,
    SourceBindingPlan,
    SourceBindingRequest,
    SourceBindingResult,
)
from fervis.lookup.source_binding.metric_fit import (
    METRIC_FIT_DECISION,
    METRIC_FIT_DECISIONS,
)
from fervis.lookup.source_binding.membership_tests import (
    membership_test_key,
    membership_tests_by_key,
)
from fervis.lookup.operation_families.source_binding_registry import (
    source_binding_metric_evidence_ids_by_requested_fact,
)
from fervis.lookup.source_binding.normal_instance_roles import (
    NORMAL_INSTANCE_NO_EXCLUDED_ROLE,
    NORMAL_INSTANCE_UNKNOWN_EXCLUDED_ROLE,
)
from fervis.lookup.source_binding.param_surface import (
    param_has_default_value,
)
from fervis.lookup.source_binding.param_values import canonical_param_value
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTargetIndex,
    source_binding_target_index,
)
from fervis.lookup.source_binding.population_bindings import (
    PopulationBindingIndex,
)
from fervis.lookup.source_binding.review_scope import (
    ReviewScopeDecision,
    SourceBindingReviewScope,
    source_binding_review_scope,
)
from fervis.lookup.source_binding.review_surface import (
    FiniteChoiceReviewAxis,
    source_binding_review_surface,
)
from fervis.lookup.source_binding.role_selection import (
    bound_plan_selection_for_source_binding,
)
from fervis.lookup.source_binding.parser_common import (
    _dict,
    _required_dicts,
    _required_strings,
    _text,
    _optional_text,
)
from fervis.lookup.source_binding.terminal_parser import (
    _plan_clarification,
    _plan_impossible,
)


@dataclass(frozen=True)
class _ParamDecisionParse:
    binding_sets: tuple[tuple[EndpointParamBinding, ...], ...]


@dataclass(frozen=True)
class _RowPredicateParse:
    filters: tuple[RelationSourceRowFilter, ...] = ()
    population_choices: tuple[RelationSourcePopulationChoice, ...] = ()


@dataclass(frozen=True)
class _PopulationChoiceSet:
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...]


@dataclass(frozen=True)
class _DerivedFiniteChoiceParamDecisions:
    param_decisions: dict[str, dict[str, Any]]
    population_choices: tuple[RelationSourcePopulationChoice, ...]


def parse_source_binding(
    payload: dict[str, Any],
    *,
    request: SourceBindingRequest,
) -> SourceBindingResult:
    outcome = _dict(payload.get("outcome"), "outcome")
    kind = _text(outcome.get("kind"))
    if kind == "source_bindings":
        plan_output = provider_output.SourceBindingPlanOutput.parse(outcome)
        target_index = source_binding_target_index(request)
        candidates = source_candidates(request)
        review_scope = source_binding_review_scope(
            request,
            candidates_by_id=candidates,
            target_index=target_index,
        )
        (
            normalized_plan,
            effective_param_ids_by_index,
            population_choices_by_index,
        ) = _source_binding_payload_with_derived_finite_choices(
            plan_output,
            request,
            target_index=target_index,
            review_scope=review_scope,
            candidates=candidates,
        )
        return SourceBindingResult(
            outcome=_source_binding_plan(
                normalized_plan,
                request,
                target_index=target_index,
                review_scope=review_scope,
                candidates=candidates,
                effective_param_ids_by_index=effective_param_ids_by_index,
                population_choices_by_index=population_choices_by_index,
            )
        )
    if kind == "needs_clarification":
        return SourceBindingResult(outcome=_plan_clarification(outcome))
    if kind == "impossible":
        return SourceBindingResult(outcome=_plan_impossible(outcome, request=request))
    raise ValueError(f"unsupported source binding outcome: {kind}")


def _source_binding_payload_with_derived_finite_choices(
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
        raw_param_decisions = _normalized_param_decisions(
            parsed_invocation.param_decisions,
            parse_provider_output=True,
        )
        derived = _derived_finite_choice_param_decisions(
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


def _derived_finite_choice_param_decisions(
    raw_reviews: Any,
    *,
    candidate: Any,
    requested_fact_id: str,
    binding_target_id: str,
    request: SourceBindingRequest,
    review_scope: SourceBindingReviewScope,
    answer_population: provider_output.AnswerPopulationOutput,
    raw_param_decision_ids: tuple[str, ...],
) -> _DerivedFiniteChoiceParamDecisions:
    reviews = _dict(raw_reviews, "finite_choice_param_reviews")
    review_surface = source_binding_review_surface(candidate)
    scoped_test_ids_by_param = {
        param_id: test_ids
        for param_id in review_surface.finite_choice_params
        if (
            test_ids := review_scope.finite_choice_param_test_ids(
                binding_target_id,
                param_id,
            )
        )
    }
    finite_choice_axes = {
        param_id: review_surface.finite_choice_params[param_id]
        for param_id in scoped_test_ids_by_param
    }
    expected_param_ids = set(finite_choice_axes)
    if set(reviews) != expected_param_ids:
        raise ValueError(
            "finite choice param reviews must cover every finite-choice population param"
        )
    authored_choice_decisions = expected_param_ids & set(raw_param_decision_ids)
    if authored_choice_decisions:
        raise ValueError(
            "finite-choice population params must be derived from choice reviews"
        )
    output: dict[str, dict[str, Any]] = {}
    population_choices: list[RelationSourcePopulationChoice] = []
    population_roles_by_id = _candidate_population_roles_by_id(candidate)
    for param_id, axis in finite_choice_axes.items():
        out_of_scope_decisions = (
            review_scope.finite_choice_param_out_of_scope_decisions(
                binding_target_id,
                param_id,
            )
        )
        tests_by_id = _answer_population_tests_by_id(
            request=request,
            requested_fact_id=requested_fact_id,
            scoped_test_ids=scoped_test_ids_by_param[param_id],
        )
        _controlled_population_role(
            reviews.get(param_id),
            population_roles_by_id=population_roles_by_id,
            path=f"finite_choice_param_reviews.{param_id}",
        )
        include_values, exclude_values = _reviewed_choice_sets(
            reviews.get(param_id),
            axis=axis,
            tests_by_id=tests_by_id,
        )
        population_choices.append(
            RelationSourcePopulationChoice(
                controller_kind=PopulationChoiceControllerKind.QUERY_PARAM,
                controller_id=param_id,
                field_id=param_id,
                included_values=include_values,
                excluded_values=exclude_values,
                proof_refs=_population_choice_proof_refs(
                    f"population_choice:{param_id}",
                    out_of_scope_decisions,
                ),
                review_scope_decisions=_relation_review_scope_decisions(
                    out_of_scope_decisions
                ),
            )
        )
        if axis.can_be_omitted(include_values=include_values):
            continue
        output[param_id] = {
            "population_intent": _text(answer_population.intent_text),
            "match_basis_explanation": (
                "Derived from finite_choice_param_reviews because at least one shown "
                "choice does not satisfy the requested answer population tests."
            ),
            "population_choice_set": {
                "include_values": list(include_values),
                "exclude_values": list(exclude_values),
            },
        }
    return _DerivedFiniteChoiceParamDecisions(
        param_decisions=output,
        population_choices=tuple(population_choices),
    )


def _answer_population_tests_by_id(
    *,
    request: SourceBindingRequest,
    requested_fact_id: str,
    scoped_test_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not scoped_test_ids:
        return {}
    fact = next(
        (item for item in request.requested_facts if item.id == requested_fact_id),
        None,
    )
    if fact is None or fact.answer_population is None:
        raise ValueError("finite choice reviews require answer population tests")
    tests_by_id = membership_tests_by_key(fact.answer_population.membership_tests)
    return {test_id: tests_by_id[test_id] for test_id in scoped_test_ids}


def _population_choice_proof_refs(
    base_ref: str,
    out_of_scope_decisions: tuple[ReviewScopeDecision, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                base_ref,
                *(
                    ref
                    for decision in out_of_scope_decisions
                    for ref in decision.proof_refs
                ),
            )
        )
    )


def _relation_review_scope_decisions(
    decisions: tuple[ReviewScopeDecision, ...],
) -> tuple[RelationSourceReviewScopeDecision, ...]:
    return tuple(
        RelationSourceReviewScopeDecision(
            membership_test_id=decision.membership_test_id,
            decision=RelationReviewScopeDecisionKind(decision.decision.value),
            axis_kind=decision.axis_kind.value,
            axis_id=decision.axis_id,
            owner_surface_ids=decision.owner_surface_ids,
            proof_refs=decision.proof_refs,
        )
        for decision in decisions
    )


def _reviewed_choice_sets(
    raw_reviews: Any,
    *,
    axis: FiniteChoiceReviewAxis,
    tests_by_id: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    param_id = axis.axis_id
    raw_review = provider_output.FiniteChoiceParamReviewOutput.parse(raw_reviews)
    _validate_population_test_basis(
        raw_review.population_test_basis,
        tests_by_id=tests_by_id,
        path=f"finite_choice_param_reviews.{param_id}.population_test_basis",
    )
    reviews = _required_dicts(
        raw_review.choice_reviews,
        f"finite_choice_param_reviews.{param_id}.choice_reviews",
    )
    choices = axis.choices
    seen: set[str] = set()
    reviewed_effects: list[tuple[str, dict[str, str]]] = []
    choice_inclusions: dict[str, str] = {}
    for raw_item in reviews:
        raw = provider_output.FiniteChoiceReviewOutput.parse(raw_item)
        choice = canonical_param_value(raw.choice_option_id)
        if choice not in choices:
            raise ValueError("finite choice review references unknown choice")
        if choice in seen:
            raise ValueError("duplicate finite choice review")
        seen.add(choice)
        if not _text(raw.choice_domain_meaning).strip():
            raise ValueError("finite choice review requires domain meaning")
        if not _text(raw.choice_inclusion_basis).strip():
            raise ValueError("finite choice review requires inclusion basis")
        inclusion = _text(raw.choice_inclusion)
        if inclusion not in {"INCLUDE", "EXCLUDE"}:
            raise ValueError("finite choice review requires choice_inclusion")
        choice_inclusions[choice] = inclusion
        test_effects = _population_test_effects(
            raw.population_test_results,
            axis=axis,
            tests_by_id=tests_by_id,
            path=f"finite_choice_param_reviews.{param_id}.{choice}.population_test_results",
        )
        _validate_choice_inclusion_consistency(
            inclusion=inclusion,
            test_effects=test_effects,
            tests_by_id=tests_by_id,
            test_ids=_active_test_ids_for_choice(test_effects, tests_by_id),
            choice=choice,
        )
        reviewed_effects.append((choice, test_effects))
    if seen != set(choices):
        raise ValueError("finite choice reviews must cover every choice")
    active_test_ids = _active_membership_test_ids(
        reviewed_effects,
        tests_by_id=tests_by_id,
    )
    include_values: list[str] = []
    exclude_values: list[str] = []
    for choice, test_effects in reviewed_effects:
        if _choice_is_included(
            test_effects=test_effects,
            tests_by_id=tests_by_id,
            test_ids=active_test_ids,
            choice_inclusion=choice_inclusions.get(choice),
        ):
            include_values.append(choice)
            continue
        exclude_values.append(choice)
    if not include_values:
        raise ValueError("finite choice reviews must include at least one choice")
    return tuple(include_values), tuple(exclude_values)


def _validate_population_test_basis(
    raw_basis: Any,
    *,
    tests_by_id: dict[str, Any],
    path: str,
) -> None:
    basis = _dict(raw_basis, path)
    if set(basis) != set(tests_by_id):
        raise ValueError(
            "finite choice population test basis must cover membership tests"
        )
    for test_id in tests_by_id:
        item = provider_output.PopulationTestBasisOutput.parse(basis.get(test_id))
        if not _text(item.test_question).strip():
            raise ValueError("finite choice population test basis requires question")
        if not _text(item.role_scoped_test_question).strip():
            raise ValueError(
                "finite choice population test basis requires role-scoped question"
            )


def _active_membership_test_ids(
    reviewed_effects: list[tuple[str, dict[str, str]]],
    *,
    tests_by_id: dict[str, Any],
) -> tuple[str, ...]:
    output: list[str] = []
    for test_id in tests_by_id:
        effects = tuple(test_effects[test_id] for _, test_effects in reviewed_effects)
        is_active = any(effect != "DOES_NOT_DECIDE_TEST" for effect in effects)
        if is_active:
            output.append(test_id)
    return tuple(output)


def _population_test_effects(
    raw_results: Any,
    *,
    axis: FiniteChoiceReviewAxis,
    tests_by_id: dict[str, Any],
    path: str,
) -> dict[str, str]:
    results = _dict(raw_results, path)
    seen: set[str] = set()
    effects: dict[str, str] = {}
    expected = set(tests_by_id)
    if set(results) != expected:
        raise ValueError("finite choice reviews must answer membership tests")
    for test_id in tests_by_id:
        raw_value = results.get(test_id)
        if test_id not in tests_by_id:
            raise ValueError("finite choice review references unknown population test")
        if test_id in seen:
            raise ValueError("duplicate finite choice population test result")
        seen.add(test_id)
        raw = (
            provider_output.NormalInstanceTestResultOutput.parse(raw_value)
            if tests_by_id[test_id].kind
            == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
            else provider_output.StandardPopulationTestResultOutput.parse(raw_value)
        )
        effect = _validate_normal_instance_test_effect(
            raw,
            test=tests_by_id[test_id],
            axis=axis,
            path=f"{path}.{test_id}",
        )
        if effect is None:
            if not _text(raw.test_basis).strip():
                raise ValueError("finite choice population test requires basis")
            if not _text(raw.population_consequence).strip():
                raise ValueError(
                    "finite choice population test requires population consequence"
                )
            effect = _text(raw.test_effect)
            if effect not in {
                "SATISFIES_TEST",
                "CONFLICTS_WITH_TEST",
                "DOES_NOT_DECIDE_TEST",
                "UNKNOWN_TEST_EFFECT",
            }:
                raise ValueError("unsupported finite choice population test effect")
        effects[test_id] = effect
    return effects


def _validate_normal_instance_test_effect(
    raw: Any,
    *,
    test: Any,
    axis: FiniteChoiceReviewAxis,
    path: str,
) -> str | None:
    if test.kind != AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD:
        return None
    if test.polarity != AnswerPopulationMembershipTestPolarity.MUST_PASS:
        raise ValueError("normal instance review requires must-pass guard")
    profile = getattr(test, "normal_instance_profile", None)
    if profile is None:
        raise ValueError("normal instance review requires normal instance profile")
    disposition = provider_output.NormalInstanceDispositionOutput.parse(raw.disposition)
    if not _text(raw.role_match_basis).strip():
        raise ValueError("normal instance role match requires reason")
    role_effect = _normal_instance_role_match_effect(
        disposition,
        axis=axis,
        test_id=membership_test_key(test),
        path=f"{path}.disposition",
    )
    override_applies = raw.explicit_user_override_applies
    if not isinstance(override_applies, bool):
        raise ValueError("normal instance review requires explicit override decision")
    override_evidence = _normal_instance_override_evidence(
        raw.explicit_user_override_evidence,
        path=f"{path}.explicit_user_override_evidence",
    )
    if override_applies and not override_evidence:
        raise ValueError("normal instance review override evidence is required")
    if override_applies and not role_effect["matched"]:
        raise ValueError("normal instance review override requires matched role")
    if not override_applies and override_evidence:
        raise ValueError(
            "normal instance review override evidence conflicts with decision"
        )
    if not _text(raw.population_consequence).strip():
        raise ValueError("normal instance review requires population consequence")
    effect = _text(disposition.test_effect)
    if effect not in {
        "SATISFIES_TEST",
        "CONFLICTS_WITH_TEST",
        "DOES_NOT_DECIDE_TEST",
        "UNKNOWN_TEST_EFFECT",
    }:
        raise ValueError("unsupported finite choice population test effect")
    _validate_normal_instance_test_effect_consistency(
        effect=effect,
        matched=role_effect["matched"],
        unknown=role_effect["unknown"],
        explicit_user_override_applies=override_applies,
    )
    return effect


def _normal_instance_role_match_effect(
    review: Any,
    *,
    axis: FiniteChoiceReviewAxis,
    test_id: str,
    path: str,
) -> dict[str, bool]:
    profile = axis.normal_instance_profile(test_id)
    if profile is None:
        raise ValueError("normal instance review requires role profile")
    matched_role = _text(review.matched_excluded_role)
    allowed_roles = set(profile.excluded_role_ids)
    if not allowed_roles:
        raise ValueError("normal instance role profile requires excluded roles")
    if matched_role == NORMAL_INSTANCE_NO_EXCLUDED_ROLE:
        return {"matched": False, "unknown": False}
    if matched_role == NORMAL_INSTANCE_UNKNOWN_EXCLUDED_ROLE:
        return {"matched": False, "unknown": True}
    if matched_role not in allowed_roles:
        raise ValueError("normal instance review references unknown excluded role")
    return {"matched": True, "unknown": False}


def _validate_normal_instance_test_effect_consistency(
    *,
    effect: str,
    matched: bool,
    unknown: bool,
    explicit_user_override_applies: bool,
) -> None:
    if unknown:
        if effect != "UNKNOWN_TEST_EFFECT":
            raise ValueError(
                "normal instance review effect conflicts with unknown role"
            )
        return
    if matched and explicit_user_override_applies:
        if effect != "SATISFIES_TEST":
            raise ValueError("normal instance review effect conflicts with override")
        return
    if matched:
        if effect != "CONFLICTS_WITH_TEST":
            raise ValueError("normal instance review effect conflicts with role match")
        return
    if effect not in {"SATISFIES_TEST", "DOES_NOT_DECIDE_TEST"}:
        raise ValueError("normal instance review effect conflicts with role match")


def _normal_instance_override_evidence(
    raw: Any,
    *,
    path: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, list):
        raise ValueError("normal instance review requires override evidence")
    output: list[dict[str, str]] = []
    for item in raw:
        evidence = provider_output.NormalInstanceOverrideEvidenceOutput.parse(item)
        source_text = _text(evidence.source_text)
        if not source_text.strip():
            raise ValueError(
                "normal instance review override evidence requires source text"
            )
        reason = NormalInstanceExplicitOverrideReason(_text(evidence.reason))
        output.append({"source_text": source_text, "reason": reason.value})
    return tuple(output)


def _controlled_population_role(
    raw: Any,
    *,
    population_roles_by_id: dict[str, dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    if not population_roles_by_id:
        raise ValueError("finite choice population tests require population_roles")
    role = provider_output.FiniteChoiceParamReviewOutput.parse(raw)
    role_id = _text(role.controlled_population_role_id)
    expected = population_roles_by_id.get(role_id)
    if expected is None:
        raise ValueError("finite choice param references unknown role")
    if not role_id:
        raise ValueError("finite choice param requires role id")
    if not _text(role.role_selection_basis).strip():
        raise ValueError("finite choice param requires role selection basis")
    return expected


def _candidate_population_roles_by_id(candidate: Any) -> dict[str, dict[str, Any]]:
    return {
        role.role_id: {"role_kind": "", "role_text": ""}
        for role in source_binding_review_surface(candidate).population_roles
    }


def _choice_is_included(
    *,
    test_effects: dict[str, str],
    tests_by_id: dict[str, Any],
    test_ids: tuple[str, ...],
    choice_inclusion: str | None,
) -> bool:
    tests_decide = _population_tests_decide(
        test_effects=test_effects,
        test_ids=test_ids,
    )
    if not tests_decide:
        if choice_inclusion is None:
            return True
        return choice_inclusion == "INCLUDE"
    return _population_tests_allow_choice(
        test_effects=test_effects,
        tests_by_id=tests_by_id,
        test_ids=test_ids,
    )


def _population_tests_decide(
    *,
    test_effects: dict[str, str],
    test_ids: tuple[str, ...],
) -> bool:
    return any(
        test_effects[test_id] in {"SATISFIES_TEST", "CONFLICTS_WITH_TEST"}
        for test_id in test_ids
    )


def _population_tests_allow_choice(
    *,
    test_effects: dict[str, str],
    tests_by_id: dict[str, Any],
    test_ids: tuple[str, ...],
) -> bool:
    for test_id in test_ids:
        test = tests_by_id[test_id]
        effect = test_effects[test_id]
        if effect == "DOES_NOT_DECIDE_TEST":
            continue
        if effect == "UNKNOWN_TEST_EFFECT":
            return False
        if (
            test.polarity == AnswerPopulationMembershipTestPolarity.MUST_PASS
            and effect == "CONFLICTS_WITH_TEST"
        ):
            return False
        if (
            test.polarity == AnswerPopulationMembershipTestPolarity.MUST_FAIL
            and effect == "SATISFIES_TEST"
        ):
            return False
    return True


def _active_test_ids_for_choice(
    test_effects: dict[str, str],
    tests_by_id: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        test_id
        for test_id in tests_by_id
        if test_effects[test_id] != "DOES_NOT_DECIDE_TEST"
    )


def _validate_choice_inclusion_consistency(
    *,
    inclusion: str,
    test_effects: dict[str, str],
    tests_by_id: dict[str, Any],
    test_ids: tuple[str, ...],
    choice: str,
) -> None:
    for test_id in test_ids:
        test = tests_by_id[test_id]
        effect = test_effects[test_id]
        if effect not in {"SATISFIES_TEST", "CONFLICTS_WITH_TEST"}:
            continue
        if (
            test.polarity == AnswerPopulationMembershipTestPolarity.MUST_PASS
            and effect == "CONFLICTS_WITH_TEST"
            and inclusion != "EXCLUDE"
        ):
            raise ValueError(
                f"finite choice review {choice} inclusion conflicts with test {test_id}"
            )
        if (
            test.polarity == AnswerPopulationMembershipTestPolarity.MUST_FAIL
            and effect == "SATISFIES_TEST"
            and inclusion != "EXCLUDE"
        ):
            raise ValueError(
                f"finite choice review {choice} inclusion conflicts with test {test_id}"
            )


def _source_binding_plan(
    payload: provider_output.SourceBindingPlanOutput,
    request: SourceBindingRequest,
    *,
    target_index: SourceBindingTargetIndex,
    review_scope: SourceBindingReviewScope,
    candidates: dict[str, Any],
    effective_param_ids_by_index: dict[int, tuple[str, ...]] | None = None,
    population_choices_by_index: (
        dict[int, tuple[RelationSourcePopulationChoice, ...]] | None
    ) = None,
) -> SourceBindingPlan:
    value_candidates_by_relation_id = _value_candidates_by_source_relation_id(
        candidates.values()
    )
    requested_fact_output_ids = {
        fact.id: {output.id for output in fact.answer_outputs}
        for fact in request.requested_facts
    }
    metric_fit_reviews = _metric_fit_interpretations_by_requested_fact(
        payload,
        request=request,
    )
    seen_binding_target_ids: set[str] = set()
    output: list[BoundSource] = []
    for index, parsed_invocation in enumerate(payload.source_invocations, start=1):
        target = target_index.require(
            _text(parsed_invocation.binding_target_id)
        )
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
        answer_population, population_binding = _answer_population(
            parsed_invocation.answer_population,
            request=request,
            requested_fact_id=requested_fact_id,
            candidate=candidate,
        )
        param_decisions = _param_decision_binding_sets(
            parsed_invocation.param_decisions,
            candidate=candidate,
            available_values=request.available_values,
            answer_population=answer_population,
            effective_param_ids=(effective_param_ids_by_index or {}).get(index),
        )
        row_predicates = _row_predicate_filters(
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
        param_binding_sets = tuple(
            _merged_param_bindings(
                base_param_bindings,
                model_param_bindings,
            )
            for base_param_bindings in candidate_base_binding_sets
            for model_param_bindings in param_decisions.binding_sets
        )
        population_choices = (
            *((population_choices_by_index or {}).get(index, ())),
            *row_predicates.population_choices,
        )
        row_filters = row_predicates.filters
        fulfillments = _source_fulfillments(
            parsed_invocation.fulfillment_decisions,
            requested_fact_id=requested_fact_id,
            answer_output_ids=set(target.answer_output_ids),
            candidate=candidate,
            plan_shape=target.plan_shape,
            metric_fit_reviews_by_requested_output=metric_fit_reviews,
        )
        source, source_invocations = _bound_relation_source(
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
        evidence_items = _candidate_source_evidence_items(candidate)
        available_fields = _candidate_source_fields(
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
            cardinality=_candidate_cardinality(candidate),
            evidence_items=evidence_items,
            available_field_ids=tuple(
                sorted(field.field_id for field in available_fields)
            ),
            available_fields=available_fields,
            applied_filters=_candidate_applied_filters(candidate),
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
        raise ValueError("source binding must cover one complete source binding role set")


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
            evidence_items=_candidate_source_evidence_items(candidate),
        )
        for index, candidate in enumerate(candidates)
        if candidate.value_id
        and _candidate_value_is_used_by_bound_source(
            candidate,
            bound,
        )
    )


def _candidate_value_is_used_by_bound_source(
    candidate: Any,
    bound: BoundSource,
) -> bool:
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        return True
    source_field_id = str(payload.get("source_field_id") or "")
    if not source_field_id:
        return True
    answer_field_ids = {
        item.field_id
        for item in bound.evidence_items
        if item.evidence_id
        in {
            evidence_id
            for fulfillment in bound.fulfillments
            for evidence_id in (
                *fulfillment.metric_measure_evidence_ids,
                *fulfillment.row_count_basis_evidence_ids,
                *fulfillment.group_key_evidence_ids,
            )
        }
        and item.field_id
    }
    return not answer_field_ids or source_field_id in answer_field_ids


def _answer_population(
    raw_value: Any,
    *,
    request: SourceBindingRequest,
    requested_fact_id: str,
    candidate: Any,
) -> tuple[AnswerPopulation, dict[str, Any]]:
    raw = provider_output.AnswerPopulationOutput.parse(raw_value)
    population_binding_id = _text(raw.population_binding_id)
    binding = _candidate_population_binding(
        population_binding_id,
        candidate=candidate,
    )
    PopulationBindingIndex.from_request(request).validate_selection(
        requested_fact_id=requested_fact_id,
        candidate=candidate,
        population_binding_id=population_binding_id,
    )
    intent_text = _text(raw.intent_text)
    return (
        AnswerPopulation(
            population_binding_id=population_binding_id,
            intent_text=intent_text,
            match_basis_explanation=_text(raw.match_basis_explanation),
        ),
        binding,
    )


def _bound_relation_source(
    *,
    candidate: Any,
    population_binding: dict[str, Any],
    param_binding_sets: tuple[tuple[EndpointParamBinding, ...], ...],
    population_choices: tuple[RelationSourcePopulationChoice, ...],
) -> tuple[Any, tuple[Any, ...]]:
    if (
        str(population_binding.get("kind") or "") == "exact_row_set"
        and str(getattr(candidate, "kind", "") or "") == "prior_answer_rows"
    ):
        basis = _dict(population_binding.get("basis"), "answer_population.basis")
        memory_relation_id = _text(basis.get("memory_relation_id"))
        return (
            RelationSource(
                kind=SourceKind.MEMORY_READ,
                memory_relation_id=memory_relation_id,
                population_choices=population_choices,
                proof_refs=_required_strings(
                    basis.get("proof_refs"),
                    "answer_population.basis.proof_refs",
                ),
            ),
            (),
        )
    source = candidate.source
    source_invocations: tuple[Any, ...] = ()
    if source is not None:
        source_invocations = tuple(
            replace(
                source,
                param_bindings=param_bindings,
                population_choices=population_choices,
            )
            for param_bindings in param_binding_sets
        )
        source = source_invocations[0]
    return source, source_invocations


def _candidate_population_binding(
    population_binding_id: str,
    *,
    candidate: Any,
) -> dict[str, Any]:
    bindings = {
        binding_id: item
        for item in getattr(candidate, "population_bindings", ())
        if isinstance(item, dict)
        for binding_id in (str(item.get("population_binding_id") or ""),)
        if binding_id
    }
    binding = bindings.get(population_binding_id)
    if binding is None:
        raise ValueError("answer population references unknown population binding")
    return binding


def _source_fulfillments(
    raw_fulfillment_decisions: Any,
    *,
    requested_fact_id: str,
    answer_output_ids: set[str],
    candidate: Any,
    plan_shape: str,
    metric_fit_reviews_by_requested_output: dict[str, dict[str, dict[str, str]]],
) -> tuple[SourceFulfillment, ...]:
    output: list[SourceFulfillment] = []
    seen_support_set_ids: set[str] = set()
    raw_decisions = _dict(raw_fulfillment_decisions, "fulfillment_decisions")
    if not answer_output_ids:
        if raw_decisions:
            raise ValueError("binding target does not allow answer fulfillment")
        return ()
    if not raw_decisions:
        raise ValueError("fulfillment_decisions must contain at least one value")
    for answer_output_id, raw_value in raw_decisions.items():
        raw = provider_output.FulfillmentDecisionOutput.parse(raw_value)
        if answer_output_id not in answer_output_ids:
            raise ValueError("source fulfillment references unknown answer output")
        choice_id = _text(raw.fulfillment_choice_id)
        support_set_id = _source_fulfillment_support_set_id(
            choice_id,
            answer_output_id=answer_output_id,
            candidate=candidate,
        )
        if support_set_id in seen_support_set_ids:
            raise ValueError("duplicate source fulfillment support set")
        seen_support_set_ids.add(support_set_id)
        slots = _source_fulfillment_support_set_slots(
            support_set_id,
            answer_output_id=answer_output_id,
            candidate=candidate,
        )
        explanation = _text(raw.match_basis_explanation)
        selected_metric_measure_evidence_ids = tuple(
            dict.fromkeys(_slot_evidence_ids(slots, key="metric_measure_evidence"))
        )
        selected_row_count_basis_evidence_ids = tuple(
            dict.fromkeys(_slot_evidence_ids(slots, key="row_count_basis_evidence"))
        )
        selected_metric_measure_evidence_ids = _fitting_metric_measure_evidence_ids(
            requested_fact_id=requested_fact_id,
            answer_output_id=answer_output_id,
            selected_metric_measure_evidence_ids=(selected_metric_measure_evidence_ids),
            metric_fit_reviews_by_requested_output=(
                metric_fit_reviews_by_requested_output
            ),
        )
        if _plan_shape_uses_row_count_as_metric(plan_shape):
            selected_row_count_basis_evidence_ids = (
                _fitting_row_count_basis_evidence_ids(
                    requested_fact_id=requested_fact_id,
                    answer_output_id=answer_output_id,
                    selected_row_count_basis_evidence_ids=(
                        selected_row_count_basis_evidence_ids
                    ),
                    metric_fit_reviews_by_requested_output=(
                        metric_fit_reviews_by_requested_output
                    ),
                )
            )
        if plan_shape in {"aggregate_by_group", "ranked_aggregate"}:
            selected_metric_measure_evidence_ids = tuple(
                dict.fromkeys(
                    (
                        *selected_metric_measure_evidence_ids,
                        *_candidate_fitting_metric_measure_evidence_ids(
                            requested_fact_id=requested_fact_id,
                            answer_output_id=answer_output_id,
                            candidate_metric_measure_evidence_ids=(
                                _candidate_metric_measure_evidence_ids(candidate)
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
                        *_candidate_fitting_row_count_basis_evidence_ids(
                            requested_fact_id=requested_fact_id,
                            answer_output_id=answer_output_id,
                            candidate_row_count_basis_evidence_ids=(
                                _candidate_row_count_basis_evidence_ids(candidate)
                            ),
                            metric_fit_reviews_by_requested_output=(
                                metric_fit_reviews_by_requested_output
                            ),
                        ),
                    )
                )
            )
            selected_group_key_evidence_ids = tuple(
                dict.fromkeys(
                    (
                        *_slot_evidence_ids(slots, key="group_key_evidence"),
                        *_candidate_support_set_evidence_ids(
                            candidate,
                            answer_output_id=answer_output_id,
                            key="group_key_evidence",
                        ),
                    )
                )
            )
        else:
            selected_group_key_evidence_ids = _slot_evidence_ids(
                slots,
                key="group_key_evidence",
            )
        output.append(
            SourceFulfillment(
                requested_fact_id=requested_fact_id,
                answer_output_id=answer_output_id,
                match_basis_explanation=explanation,
                fulfillment_support_set_id=support_set_id,
                metric_measure_evidence_ids=(selected_metric_measure_evidence_ids),
                row_count_basis_evidence_ids=(selected_row_count_basis_evidence_ids),
                scope_evidence_ids=_slot_evidence_ids(slots, key="scope_evidence"),
                group_key_evidence_ids=selected_group_key_evidence_ids,
                metric_fit_bases=_source_metric_fit_bases(
                    requested_fact_id=requested_fact_id,
                    answer_output_id=answer_output_id,
                    evidence_ids=(
                        *selected_metric_measure_evidence_ids,
                        *selected_row_count_basis_evidence_ids,
                    ),
                    metric_fit_reviews_by_requested_output=(
                        metric_fit_reviews_by_requested_output
                    ),
                ),
            )
        )
    return tuple(output)


def _candidate_support_set_evidence_ids(
    candidate: Any,
    *,
    answer_output_id: str,
    key: str,
) -> tuple[str, ...]:
    payload = getattr(candidate, "payload", None)
    available = _candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            evidence_id
            for support_set in (payload or {}).get("fulfillment_support_sets") or ()
            if isinstance(support_set, dict)
            and str(support_set.get("answer_output_id") or "") == answer_output_id
            for slot in support_set.get("fulfillment_slots") or ()
            if isinstance(slot, dict)
            for item in slot.get(key) or ()
            if isinstance(item, dict)
            and not (
                key == "group_key_evidence"
                and str(item.get("type") or "").lower() == "row_population"
            )
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id and evidence_id in available
        )
    )


def _candidate_metric_measure_evidence_ids(candidate: Any) -> tuple[str, ...]:
    available = _candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            evidence_id
            for item in _candidate_evidence_items(candidate)
            if evidence_item_can_measure(item)
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id and evidence_id in available
        )
    )


def _plan_shape_uses_row_count_as_metric(plan_shape: str) -> bool:
    return plan_shape in {"aggregate_scalar", "aggregate_by_group", "ranked_aggregate"}


def _candidate_row_count_basis_evidence_ids(candidate: Any) -> tuple[str, ...]:
    available = _candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            evidence_id
            for item in _candidate_evidence_items(candidate)
            if str(item.get("type") or "").lower() == "row_population"
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id and evidence_id in available
        )
    )


def _candidate_evidence_items(candidate: Any) -> tuple[dict[str, Any], ...]:
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        return ()
    return tuple(
        item for item in payload.get("evidence_items") or () if isinstance(item, dict)
    )


def _metric_fit_interpretations_by_requested_fact(
    payload: provider_output.SourceBindingPlanOutput,
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


def _fitting_metric_measure_evidence_ids(
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


def _candidate_fitting_metric_measure_evidence_ids(
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


def _fitting_row_count_basis_evidence_ids(
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


def _candidate_fitting_row_count_basis_evidence_ids(
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


def _source_metric_fit_bases(
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


def _source_fulfillment_support_set_slots(
    support_set_id: str,
    *,
    answer_output_id: str,
    candidate: Any,
) -> tuple[dict[str, Any], ...]:
    support_set = _candidate_fulfillment_support_sets_by_id(candidate).get(
        support_set_id
    )
    if support_set is None:
        raise ValueError("source fulfillment references unknown support set")
    if str(support_set.get("answer_output_id") or "") != answer_output_id:
        raise ValueError("source fulfillment support set mismatches answer output")
    selected_slots = [
        slot
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
    ]
    evidence_ids = tuple(
        evidence_id
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "group_key_evidence",
        )
        for evidence_id in _slot_evidence_ids(tuple(selected_slots), key=key)
    )
    if not evidence_ids:
        raise ValueError("source fulfillment slot requires evidence")
    available = _candidate_evidence_ids(candidate)
    all_slot_evidence = {
        evidence_id
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "scope_evidence",
            "group_key_evidence",
        )
        for evidence_id in _slot_evidence_ids(tuple(selected_slots), key=key)
    }
    if all_slot_evidence - available:
        raise ValueError("source fulfillment slot references unknown evidence")
    return tuple(selected_slots)


def _source_fulfillment_support_set_id(
    choice_id: str,
    *,
    answer_output_id: str,
    candidate: Any,
) -> str:
    support_set = _candidate_fulfillment_support_sets_by_choice_id(candidate).get(
        choice_id
    )
    if support_set is None:
        raise ValueError("source fulfillment references unknown choice")
    if str(support_set.get("answer_output_id") or "") != answer_output_id:
        raise ValueError("source fulfillment choice mismatches answer output")
    support_set_id = str(support_set.get("fulfillment_support_set_id") or "")
    if not support_set_id:
        raise ValueError("source fulfillment choice is missing internal support set")
    return support_set_id


def _slot_evidence_ids(
    slots: tuple[dict[str, Any], ...], *, key: str
) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for slot in slots
        for item in slot.get(key) or ()
        if isinstance(item, dict)
        for evidence_id in (str(item.get("evidence_id") or ""),)
        if evidence_id
    )


def _candidate_fulfillment_support_sets_by_id(
    candidate: Any,
) -> dict[str, dict[str, Any]]:
    payload = getattr(candidate, "payload", None)
    return {
        support_set_id: item
        for item in (payload or {}).get("fulfillment_support_sets") or ()
        if isinstance(item, dict)
        for support_set_id in (str(item.get("fulfillment_support_set_id") or ""),)
        if support_set_id
    }


def _candidate_fulfillment_support_sets_by_choice_id(
    candidate: Any,
) -> dict[str, dict[str, Any]]:
    payload = getattr(candidate, "payload", None)
    return {
        choice_id: item
        for item in (payload or {}).get("fulfillment_support_sets") or ()
        if isinstance(item, dict)
        for choice_id in (str(item.get("fulfillment_choice_id") or ""),)
        if choice_id
    }


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


def _candidate_evidence_ids(candidate: Any) -> set[str]:
    payload = getattr(candidate, "payload", None)
    evidence_items = payload.get("evidence_items") if isinstance(payload, dict) else ()
    if evidence_items:
        return {
            evidence_id
            for item in evidence_items or ()
            if isinstance(item, dict)
            for evidence_id in (str(item.get("evidence_id") or "").strip(),)
            if evidence_id
        }
    field_ids = _candidate_field_ids(candidate)
    if field_ids:
        return field_ids
    value_id = str(getattr(candidate, "value_id", "") or "").strip()
    return {value_id} if value_id else set()


def _candidate_cardinality(candidate: Any) -> str:
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("cardinality") or "").strip()


def _candidate_applied_filters(candidate: Any) -> tuple[dict[str, Any], ...]:
    payload = getattr(candidate, "payload", None)
    filters = payload.get("applied_filters") if isinstance(payload, dict) else ()
    return tuple(dict(item) for item in filters or () if isinstance(item, dict))


def _candidate_field_ids(candidate: Any) -> set[str]:
    return {
        field_id
        for field in candidate.fields
        if isinstance(field, dict)
        for field_id in (str(field.get("field_id") or field.get("id") or "").strip(),)
        if field_id
    }


def _candidate_source_fields(
    candidate: Any,
    *,
    evidence_items: tuple[SourceEvidenceItem, ...] = (),
    fulfillments: tuple[SourceFulfillment, ...] = (),
    row_filters: tuple[RelationSourceRowFilter, ...] = (),
) -> tuple[SourceField, ...]:
    fields = [
        SourceField(
            field_id=field_id,
            type=str(field.get("type") or ""),
            roles=tuple(str(role) for role in field.get("roles") or ()),
            label=str(field.get("label") or ""),
            row_cardinality=str(field.get("row_cardinality") or ""),
            identity=_identity_metadata(field.get("identity")),
        )
        for field in candidate.fields
        if isinstance(field, dict)
        for field_id in (str(field.get("field_id") or field.get("id") or "").strip(),)
        if field_id and _candidate_field_selectable_for_planning(field)
    ]
    existing_field_ids = {field.field_id for field in fields}
    selected_evidence_ids = {
        evidence_id
        for fulfillment in fulfillments
        for evidence_id in fulfillment.all_evidence_ids()
    }
    fields.extend(
        SourceField(
            field_id=item.field_id,
            type=item.type,
            row_cardinality=item.row_cardinality,
            identity=item.identity,
        )
        for item in evidence_items
        if item.evidence_id in selected_evidence_ids
        and item.field_id
        and item.type != "row_population"
        and _field_type_selectable_for_planning(item.type)
        and item.field_id not in existing_field_ids
    )
    existing_field_ids.update(field.field_id for field in fields)
    predicate_types = {
        axis.field_id: axis.field_type
        for axis in source_binding_review_surface(candidate).row_predicates.values()
    }
    fields.extend(
        SourceField(
            field_id=row_filter.field_id,
            type=predicate_types.get(row_filter.field_id, ""),
            roles=("predicate",),
        )
        for row_filter in row_filters
        if row_filter.field_id and row_filter.field_id not in existing_field_ids
    )
    return tuple(fields)


def _candidate_field_selectable_for_planning(field: dict[str, Any]) -> bool:
    return _field_type_selectable_for_planning(str(field.get("type") or ""))


def _field_type_selectable_for_planning(field_type: str) -> bool:
    return field_type.lower() != "object"


def _candidate_source_evidence_items(candidate: Any) -> tuple[SourceEvidenceItem, ...]:
    payload = getattr(candidate, "payload", None)
    evidence_items = payload.get("evidence_items") if isinstance(payload, dict) else ()
    row_cardinality_by_field_id = {
        str(field.get("field_id") or field.get("id") or "").strip(): str(
            field.get("row_cardinality") or ""
        ).strip()
        for field in getattr(candidate, "fields", ())
        if isinstance(field, dict)
        and str(field.get("field_id") or field.get("id") or "").strip()
    }
    return tuple(
        SourceEvidenceItem(
            evidence_id=evidence_id,
            field_id=str(item.get("field_id") or "").strip(),
            value_id=str(item.get("value_id") or "").strip(),
            type=str(item.get("type") or "").strip(),
            row_cardinality=(
                str(item.get("row_cardinality") or "").strip()
                or row_cardinality_by_field_id.get(
                    str(item.get("field_id") or "").strip(), ""
                )
            ),
            row_source_id=str(item.get("row_source_id") or "").strip(),
            identity=_identity_metadata(item.get("identity")),
        )
        for item in evidence_items or ()
        if isinstance(item, dict)
        for evidence_id in (str(item.get("evidence_id") or "").strip(),)
        if evidence_id
    )


def _identity_metadata(raw: Any) -> IdentityMetadata | None:
    if not isinstance(raw, dict) or not raw:
        return None
    entity_ref = str(raw.get("entity_ref") or raw.get("entityRef") or "").strip()
    identity_field = str(raw.get("identity_field") or raw.get("idField") or "").strip()
    if not entity_ref or not identity_field:
        return None
    return IdentityMetadata(
        entity_ref=entity_ref,
        identity_field=identity_field,
        primary_key=bool(raw.get("primary_key") or raw.get("primaryKey")),
        stable=bool(raw.get("stable", True)),
    )


def _param_decision_binding_sets(
    raw_decisions: Any,
    *,
    candidate: Any,
    available_values: tuple[FactValue, ...],
    answer_population: AnswerPopulation,
    effective_param_ids: tuple[str, ...] | None = None,
) -> _ParamDecisionParse:
    params_by_id = {
        str(param.get("param_id") or ""): param
        for param in candidate.params
        if isinstance(param, dict) and _param_is_model_bindable(param)
    }
    if effective_param_ids is not None:
        effective = set(effective_param_ids)
        params_by_id = {
            param_id: param
            for param_id, param in params_by_id.items()
            if param_id in effective
        }
    options_by_id = _param_decision_options_by_id(params_by_id)
    output: list[tuple[tuple[EndpointParamBinding, ...], ...]] = []
    normalized_decisions = _normalized_param_decisions(raw_decisions)
    if not params_by_id and not normalized_decisions:
        return _ParamDecisionParse(binding_sets=((),))
    for param_id, raw in normalized_decisions.items():
        if param_id not in params_by_id:
            raise ValueError("source param decision references unknown param")
        match_basis_explanation = _text(raw.get("match_basis_explanation")).strip()
        if not match_basis_explanation:
            raise ValueError("source param decision requires match basis explanation")
        _validate_param_population_intent(raw)
        param = params_by_id[param_id]
        if param.get("choices") and "population_choice_set" in raw:
            choice_set = _population_choice_set(raw, param=param)
            output.append(
                _param_binding_sets(
                    param_id=param_id,
                    value=choice_set.included_values,
                    param=param,
                )
            )
            continue
        decision_id = _text(raw.get("param_decision_id"))
        option = options_by_id.get(decision_id)
        if option is None:
            raise ValueError("source param decision references unknown option")
        if str(option.get("param_id") or "") != param_id:
            raise ValueError("source param decision references mismatched param")
        decision = str(option.get("decision") or "")
        if decision == "use_default":
            if not param_has_default_value(param):
                raise ValueError(
                    "source param decision uses default but param has no default"
                )
            output.append(
                (
                    (
                        EndpointParamBinding(
                            param_id=param_id,
                            value=option.get("value", param.get("default")),
                        ),
                    ),
                )
            )
            continue
        if decision != "bind":
            raise ValueError("unsupported source param decision")
        value = str(option.get("value") or "")
        choices = param.get("choices")
        if choices and value not in {str(choice) for choice in choices}:
            raise ValueError("source binding param value is not an available choice")
        proof_refs: tuple[str, ...] = ()
        binding_values = param.get("binding_values")
        if binding_values:
            allowed_value_ids = {
                str(item.get("value") or "")
                for item in binding_values
                if isinstance(item, dict)
            }
            if value not in allowed_value_ids:
                raise ValueError("source binding param value is not bindable")
            value, proof_refs = _resolved_binding_value(
                value,
                param=param,
                option=option,
                available_values=available_values,
            )
        output.append(
            _param_binding_sets(
                param_id=param_id,
                value=value,
                param=param,
                proof_refs=proof_refs,
            )
        )
    missing_param_ids = {
        param_id
        for param_id, param in params_by_id.items()
        if param_id not in normalized_decisions
        and _param_requires_explicit_decision(param)
    }
    if missing_param_ids:
        raise ValueError("source binding missing explicit param decision")
    if not output:
        return _ParamDecisionParse(binding_sets=((),))
    return _ParamDecisionParse(
        binding_sets=tuple(
            tuple(binding for group in groups for binding in group)
            for groups in product(*output)
        ),
    )


def _row_predicate_filters(
    raw_reviews: Any,
    *,
    candidate: Any,
    request: SourceBindingRequest,
    requested_fact_id: str,
    binding_target_id: str,
    review_scope: SourceBindingReviewScope,
) -> _RowPredicateParse:
    reviews = _dict(raw_reviews, "row_predicate_reviews")
    candidate_predicates_by_id = source_binding_review_surface(candidate).row_predicates
    scoped_test_ids_by_predicate = {
        predicate_id: test_ids
        for predicate_id in candidate_predicates_by_id
        if (
            test_ids := review_scope.row_predicate_test_ids(
                binding_target_id,
                predicate_id,
            )
        )
    }
    predicates_by_id = {
        predicate_id: candidate_predicates_by_id[predicate_id]
        for predicate_id in scoped_test_ids_by_predicate
    }
    if not predicates_by_id and not reviews:
        return _RowPredicateParse()
    missing_predicate_ids = set(predicates_by_id) - set(reviews)
    if missing_predicate_ids:
        raise ValueError("source binding missing row predicate review")
    filters: list[RelationSourceRowFilter] = []
    population_choices: list[RelationSourcePopulationChoice] = []
    for predicate_id, raw in reviews.items():
        predicate = predicates_by_id.get(predicate_id)
        if predicate is None:
            raise ValueError("row predicate review references unknown predicate")
        out_of_scope_decisions = review_scope.row_predicate_out_of_scope_decisions(
            binding_target_id,
            predicate_id,
        )
        tests_by_id = _answer_population_tests_by_id(
            request=request,
            requested_fact_id=requested_fact_id,
            scoped_test_ids=scoped_test_ids_by_predicate[predicate_id],
        )
        allowed_values = predicate.allowed_values
        values = _row_predicate_include_values(
            raw,
            allowed_values=allowed_values,
            tests_by_id=tests_by_id,
            path=f"row_predicate_reviews.{predicate_id}",
        )
        excluded_values = tuple(
            value for value in allowed_values if value not in values
        )
        field_id = predicate.field_id
        if not field_id:
            raise ValueError("row predicate missing field")
        population_choices.append(
            RelationSourcePopulationChoice(
                controller_kind=PopulationChoiceControllerKind.ROW_PREDICATE,
                controller_id=predicate_id,
                field_id=field_id,
                included_values=values,
                excluded_values=excluded_values,
                proof_refs=_population_choice_proof_refs(
                    f"row_predicate:{predicate_id}",
                    out_of_scope_decisions,
                ),
                review_scope_decisions=_relation_review_scope_decisions(
                    out_of_scope_decisions
                ),
            )
        )
        if set(values) == set(allowed_values):
            continue
        filters.append(
            RelationSourceRowFilter(
                field_id=field_id,
                operator=predicate.operator,
                values=values,
                proof_refs=(f"row_predicate:{predicate_id}",),
            )
        )
    return _RowPredicateParse(
        filters=tuple(filters),
        population_choices=tuple(population_choices),
    )


def _row_predicate_include_values(
    raw_review: Any,
    *,
    allowed_values: tuple[str, ...],
    tests_by_id: dict[str, Any],
    path: str,
) -> tuple[str, ...]:
    if not allowed_values:
        raise ValueError("row predicate requires allowed values")
    reviewed_effects = _reviewed_row_predicate_effects(
        raw_review,
        allowed_values=allowed_values,
        tests_by_id=tests_by_id,
        path=f"{path}.choice_reviews",
    )
    active_test_ids = _row_predicate_filter_test_ids(
        reviewed_effects,
        tests_by_id=tests_by_id,
        path=f"{path}.choice_reviews",
    )
    if not _has_decisive_row_predicate_effect(
        reviewed_effects,
        test_ids=active_test_ids,
    ):
        return allowed_values
    values = tuple(
        value
        for value, test_effects in reviewed_effects
        if _choice_is_included(
            test_effects=test_effects,
            tests_by_id=tests_by_id,
            test_ids=active_test_ids,
            choice_inclusion=None,
        )
    )
    if not values:
        raise ValueError("row predicate review must include at least one value")
    return values


def _row_predicate_filter_test_ids(
    reviewed_effects: list[tuple[str, dict[str, str]]],
    *,
    tests_by_id: dict[str, Any],
    path: str,
) -> tuple[str, ...]:
    output: list[str] = []
    for test_id in tests_by_id:
        effects = tuple(test_effects[test_id] for _, test_effects in reviewed_effects)
        unique_effects = set(effects)
        if unique_effects <= {"DOES_NOT_DECIDE_TEST", "UNKNOWN_TEST_EFFECT"}:
            continue
        if all(
            _population_tests_allow_choice(
                test_effects={test_id: effect},
                tests_by_id=tests_by_id,
                test_ids=(test_id,),
            )
            for effect in effects
        ):
            continue
        output.append(test_id)
    return tuple(output)


def _has_decisive_row_predicate_effect(
    reviewed_effects: list[tuple[str, dict[str, str]]],
    *,
    test_ids: tuple[str, ...],
) -> bool:
    return any(
        test_effects[test_id] in {"SATISFIES_TEST", "CONFLICTS_WITH_TEST"}
        for _, test_effects in reviewed_effects
        for test_id in test_ids
    )


def _reviewed_row_predicate_effects(
    raw_review: Any,
    *,
    allowed_values: tuple[str, ...],
    tests_by_id: dict[str, Any],
    path: str,
) -> list[tuple[str, dict[str, str]]]:
    review = provider_output.RowPredicateReviewOutput.parse(raw_review)
    raw_choices = _required_dicts(review.choice_reviews, path)
    seen: set[str] = set()
    output: list[tuple[str, dict[str, str]]] = []
    for raw_value in raw_choices:
        raw = provider_output.RowPredicateChoiceReviewOutput.parse(raw_value)
        value = _text(raw.choice_option_id)
        if value not in allowed_values:
            raise ValueError("row predicate review references unknown value")
        if value in seen:
            raise ValueError("duplicate row predicate value review")
        seen.add(value)
        if not _text(raw.choice_domain_meaning).strip():
            raise ValueError("row predicate value review requires domain meaning")
        output.append(
            (
                value,
                _row_predicate_population_test_effects(
                    raw.population_test_results,
                    tests_by_id=tests_by_id,
                    path=f"{path}.{value}.population_test_results",
                ),
            )
        )
    if seen != set(allowed_values):
        raise ValueError("row predicate reviews must cover every value")
    return output


def _row_predicate_population_test_effects(
    raw_results: Any,
    *,
    tests_by_id: dict[str, Any],
    path: str,
) -> dict[str, str]:
    results = _dict(raw_results, path)
    if set(results) != set(tests_by_id):
        raise ValueError("row predicate reviews must answer membership tests")
    effects: dict[str, str] = {}
    for test_id, test in tests_by_id.items():
        raw = provider_output.RowPredicatePopulationTestResultOutput.parse(results.get(test_id))
        if _text(raw.test_id) != test_id:
            raise ValueError("row predicate population test id must match result key")
        if not _text(raw.test_question).strip():
            raise ValueError("row predicate population test requires question")
        if not _text(raw.role_scoped_test_question).strip():
            raise ValueError(
                "row predicate population test requires role-scoped question"
            )
        if not _text(raw.because).strip():
            raise ValueError("row predicate population test requires reason")
        effect = _text(raw.test_effect)
        if effect not in {
            "SATISFIES_TEST",
            "CONFLICTS_WITH_TEST",
            "DOES_NOT_DECIDE_TEST",
            "UNKNOWN_TEST_EFFECT",
        }:
            raise ValueError("unsupported row predicate population test effect")
        effects[test_id] = effect
    return effects

def _population_choice_set(
    raw: dict[str, Any],
    *,
    param: dict[str, Any],
) -> _PopulationChoiceSet:
    if "param_decision_id" in raw:
        raise ValueError("choice params require population choice set")
    choice_set = _dict(raw.get("population_choice_set"), "population_choice_set")
    include_values = tuple(
        canonical_param_value(value) for value in choice_set.get("include_values") or ()
    )
    exclude_values = tuple(
        canonical_param_value(value) for value in choice_set.get("exclude_values") or ()
    )
    if not include_values:
        raise ValueError("population choice set requires included values")
    choices = {canonical_param_value(choice) for choice in param.get("choices") or ()}
    include_set = set(include_values)
    exclude_set = set(exclude_values)
    if include_set & exclude_set:
        raise ValueError("population choice set cannot overlap")
    if include_set | exclude_set != choices:
        raise ValueError("population choice set must cover every choice")
    if any(value not in choices for value in include_values):
        raise ValueError("population choice set includes unknown choice")
    return _PopulationChoiceSet(
        included_values=include_values,
        excluded_values=exclude_values,
    )


def _param_binding_sets(
    *,
    param_id: str,
    value: object,
    param: dict[str, Any],
    proof_refs: tuple[str, ...] = (),
) -> tuple[tuple[EndpointParamBinding, ...], ...]:
    if isinstance(value, tuple) and not _param_accepts_collection(param):
        return tuple(
            (
                EndpointParamBinding(
                    param_id=param_id,
                    value=item,
                    proof_refs=proof_refs,
                ),
            )
            for item in value
        )
    return (
        (
            EndpointParamBinding(
                param_id=param_id,
                value=value,
                proof_refs=proof_refs,
            ),
        ),
    )


def _param_accepts_collection(param: dict[str, Any]) -> bool:
    return str(param.get("type") or "").strip() in {"array", "list"}


def _normalized_param_decisions(
    raw_decisions: Any,
    *,
    parse_provider_output: bool = False,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if isinstance(raw_decisions, dict):
        for raw_param_id, raw_value in raw_decisions.items():
            param_id = str(raw_param_id)
            if param_id in output:
                raise ValueError("duplicate source param decision")
            output[param_id] = (
                vars(provider_output.ParamDecisionOutput.parse(raw_value))
                if parse_provider_output
                else _dict(raw_value, f"param_decisions.{param_id}")
            )
        return output
    raise ValueError("param_decisions must be an object")


def _param_decision_options_by_id(
    params_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for param_id, param in params_by_id.items():
        for option in param.get("decision_options") or ():
            if not isinstance(option, dict):
                continue
            decision_id = str(option.get("param_decision_id") or "")
            if not decision_id:
                continue
            output[decision_id] = {**option, "param_id": param_id}
    return output


def _validate_param_population_intent(
    raw: dict[str, Any],
) -> str:
    if "population_intent" not in raw:
        raise ValueError("source param decision requires population intent")
    population_intent = _optional_text(raw.get("population_intent"))
    if not population_intent:
        raise ValueError("source param decision requires non-empty population intent")
    return population_intent


def _param_requires_explicit_decision(param: dict[str, Any]) -> bool:
    return bool(param.get("required")) or bool(param.get("choices"))


def _param_is_model_bindable(param: dict[str, Any]) -> bool:
    decision_options = param.get("decision_options")
    return isinstance(decision_options, list) and bool(decision_options)


def _resolved_binding_value(
    value: str,
    *,
    param: dict[str, Any],
    option: dict[str, Any],
    available_values: tuple[FactValue, ...],
) -> tuple[object, tuple[str, ...]]:
    if str(param.get("type") or "") == "boolean" and value in {"true", "false"}:
        return value == "true", ()
    values_by_id = {item.id: item for item in available_values}
    fact_value = values_by_id.get(value)
    if fact_value is None:
        return value, ()
    component = _value_component_from_option(option)
    resolved = value_component(fact_value, component)
    if fact_value.kind == ValueKind.IDENTITY_SET:
        return resolved, tuple(fact_value.proof_refs)
    return str(resolved), tuple(fact_value.proof_refs)


def _value_component_from_option(
    option: dict[str, Any],
) -> ValueComponent | TimeComponent:
    raw_component = str(option.get("value_component") or "").strip()
    if raw_component == TimeComponent.START.value:
        return TimeComponent.START
    if raw_component == TimeComponent.END.value:
        return TimeComponent.END
    if raw_component == TimeComponent.INSTANT.value:
        return TimeComponent.INSTANT
    return ValueComponent.VALUE


def _merged_param_bindings(
    applied: tuple[EndpointParamBinding, ...],
    model_authored: tuple[EndpointParamBinding, ...],
) -> tuple[EndpointParamBinding, ...]:
    output: list[EndpointParamBinding] = []
    seen: set[str] = set()
    for binding in (*applied, *model_authored):
        if binding.param_id in seen:
            raise ValueError("duplicate source param binding")
        seen.add(binding.param_id)
        output.append(binding)
    return tuple(output)
