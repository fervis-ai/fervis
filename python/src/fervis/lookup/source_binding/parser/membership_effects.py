"""Membership-test effect parsing for source binding."""

from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.relations import (
    RelationSourceReviewScopeDecision,
    ReviewScopeDecisionKind as RelationReviewScopeDecisionKind,
)
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    NormalInstanceExplicitOverrideReason,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.membership_tests import (
    membership_test_key,
    membership_tests_by_key,
)
from fervis.lookup.source_binding.normal_instance_roles import (
    NORMAL_INSTANCE_NO_EXCLUDED_ROLE,
    NORMAL_INSTANCE_UNKNOWN_EXCLUDED_ROLE,
)
from fervis.lookup.source_binding.param_values import canonical_param_value
from fervis.lookup.source_binding.parser_common import _text
from fervis.lookup.source_binding.review_scope import ReviewScopeDecision
from fervis.lookup.source_binding.review_surface import FiniteChoiceReviewAxis
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.provider_contract import ProviderObject


__all__ = [
    "answer_population_tests_by_id",
    "choice_is_included",
    "population_choice_proof_refs",
    "population_tests_allow_choice",
    "relation_review_scope_decisions",
    "review_finite_choice_sets",
]


def answer_population_tests_by_id(
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


def population_choice_proof_refs(
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


def relation_review_scope_decisions(
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


def review_finite_choice_sets(
    review: provider_output.FiniteChoiceParamReviewOutput,
    *,
    axis: FiniteChoiceReviewAxis,
    tests_by_id: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    param_id = axis.axis_id
    _validate_population_test_basis(
        review.population_test_basis,
        tests_by_id=tests_by_id,
        path=f"finite_choice_param_reviews.{param_id}.population_test_basis",
    )
    reviews = review.choice_reviews
    choices = axis.choices
    seen: set[str] = set()
    reviewed_effects: list[tuple[str, dict[str, str]]] = []
    choice_inclusions: dict[str, str] = {}
    for raw in reviews:
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
        if choice_is_included(
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
    satisfied_test_ids = satisfied_membership_test_ids(
        reviewed_effects,
        included_values=tuple(include_values),
        tests_by_id=tests_by_id,
    )
    return tuple(include_values), tuple(exclude_values), satisfied_test_ids


def satisfied_membership_test_ids(
    reviewed_effects: list[tuple[str, dict[str, str]]],
    *,
    included_values: tuple[str, ...],
    tests_by_id: dict[str, Any],
) -> tuple[str, ...]:
    included = set(included_values)
    satisfied: list[str] = []
    for test_id, test in tests_by_id.items():
        required_effect = (
            "SATISFIES_TEST"
            if test.polarity == AnswerPopulationMembershipTestPolarity.MUST_PASS
            else "CONFLICTS_WITH_TEST"
        )
        included_effects = tuple(
            effects[test_id] for value, effects in reviewed_effects if value in included
        )
        if included_effects and all(
            effect == required_effect for effect in included_effects
        ):
            satisfied.append(test_id)
    return tuple(satisfied)


def _validate_population_test_basis(
    basis: dict[str, provider_output.PopulationTestBasisOutput],
    *,
    tests_by_id: dict[str, Any],
    path: str,
) -> None:
    if set(basis) != set(tests_by_id):
        raise ValueError(
            "finite choice population test basis must cover membership tests"
        )
    for test_id in tests_by_id:
        item = basis[test_id]
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
    results: dict[str, ProviderObject],
    *,
    axis: FiniteChoiceReviewAxis,
    tests_by_id: dict[str, Any],
    path: str,
) -> dict[str, str]:
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
        test = tests_by_id[test_id]
        if test.kind == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD:
            normal_result = provider_output.NormalInstanceTestResultOutput.parse(
                raw_value
            )
            effect = _validate_normal_instance_test_effect(
                normal_result,
                test=test,
                axis=axis,
                path=f"{path}.{test_id}",
            )
        else:
            standard_result = provider_output.StandardPopulationTestResultOutput.parse(
                raw_value
            )
            effect = _standard_population_test_effect(standard_result)
        effects[test_id] = effect
    return effects


def _standard_population_test_effect(
    result: provider_output.StandardPopulationTestResultOutput,
) -> str:
    if not _text(result.test_basis).strip():
        raise ValueError("finite choice population test requires basis")
    if not _text(result.population_consequence).strip():
        raise ValueError(
            "finite choice population test requires population consequence"
        )
    effect = _text(result.test_effect)
    if effect not in {
        "SATISFIES_TEST",
        "CONFLICTS_WITH_TEST",
        "DOES_NOT_DECIDE_TEST",
        "UNKNOWN_TEST_EFFECT",
    }:
        raise ValueError("unsupported finite choice population test effect")
    return effect


def _validate_normal_instance_test_effect(
    raw: provider_output.NormalInstanceTestResultOutput,
    *,
    test: Any,
    axis: FiniteChoiceReviewAxis,
    path: str,
) -> str:
    if test.polarity != AnswerPopulationMembershipTestPolarity.MUST_PASS:
        raise ValueError("normal instance review requires must-pass guard")
    profile = getattr(test, "normal_instance_profile", None)
    if profile is None:
        raise ValueError("normal instance review requires normal instance profile")
    disposition = raw.disposition
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
    items: tuple[provider_output.NormalInstanceOverrideEvidenceOutput, ...],
    *,
    path: str,
) -> tuple[dict[str, str], ...]:
    output: list[dict[str, str]] = []
    for evidence in items:
        source_text = _text(evidence.source_text)
        if not source_text.strip():
            raise ValueError(
                "normal instance review override evidence requires source text"
            )
        reason = NormalInstanceExplicitOverrideReason(_text(evidence.reason))
        output.append({"source_text": source_text, "reason": reason.value})
    return tuple(output)


def choice_is_included(
    *,
    test_effects: dict[str, str],
    tests_by_id: dict[str, Any],
    test_ids: tuple[str, ...],
    choice_inclusion: str | None,
) -> bool:
    if choice_inclusion == "EXCLUDE":
        return False
    tests_decide = _population_tests_decide(
        test_effects=test_effects,
        test_ids=test_ids,
    )
    if not tests_decide:
        return True
    return population_tests_allow_choice(
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


def population_tests_allow_choice(
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
