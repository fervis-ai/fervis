"""Normal-instance role profiles for source binding."""

from __future__ import annotations

from typing import Any

from fervis.lookup.question_contract import AnswerPopulationMembershipTestKind
from fervis.lookup.source_binding.membership_tests import membership_test_key
from fervis.lookup.source_binding.param_values import canonical_param_value
from fervis.lookup.source_binding.candidates.candidate_tree import (
    CandidateTreeContext,
    map_source_candidate_tree,
)


NORMAL_INSTANCE_ROLE_PROFILES_KEY = "normal_instance_role_profiles"
NORMAL_INSTANCE_NO_EXCLUDED_ROLE = "NONE"
NORMAL_INSTANCE_UNKNOWN_EXCLUDED_ROLE = "UNKNOWN"


def with_normal_instance_role_profiles(
    payload: dict[str, Any],
    *,
    request: Any,
) -> dict[str, Any]:
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _candidate_with_role_profiles_for_tree(
            candidate,
            context=context,
            facts_by_id=facts_by_id,
        ),
        top_level_keys=(),
    )


def _candidate_with_role_profiles_for_tree(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
    facts_by_id: dict[str, Any],
) -> dict[str, Any]:
    fact = facts_by_id.get(context.requested_fact_id)
    if fact is None:
        return candidate
    return _candidate_with_role_profiles(candidate, fact=fact)


def _candidate_with_role_profiles(
    candidate: dict[str, Any],
    *,
    fact: Any,
) -> dict[str, Any]:
    params = candidate.get("params")
    if not isinstance(params, list):
        return candidate
    if not str(candidate.get("source_candidate_id") or ""):
        return candidate
    output = dict(candidate)
    output["params"] = [
        _param_with_role_profiles(
            param,
            fact=fact,
        )
        for param in params
        if isinstance(param, dict)
    ]
    return output


def _param_with_role_profiles(
    param: dict[str, Any],
    *,
    fact: Any,
) -> dict[str, Any]:
    choices = tuple(
        canonical_param_value(choice) for choice in param.get("choices") or ()
    )
    if not choices or not isinstance(param.get("population_contract"), dict):
        return param
    tests = tuple(
        test
        for test in getattr(
            getattr(fact, "answer_population", None), "membership_tests", ()
        )
        if test.kind == AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD
        and test.normal_instance_profile is not None
    )
    if not tests:
        return param
    output = dict(param)
    output[NORMAL_INSTANCE_ROLE_PROFILES_KEY] = [
        _role_profile_for_test(test) for test in tests
    ]
    return output


def _role_profile_for_test(test: Any) -> dict[str, object]:
    return {
        "test_id": membership_test_key(test),
        "subject_text": test.normal_instance_profile.subject_text,
        "excluded_state_roles": [
            {
                "role": role.role.value,
                "role_definition": role.definition,
            }
            for role in test.normal_instance_profile.excluded_state_roles
        ],
        "match_policy": (
            "For each choice, select any clearly supported excluded role. Use NONE "
            "only when no excluded role applies. Use UNKNOWN only when the prompt "
            "data is insufficient to classify the choice against these roles."
        ),
    }
