"""One parser for bounded source-population test effects."""

from __future__ import annotations

from fervis.lookup.answer_program.relations import (
    PopulationCoverageClaim,
    PopulationCoverageRole,
)
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    MembershipTestRef,
    RequestedFactAnswerPopulationMembershipTest,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.membership_tests import membership_test_key


_TEST_EFFECTS = (
    "SATISFIES_TEST",
    "CONFLICTS_WITH_TEST",
    "DOES_NOT_DECIDE_TEST",
    "UNKNOWN_TEST_EFFECT",
)


def population_test_basis_payload(
    tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...],
    *,
    role_text: str,
) -> dict[str, dict[str, str]]:
    return {
        membership_test_key(test): {
            "test_question": test.test_question,
            "role_scoped_test_question": role_scoped_test_question(
                test.test_question,
                role_text=role_text,
            ),
        }
        for test in tests
    }


def population_test_results_schema(
    test_ids: tuple[str, ...],
) -> dict[str, object]:
    properties = {
        test_id: provider_output.RowPredicatePopulationTestResultOutput.schema(
            {
                "test_id": {"enum": [test_id]},
                "test_question": {"type": "string", "minLength": 1},
                "role_scoped_test_question": {
                    "type": "string",
                    "minLength": 1,
                },
                "because": {"type": "string", "minLength": 1},
                "test_effect": {"enum": list(_TEST_EFFECTS)},
            }
        )
        for test_id in test_ids
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(test_ids),
    }


def population_coverage_claims(
    results: dict[str, provider_output.RowPredicatePopulationTestResultOutput],
    *,
    tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...],
    requested_fact_id: str,
    role_text: str,
    coverage_role: PopulationCoverageRole,
    proof_refs: tuple[str, ...],
) -> tuple[PopulationCoverageClaim, ...]:
    tests_by_key = {membership_test_key(test): test for test in tests}
    if set(results) != set(tests_by_key):
        raise ValueError("population test results must cover their scoped tests")
    claims: list[PopulationCoverageClaim] = []
    for test_id, test in tests_by_key.items():
        result = results[test_id]
        _validate_result(
            result,
            test_id=test_id,
            test=test,
            role_text=role_text,
        )
        if result.test_effect in {
            "DOES_NOT_DECIDE_TEST",
            "UNKNOWN_TEST_EFFECT",
        }:
            continue
        required_effect = (
            "SATISFIES_TEST"
            if test.polarity is AnswerPopulationMembershipTestPolarity.MUST_PASS
            else "CONFLICTS_WITH_TEST"
        )
        if result.test_effect != required_effect:
            raise ValueError("source population conflicts with its membership test")
        if not proof_refs:
            raise ValueError("population coverage claim requires source mechanics")
        claims.append(
            PopulationCoverageClaim(
                test_ref=MembershipTestRef(
                    requested_fact_id=requested_fact_id,
                    membership_test_id=test.id,
                ),
                role=coverage_role,
                proof_refs=proof_refs,
            )
        )
    return tuple(claims)


def population_coverage_claims_for_satisfied_tests(
    test_ids: tuple[str, ...],
    *,
    requested_fact_id: str,
    coverage_role: PopulationCoverageRole,
    proof_refs: tuple[str, ...],
) -> tuple[PopulationCoverageClaim, ...]:
    if test_ids and not proof_refs:
        raise ValueError("population coverage claim requires source mechanics")
    return tuple(
        PopulationCoverageClaim(
            test_ref=MembershipTestRef(
                requested_fact_id=requested_fact_id,
                membership_test_id=test_id,
            ),
            role=coverage_role,
            proof_refs=proof_refs,
        )
        for test_id in test_ids
    )


def canonical_coverage_test_ids(
    test_ids: tuple[str, ...],
    *,
    tests_by_id: dict[str, RequestedFactAnswerPopulationMembershipTest],
) -> tuple[str, ...]:
    return tuple(
        test.id
        for test_id in test_ids
        if (test := tests_by_id[test_id]).kind
        is not AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY
    )


def role_scoped_test_question(test_question: str, *, role_text: str) -> str:
    return f"For {role_text}, {test_question}"


def _validate_result(
    result: provider_output.RowPredicatePopulationTestResultOutput,
    *,
    test_id: str,
    test: RequestedFactAnswerPopulationMembershipTest,
    role_text: str,
) -> None:
    if result.test_id != test_id:
        raise ValueError("population test result id mismatches its key")
    if result.test_question != test.test_question:
        raise ValueError("population test result question mismatches its test")
    expected_role_question = role_scoped_test_question(
        test.test_question,
        role_text=role_text,
    )
    if result.role_scoped_test_question != expected_role_question:
        raise ValueError("population test result mismatches its source role")
    if not result.because.strip():
        raise ValueError("population test result requires a basis")
    if result.test_effect not in _TEST_EFFECTS:
        raise ValueError("population test result has unsupported effect")
