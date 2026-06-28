"""Model-facing keys for answer-population membership tests."""

from __future__ import annotations

from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    RequestedFactAnswerPopulationMembershipTest,
)


def membership_test_key(test: RequestedFactAnswerPopulationMembershipTest) -> str:
    if test.kind == AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT:
        return f"{test.kind.value.lower()}:{test.id}"
    if test.kind in {
        AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
        AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD,
        AnswerPopulationMembershipTestKind.RAW_RECORD_GUARD,
    }:
        return test.kind.value.lower()
    raise ValueError(f"unsupported membership test kind: {test.kind!r}")


def membership_tests_by_key(
    tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...],
) -> dict[str, RequestedFactAnswerPopulationMembershipTest]:
    output: dict[str, RequestedFactAnswerPopulationMembershipTest] = {}
    for test in tests:
        key = membership_test_key(test)
        if not key.strip() or key in output:
            raise ValueError("membership test keys must be unique")
        output[key] = test
    return output
