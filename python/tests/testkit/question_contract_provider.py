from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ProviderQuestionInputOwnership:
    question_input_uses: tuple[dict[str, object], ...]
    population_use_refs_by_test_id: Mapping[str, tuple[str, ...]]


def provider_membership_tests(
    membership_tests: Iterable[Mapping[str, Any]],
    *,
    ownership: ProviderQuestionInputOwnership,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    emitted_test_ids: set[str] = set()
    for raw_test in membership_tests:
        test = dict(raw_test)
        test_id = str(test["test_id"])
        if test.get("kind") != "EXPLICIT_USER_CONSTRAINT":
            continue
        emitted_test_ids.add(test_id)
        test.pop("test_id")
        test.pop("kind")
        test.pop("owned_question_input_refs", None)
        test.pop("question_input_refs", None)
        test.pop("question_input_use_refs", None)
        test["population_use_refs"] = list(
            ownership.population_use_refs_by_test_id.get(test_id, ())
        )
        output.append(test)
    for test_id in sorted(
        set(ownership.population_use_refs_by_test_id) - emitted_test_ids
    ):
        output.append(
            {
                "polarity": "MUST_PASS",
                "test_question": "Does this candidate satisfy the supplied input?",
                "population_use_refs": list(
                    ownership.population_use_refs_by_test_id[test_id]
                ),
            }
        )
    return output


def provider_answer_population(
    population: Mapping[str, Any],
    *,
    ownership: ProviderQuestionInputOwnership,
) -> dict[str, Any]:
    return {
        "membership_tests": provider_membership_tests(
            population.get("membership_tests") or (),
            ownership=ownership,
        )
    }


def provider_question_input_ownership(
    *,
    group_key_input_refs: Iterable[str] = (),
    compute_input_refs: Iterable[str] = (),
    population_input_refs_by_test_id: Mapping[str, Iterable[str]] | None = None,
    result_limit_input_ref: str = "",
) -> ProviderQuestionInputOwnership:
    """Project test-owned semantic edges into the provider ownership contract."""

    uses: list[dict[str, object]] = []
    owner_by_input_ref: dict[str, str] = {}

    def add(input_ref: str, owner_kind: str) -> None:
        normalized_ref = str(input_ref).strip()
        if not normalized_ref:
            raise ValueError("question input reference is required")
        if normalized_ref in owner_by_input_ref:
            raise ValueError(f"question input has multiple owners: {normalized_ref}")
        uses.append(
            {
                "input_ref": normalized_ref,
                "owner_kind": owner_kind,
            }
        )
        owner_by_input_ref[normalized_ref] = owner_kind

    for input_ref in group_key_input_refs:
        add(input_ref, "GROUP_KEY")

    for input_ref in compute_input_refs:
        add(input_ref, "COMPUTE_EXPRESSION")

    tests_by_input_ref: dict[str, list[str]] = {}
    for test_id, input_refs in (population_input_refs_by_test_id or {}).items():
        normalized_test_id = str(test_id).strip()
        if not normalized_test_id:
            raise ValueError("membership test ID is required")
        for input_ref in input_refs:
            normalized_ref = str(input_ref).strip()
            owner_kind = owner_by_input_ref.get(normalized_ref)
            if owner_kind not in {None, "POPULATION_TESTS"}:
                raise ValueError(
                    f"question input has multiple owners: {normalized_ref}"
                )
            tests_by_input_ref.setdefault(normalized_ref, []).append(normalized_test_id)

    refs_by_test_id: dict[str, list[str]] = {
        str(test_id): [] for test_id in (population_input_refs_by_test_id or {})
    }
    for index, (input_ref, test_ids) in enumerate(tests_by_input_ref.items(), start=1):
        add(input_ref, "POPULATION_TESTS")
        use_id = f"population_use_{index}"
        uses[-1]["use_id"] = use_id
        for test_id in test_ids:
            refs_by_test_id[test_id].append(use_id)

    if result_limit_input_ref:
        add(result_limit_input_ref, "RESULT_LIMIT")

    return ProviderQuestionInputOwnership(
        question_input_uses=tuple(uses),
        population_use_refs_by_test_id={
            test_id: tuple(refs) for test_id, refs in refs_by_test_id.items()
        },
    )
