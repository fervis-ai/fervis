from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ProviderQuestionInputOwnership:
    question_input_uses: tuple[dict[str, str], ...]
    question_input_use_refs_by_test_id: Mapping[str, tuple[str, ...]]


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
        emitted_test_ids.add(test_id)
        test.pop("owned_question_input_refs", None)
        test["question_input_use_refs"] = list(
            ownership.question_input_use_refs_by_test_id.get(test_id, ())
        )
        output.append(test)
    for test_id in sorted(
        set(ownership.question_input_use_refs_by_test_id) - emitted_test_ids
    ):
        output.append(
            {
                "test_id": test_id,
                "kind": "EXPLICIT_USER_CONSTRAINT",
                "polarity": "MUST_PASS",
                "test_question": "Does this candidate satisfy the supplied input?",
                "question_input_use_refs": list(
                    ownership.question_input_use_refs_by_test_id[test_id]
                ),
            }
        )
    return output


def provider_question_input_ownership(
    *,
    group_key_input_refs: Iterable[str] = (),
    population_input_refs_by_test_id: Mapping[str, Iterable[str]] | None = None,
    result_limit_input_ref: str = "",
) -> ProviderQuestionInputOwnership:
    """Project test-owned semantic edges into the provider ownership contract."""

    uses: list[dict[str, str]] = []
    use_id_by_input_ref: dict[str, str] = {}
    owner_by_input_ref: dict[str, str] = {}

    def add(input_ref: str, owner_kind: str) -> str:
        normalized_ref = str(input_ref).strip()
        if not normalized_ref:
            raise ValueError("question input reference is required")
        if normalized_ref in use_id_by_input_ref:
            raise ValueError(f"question input has multiple owners: {normalized_ref}")
        use_id = f"use_{len(uses) + 1}"
        use = {
            "input_ref": normalized_ref,
            "owner_kind": owner_kind,
        }
        if owner_kind == "POPULATION_TESTS":
            use["use_id"] = use_id
        uses.append(use)
        use_id_by_input_ref[normalized_ref] = use_id
        owner_by_input_ref[normalized_ref] = owner_kind
        return use_id

    for input_ref in group_key_input_refs:
        add(input_ref, "GROUP_KEY")

    refs_by_test_id: dict[str, tuple[str, ...]] = {}
    for test_id, input_refs in (population_input_refs_by_test_id or {}).items():
        normalized_test_id = str(test_id).strip()
        if not normalized_test_id:
            raise ValueError("membership test ID is required")
        test_use_ids: list[str] = []
        for input_ref in input_refs:
            normalized_ref = str(input_ref).strip()
            use_id = use_id_by_input_ref.get(normalized_ref)
            if use_id is None:
                use_id = add(normalized_ref, "POPULATION_TESTS")
            elif owner_by_input_ref[normalized_ref] != "POPULATION_TESTS":
                raise ValueError(
                    f"question input has multiple owners: {normalized_ref}"
                )
            test_use_ids.append(use_id)
        refs_by_test_id[normalized_test_id] = tuple(test_use_ids)

    if result_limit_input_ref:
        add(result_limit_input_ref, "RESULT_LIMIT")

    return ProviderQuestionInputOwnership(
        question_input_uses=tuple(uses),
        question_input_use_refs_by_test_id=refs_by_test_id,
    )
