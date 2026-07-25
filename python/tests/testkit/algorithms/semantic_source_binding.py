from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.subject_obligations import (
    derive_subject_choice_membership,
    explicit_override_from_application_owners,
)
from fervis.lookup.source_binding.param_binding_sets import (
    finite_choice_parameter_is_omittable,
)

from tests.testkit.assertions import subset_mismatches


def run_semantic_source_binding_case(payload: dict[str, Any]) -> list[str]:
    request = payload["input"]
    mode = str(request["mode"])
    if mode == "finite_choice_parameter_omission":
        included_values = tuple(str(value) for value in request["included_values"])
        return subset_mismatches(
            actual={
                "parameter_omitted": finite_choice_parameter_is_omittable(
                    required=bool(request["required"]),
                    default=(
                        str(request["default"])
                        if request.get("default") is not None
                        else None
                    ),
                    choices=tuple(str(value) for value in request["choices"]),
                    included_values=included_values,
                ),
                "included_values": list(included_values),
            },
            expected_subset=payload["expect"]["result_contains"],
        )
    if mode == "explicit_override_from_application_owners":
        override_applies = explicit_override_from_application_owners(
            application_owner_refs=tuple(request["application_owner_refs"]),
            boolean_requirement_refs=frozenset(
                str(value) for value in request["boolean_requirement_refs"]
            ),
            matched_excluded_role=(
                str(request["matched_excluded_role"])
                if request.get("matched_excluded_role") is not None
                else None
            ),
        )
        return subset_mismatches(
            actual={"explicit_user_override_applies": override_applies},
            expected_subset=payload["expect"]["result_contains"],
        )
    if mode != "subject_choice_membership":
        return [f"unsupported semantic Source Binding mode: {mode}"]
    membership = derive_subject_choice_membership(
        choice_included=bool(request["choice_included"]),
        matched_excluded_role=(
            str(request["matched_excluded_role"])
            if request.get("matched_excluded_role") is not None
            else None
        ),
        explicit_user_override_applies=bool(request["explicit_user_override_applies"]),
    )
    return subset_mismatches(
        actual={
            "included": membership.included,
            "explicit_user_override_applies": (
                membership.explicit_user_override_applies
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )
