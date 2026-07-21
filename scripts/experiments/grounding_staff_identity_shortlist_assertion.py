"""Validate Grounding's primary-key Staff resolver shortlist."""

from __future__ import annotations

from typing import Any


DETAIL_SUFFIX = "rs_00d41c447aa89efa_staff_primary_key"


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    reviews = arguments.get("known_input_binding_reviews", {})
    errors: list[str] = []
    for input_id in ("q1", "q2"):
        review = reviews.get(input_id, {})
        compatibility = review.get("resource_type_compatibility", {})
        if compatibility.get("staff") != "SAME_RESOURCE_TYPE":
            errors.append(f"{input_id}: Staff is not compatible")
        if any(
            decision != "DIFFERENT_RESOURCE_TYPE"
            for resource_type, decision in compatibility.items()
            if resource_type != "staff"
        ):
            errors.append(f"{input_id}: a non-Staff resource is compatible")
        if review.get("identifier_kind") != "PRIMARY_KEY":
            errors.append(f"{input_id}: identifier is not PRIMARY_KEY")
        positives = {
            option_id
            for option_id, option in review.get("option_reviews", {}).items()
            if option.get("resolution", {}).get("decision")
            == "CAN_RESOLVE_LOOKUP_TEXT"
        }
        expected = {
            option_id
            for option_id in review.get("option_reviews", {})
            if option_id.endswith(DETAIL_SUFFIX)
        }
        if positives != expected:
            errors.append(
                f"{input_id}: positive routes are {sorted(positives)}, "
                f"expected {sorted(expected)}"
            )
    return errors
