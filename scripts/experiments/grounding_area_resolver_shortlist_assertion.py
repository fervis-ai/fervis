"""Validate Grounding's Area/Location resolver shortlist for Nairobi."""

from __future__ import annotations

from typing import Any


AREA_LIST_OPTION = "bind_q1_input_1_rs_7de03e00183c6709_area_primary_key"
LOCATION_LIST_OPTION = "bind_q1_input_1_rs_7d752a2759a3cfdc_location_primary_key"


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    reviews = arguments.get("known_input_binding_reviews", {})
    review = reviews.get("q1_input_1", {})
    options = review.get("option_reviews", {})
    errors: list[str] = []
    for option_id, label in (
        (AREA_LIST_OPTION, "Area list"),
        (LOCATION_LIST_OPTION, "Location list"),
    ):
        resolution = options.get(option_id, {}).get("resolution", {})
        if resolution.get("decision") != "CAN_RESOLVE_LOOKUP_TEXT":
            errors.append(f"{label} is not shortlisted")
    return errors
