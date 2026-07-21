"""Validate Area identity application at the Source Binding boundary."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    errors: list[str] = []
    applications = _metric(arguments).get("resolved_input_applications", [])
    if len(applications) != 1:
        return [f"expected one resolved input application, got {len(applications)}"]

    target_ids = {
        item.get("application_target_id")
        for item in applications[0].get("applications", [])
        if isinstance(item, dict)
    }
    if "request_parameter.name" in target_ids:
        errors.append("Area identity is incorrectly applied to request_parameter.name")
    if not any(
        isinstance(target_id, str) and target_id.startswith("returned_identity.")
        for target_id in target_ids
    ):
        errors.append("Area identity is not applied to a returned identity")

    reviews = _metric(arguments).get("finite_choice_param_reviews", {})
    type_reviews = reviews.get("type", {}).get("choice_reviews", [])
    inclusions = {
        review.get("choice_option_id"): review.get("choice_inclusion")
        for review in type_reviews
        if isinstance(review, dict)
    }
    if inclusions.get("STORE") != "INCLUDE":
        errors.append("STORE is not included")
    if inclusions.get("WAREHOUSE") != "EXCLUDE":
        errors.append("WAREHOUSE is not excluded")
    return errors


def _metric(arguments: dict[str, Any]) -> dict[str, Any]:
    outcome = arguments.get("outcome", {})
    bindings = outcome.get("bindings_for_fact_1", {})
    metric = bindings.get("metric", {})
    return metric if isinstance(metric, dict) else {}
