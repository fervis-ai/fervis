"""Outcome assertions for a captured production Grounding turn."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    if "runtime_references" in context:
        return _runtime_reference_errors(arguments, context["runtime_references"])
    if "identity_task_ref" not in context or "expected_route_id" not in context:
        return ["grounding assertion requires an expected identity route"]
    task_ref = str(context["identity_task_ref"])
    expected_route_id = str(context["expected_route_id"])
    reviews = arguments.get("reference_reviews")
    if not isinstance(reviews, dict):
        return ["reference_reviews is missing"]
    task = reviews.get(task_ref)
    if not isinstance(task, dict):
        return [f"reference review is missing for {task_ref!r}"]
    resource_reviews = task.get("resource_type_reviews")
    if not isinstance(resource_reviews, dict):
        return ["resource_type_reviews is missing"]
    route_review = next(
        (
            route
            for resource in resource_reviews.values()
            if isinstance(resource, dict)
            for route_id, route in _objects(resource.get("route_reviews"))
            if route_id == expected_route_id
        ),
        None,
    )
    if route_review is None:
        return [f"resolver route review is missing for {expected_route_id!r}"]
    resolution = route_review.get("resolution")
    if not isinstance(resolution, dict):
        return ["resolver route resolution is missing"]
    errors = []
    expected_decision = str(
        context.get("expected_decision") or "CAN_RESOLVE_LOOKUP_TEXT"
    )
    if resolution.get("decision") != expected_decision:
        errors.append(
            f"resolver decision is {resolution.get('decision')!r}; "
            f"expected {expected_decision!r}"
        )
    expected_param = context.get("expected_lookup_param")
    if expected_param is not None and expected_param not in (
        resolution.get("lookup_request_params") or ()
    ):
        errors.append(f"lookup parameters omit {expected_param!r}")
    return errors


def _runtime_reference_errors(
    arguments: dict[str, Any], expected: dict[str, dict[str, Any]]
) -> list[str]:
    actual = arguments.get("references")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return ["Runtime reference tasks do not cover the expected scope"]
    errors = []
    for task_ref, outcome in expected.items():
        selected = actual.get(task_ref)
        if not isinstance(selected, dict):
            errors.append(f"{task_ref}: missing reference selection")
        elif outcome["kind"] == "literal":
            fields = selected.get("field_refs")
            accepted = outcome.get("accepted_field_ref_sets") or [outcome["field_refs"]]
            if (
                not isinstance(fields, list)
                or len(fields) != len(set(fields))
                or not any(set(fields) == set(option) for option in accepted)
            ):
                errors.append(f"{task_ref}: incorrect literal matching properties")
        elif outcome["kind"] == "descriptor":
            if (
                selected.get("field_ref") != outcome["field_ref"]
                or selected.get("choice_value") != outcome["choice_value"]
            ):
                errors.append(f"{task_ref}: incorrect descriptor property choice")
        else:
            errors.append(f"{task_ref}: unknown reference assertion kind")
    return errors


def _objects(value: object) -> tuple[tuple[str, dict[str, Any]], ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(
        (str(key), item) for key, item in value.items() if isinstance(item, dict)
    )


__all__ = ["validate"]
