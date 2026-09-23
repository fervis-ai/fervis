"""Outcome assertions for a captured production Source Binding turn."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    if not any(
        context.get(key)
        for key in (
            "expected_choice_reviews",
            "expected_finite_choice_applications",
            "expected_resolved_applications",
        )
    ):
        return ["source binding assertion requires an expected binding outcome"]
    errors = []
    errors.extend(_choice_review_errors(arguments, context))
    errors.extend(_finite_application_errors(arguments, context))
    errors.extend(_resolved_application_errors(arguments, context))
    return errors


def _choice_review_errors(
    arguments: dict[str, Any], context: dict[str, Any]
) -> list[str]:
    reviews = _choice_reviews(arguments)
    errors = []
    for expected in context.get("expected_choice_reviews", ()):
        key = (str(expected["surface_ref"]), str(expected["choice"]))
        actual = reviews.get(key)
        if actual is None:
            errors.append(f"choice review is missing: {key[0]}={key[1]}")
            continue
        for field in ("selected_by_requirements",):
            wanted = expected.get(field)
            observed = actual.get(field)
            if isinstance(wanted, list) and isinstance(observed, list):
                wanted, observed = sorted(wanted), sorted(observed)
            if wanted is not None and observed != wanted:
                errors.append(f"{key[0]}={key[1]} {field} is {observed!r}; expected {wanted!r}")
    return errors


def _finite_application_errors(
    arguments: dict[str, Any], context: dict[str, Any]
) -> list[str]:
    applications = arguments.get("finite_choice_applications")
    if not isinstance(applications, dict):
        return ["finite_choice_applications is missing"]
    actual = {
        (str(branch_ref), str(owner_ref)): application
        for branch_ref, branch in applications.items()
        if isinstance(branch, dict)
        for owner_ref, application in branch.items()
        if isinstance(application, dict)
    }
    errors = []
    for expected in context.get("expected_finite_choice_applications", ()):
        key = (str(expected["branch_ref"]), str(expected["owner_ref"]))
        if expected.get("is_null"):
            branch = applications.get(key[0], {})
            if key[1] not in branch or branch[key[1]] is not None:
                errors.append(f"finite choice must be null: {key}")
            continue
        application = actual.get(key)
        if application is None:
            errors.append(f"finite-choice application is missing: {key[0]}:{key[1]}")
            continue
        expected_surface = expected.get("surface_ref")
        if (
            expected_surface is not None
            and application.get("surface_ref") != expected_surface
        ):
            errors.append(
                f"{key[0]}:{key[1]} surface is "
                f"{application.get('surface_ref')!r}; expected {expected_surface!r}"
            )
        expected_choices = set(expected.get("selected_choice_values") or ())
        actual_choices = set(application.get("selected_choice_values") or ())
        if expected_choices and actual_choices != expected_choices:
            errors.append(
                f"{key[0]}:{key[1]} choices are {sorted(actual_choices)!r}; "
                f"expected {sorted(expected_choices)!r}"
            )
    return errors


def _resolved_application_errors(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    applications = arguments.get("resolved_input_applications", {})
    actual = [value for values in applications.values() for value in values]
    return [
        f"resolved input application is missing: {expected!r}"
        for expected in context.get("expected_resolved_applications", ())
        if not any(all(value.get(key) == wanted for key, wanted in expected.items())
                   for value in actual)
    ]


def _choice_reviews(
    arguments: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(surface_ref), str(choice)): review
        for _, branch in _objects(arguments.get("choice_requirement_applications"))
        for surface_ref, surface in _objects(branch)
        for choice, review in _objects(surface)
    }


def _objects(value: object) -> tuple[tuple[str, dict[str, Any]], ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(
        (str(key), item) for key, item in value.items() if isinstance(item, dict)
    )


__all__ = ["validate"]
