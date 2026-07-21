from __future__ import annotations

from typing import Any, Iterator


_EXCLUDED_ROLES = {
    "NOT_REALIZED",
    "CANCELED_OR_VOIDED",
    "FAILED_OR_REJECTED_BEFORE_EFFECT",
    "REVERSED_OR_CORRECTION_ARTIFACT",
    "TEST_PLACEHOLDER_OR_DEMO",
    "SUPERSEDED_DELETED_OR_NON_CURRENT_ARTIFACT",
}


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = str(context.get("label") or "")
    baseline = label.startswith("baseline:")
    case_label = label.removeprefix("baseline:")
    if baseline:
        for result in _normal_instance_results(arguments):
            if "disposition" not in result:
                continue
            if result.get("explicit_user_override_applies") is not False:
                errors.append("ordinary baseline applied a free-form override")
            if result.get("explicit_user_override_evidence") != []:
                errors.append("ordinary baseline emitted free-form override evidence")
    elif _contains_override_fields(arguments):
        errors.append("output still contains free-form override fields")

    normal_results = list(_normal_instance_results(arguments))
    if not normal_results:
        errors.append("output contains no normal-instance assessment")
    for result in normal_results:
        disposition = result.get("disposition")
        if disposition is None and result.get("test_effect") is not None:
            continue
        if not isinstance(disposition, dict):
            errors.append("normal-instance choice assessment lacks effect")
            continue
        role = disposition.get("matched_excluded_role")
        effect = disposition.get("test_effect")
        if role in _EXCLUDED_ROLES and effect != "CONFLICTS_WITH_TEST":
            errors.append(f"excluded role {role} does not conflict with the guard")
        if role == "UNKNOWN" and effect != "UNKNOWN_TEST_EFFECT":
            errors.append("unknown role does not have unknown effect")

    choices = {
        str(choice.get("choice_option_id") or ""): _derived_choice_inclusion(choice)
        for choice in _choice_reviews(arguments)
    }
    if case_label == "sales_store_count_this_month":
        if choices.get("IN_PERSON") != "INCLUDE":
            errors.append("in-person choice is not included")
        if choices.get("ONLINE") != "EXCLUDE":
            errors.append("online choice is not excluded")
    elif case_label == "no_data_low_completed_sales_future_date":
        if "COMPLETED" in choices and choices["COMPLETED"] != "INCLUDE":
            errors.append("completed choice is not included")
    elif case_label == "sales_store_top_this_month":
        pass
    elif case_label == "unqualified_sales_march":
        expected = {
            "DRAFT": "EXCLUDE",
            "PLACED": "EXCLUDE",
            "COMPLETED": "INCLUDE",
            "CANCELED": "EXCLUDE",
        }
        for choice, inclusion in expected.items():
            if choices.get(choice) != inclusion:
                errors.append(
                    f"{choice.casefold()} choice is not {inclusion.casefold()}d"
                )
    else:
        errors.append(f"unknown assertion label: {case_label}")
    return errors


def _contains_override_fields(value: object) -> bool:
    if isinstance(value, dict):
        if {
            "explicit_user_override_evidence",
            "explicit_user_override_applies",
        } & value.keys():
            return True
        return any(_contains_override_fields(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_override_fields(item) for item in value)
    return False


def _normal_instance_results(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        result = value.get("normal_instance_guard")
        if isinstance(result, dict) and (
            "disposition" in result or "test_effect" in result
        ):
            yield result
        for child in value.values():
            yield from _normal_instance_results(child)
    elif isinstance(value, list):
        for child in value:
            yield from _normal_instance_results(child)


def _choice_reviews(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        reviews = value.get("choice_reviews")
        if isinstance(reviews, list):
            yield from (review for review in reviews if isinstance(review, dict))
        for child in value.values():
            yield from _choice_reviews(child)
    elif isinstance(value, list):
        for child in value:
            yield from _choice_reviews(child)


def _derived_choice_inclusion(choice: dict[str, Any]) -> str:
    authored = str(choice.get("choice_inclusion") or "")
    if authored:
        return authored
    effects = tuple(_test_effects(choice.get("population_test_results")))
    if any(effect in {"CONFLICTS_WITH_TEST", "UNKNOWN_TEST_EFFECT"} for effect in effects):
        return "EXCLUDE"
    return "INCLUDE"


def _test_effects(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        effect = value.get("test_effect")
        if isinstance(effect, str):
            yield effect
        for child in value.values():
            yield from _test_effects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _test_effects(child)
