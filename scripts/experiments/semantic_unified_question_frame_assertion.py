"""Validate the production unified question frame by semantic outcome."""

from __future__ import annotations

from typing import Any, NamedTuple

from fervis.lookup.question_contract import (
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_frame,
)


class SuppliedValue(NamedTuple):
    meaning: str
    operands: tuple[str, ...]
    denotation: str
    instance_kind: str | None
    value_type: str

    def represents(self, expected: str) -> bool:
        normalized = expected.casefold()
        return self.meaning.casefold() == normalized or any(
            operand.casefold() == normalized for operand in self.operands
        )


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    outcome = arguments.get("outcome")
    if not isinstance(outcome, dict):
        return ["question frame outcome is missing"]
    expected_outcome = str(context.get("expected_outcome") or "question_meaning")
    if outcome.get("kind") != expected_outcome:
        return [f"outcome is {outcome.get('kind')!r}; expected {expected_outcome!r}"]
    if expected_outcome != "question_meaning":
        return []

    question = context.get("question")
    if not isinstance(question, str) or not question:
        return ["assertion context requires the question"]
    try:
        parsed = parse_semantic_question_frame(
            arguments,
            question_context_texts=(question,),
        )
    except (TypeError, ValueError) as exc:
        return [f"production question-frame parser rejected output: {exc}"]
    if not isinstance(parsed, ParsedSemanticQuestionMeaning):
        return ["question frame unexpectedly requires clarification"]

    requests = outcome.get("answer_requests")
    supplied = outcome.get("supplied_values")
    if not isinstance(requests, list):
        return ["answer_requests is missing"]
    if not isinstance(supplied, list):
        return ["supplied_values is missing"]

    errors = _request_errors(requests, context=context)
    values = (*_supplied_values(supplied), *_selection_limit_values(requests))
    errors.extend(_supplied_value_errors(values, context=context))
    errors.extend(_semantic_coverage_errors(arguments, context=context))
    return errors


def _request_errors(
    requests: list[object],
    *,
    context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_count = context.get("expected_request_count")
    if isinstance(expected_count, int) and len(requests) != expected_count:
        errors.append(
            f"answer request count is {len(requests)}; expected {expected_count}"
        )

    expected_kinds = context.get("expected_result_kinds") or ()
    expected_grouping_counts = context.get("expected_grouping_counts") or ()
    expected_output_counts = context.get("expected_output_counts") or ()
    expected_output_terms = context.get("expected_output_contains_any") or ()
    expected_selections = context.get("expected_selections") or ()
    for index, raw_request in enumerate(requests):
        if not isinstance(raw_request, dict):
            errors.append(f"answer request {index + 1} is not an object")
            continue
        result_kind = raw_request.get("result_kind")
        if index < len(expected_kinds) and result_kind != expected_kinds[index]:
            errors.append(
                f"answer request {index + 1} result kind is {result_kind!r}; "
                f"expected {expected_kinds[index]!r}"
            )

        grouping = raw_request.get("grouping_meanings")
        grouping_count = len(grouping) if isinstance(grouping, list) else -1
        if (
            index < len(expected_grouping_counts)
            and grouping_count != expected_grouping_counts[index]
        ):
            errors.append(
                f"answer request {index + 1} grouping count is {grouping_count}; "
                f"expected {expected_grouping_counts[index]}"
            )

        outputs = _returned_meanings(raw_request)
        if (
            index < len(expected_output_counts)
            and len(outputs) != expected_output_counts[index]
        ):
            errors.append(
                f"answer request {index + 1} output count is {len(outputs)}; "
                f"expected {expected_output_counts[index]}"
            )
        if index < len(expected_output_terms):
            output_text = " ".join(outputs).casefold()
            alternatives = expected_output_terms[index]
            if not any(str(term).casefold() in output_text for term in alternatives):
                errors.append(
                    f"answer request {index + 1} outputs omit every expected "
                    f"term in {alternatives!r}"
                )

        selection = raw_request.get("selection")
        selection_kind = (
            selection.get("kind") if isinstance(selection, dict) else None
        )
        if (
            index < len(expected_selections)
            and selection_kind != expected_selections[index]
        ):
            errors.append(
                f"answer request {index + 1} selection is {selection_kind!r}; "
                f"expected {expected_selections[index]!r}"
            )
    return errors


def _returned_meanings(request: dict[str, Any]) -> tuple[str, ...]:
    values_by_ref = {
        item["value_ref"]: str(item["meaning"])
        for item in request.get("answer_values") or ()
        if isinstance(item, dict)
        and isinstance(item.get("value_ref"), str)
        and isinstance(item.get("meaning"), str)
    }
    returned = [
        values_by_ref[ref]
        for ref in request.get("returned_value_refs") or ()
        if ref in values_by_ref
    ]
    returned_result = request.get("returned_result")
    if (
        isinstance(returned_result, dict)
        and returned_result.get("kind") in {"identities", "identities_and_values"}
    ):
        if request.get("result_kind") == "grouped_results":
            returned[:0] = _meaning_texts(request.get("grouping_meanings"))
        elif request.get("result_kind") == "qualifying_instances":
            returned[:0] = _meaning_texts(
                [request.get("returned_candidate_identity")]
            )
    return tuple(returned)


def _supplied_values(values: list[object]) -> tuple[SuppliedValue, ...]:
    supplied: list[SuppliedValue] = []
    for raw_value in values:
        if not isinstance(raw_value, dict):
            continue
        denotation = raw_value.get("denotation")
        value = raw_value.get("value")
        value_type = value.get("value_type") if isinstance(value, dict) else None
        operands = value.get("operands") if isinstance(value, dict) else None
        if (
            isinstance(raw_value.get("meaning"), str)
            and isinstance(denotation, dict)
            and isinstance(denotation.get("kind"), str)
            and isinstance(value_type, dict)
            and isinstance(value_type.get("kind"), str)
            and isinstance(operands, list)
            and all(isinstance(item, str) for item in operands)
        ):
            supplied.append(
                SuppliedValue(
                    meaning=raw_value["meaning"],
                    operands=tuple(operands),
                    denotation=denotation["kind"],
                    instance_kind=(
                        str(denotation["instance_kind"])
                        if denotation.get("instance_kind") is not None
                        else None
                    ),
                    value_type=value_type["kind"],
                )
            )
    return tuple(supplied)


def _selection_limit_values(requests: list[object]) -> tuple[SuppliedValue, ...]:
    limits = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        selection = request.get("selection")
        if (
            isinstance(selection, dict)
            and selection.get("kind") == "take_with_boundary_ties"
            and isinstance(selection.get("limit"), dict)
        ):
            limits.append(selection["limit"])
    return _supplied_values(limits)


def _supplied_value_errors(
    supplied: tuple[SuppliedValue, ...],
    *,
    context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    inventories = context.get("accepted_supplied_value_inventories") or ()
    actual_inventory = tuple(
        operand for value in supplied for operand in value.operands
    )
    if inventories and not any(
        _same_inventory(actual_inventory, tuple(expected))
        for expected in inventories
    ):
        errors.append(
            f"supplied-value inventory is {actual_inventory!r}; "
            f"expected one of {inventories!r}"
        )

    required = context.get("required_supplied_values") or {}
    for expected_text, expected_shape in required.items():
        matches = [
            value for value in supplied if value.represents(str(expected_text))
        ]
        if len(matches) != 1:
            errors.append(
                f"supplied value {expected_text!r} occurs {len(matches)} times"
            )
            continue
        [actual] = matches
        if not isinstance(expected_shape, dict):
            continue
        expected_denotation = expected_shape.get("denotation")
        if expected_denotation and actual.denotation != expected_denotation:
            errors.append(
                f"supplied value {expected_text!r} denotation is "
                f"{actual.denotation!r}; expected {expected_denotation!r}"
            )
        expected_type = expected_shape.get("value_type")
        if expected_type and actual.value_type != expected_type:
            errors.append(
                f"supplied value {expected_text!r} type is "
                f"{actual.value_type!r}; expected {expected_type!r}"
            )
    return errors


def _semantic_coverage_errors(
    arguments: dict[str, Any],
    *,
    context: dict[str, Any],
) -> list[str]:
    text = " ".join(_strings(arguments)).casefold()
    return [
        f"semantic frame omits required term {term!r}"
        for term in context.get("required_semantic_terms") or ()
        if str(term).casefold() not in text
    ]


def _meaning_texts(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        str(item["meaning"])
        for item in values
        if isinstance(item, dict) and isinstance(item.get("meaning"), str)
    ]


def _same_inventory(actual: tuple[str, ...], expected: tuple[object, ...]) -> bool:
    return sorted(item.casefold() for item in actual) == sorted(
        str(item).casefold() for item in expected
    )


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            text for item in value.values() for text in _strings(item)
        )
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value for text in _strings(item))
    return ()
