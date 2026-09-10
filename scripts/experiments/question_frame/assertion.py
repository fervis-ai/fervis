"""Validate the production unified question frame by semantic outcome."""

from __future__ import annotations

from typing import Any, NamedTuple
import json
from collections import Counter

from fervis.lookup.question_contract import (
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_frame,
)
from fervis.lookup.semantic_types import IntegerType, input_operand_matches_value_type


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
            conversation_text_by_resolved_input_ref=dict(
                context.get("conversation_text_by_resolved_input_ref") or {}
            ),
        )
    except (TypeError, ValueError) as exc:
        return [f"production question-frame parser rejected output: {exc}"]
    if not isinstance(parsed, ParsedSemanticQuestionMeaning):
        return ["question frame unexpectedly requires clarification"]

    requests = outcome.get("answer_requests")
    supplied = outcome.get("supplied_values")
    if not isinstance(requests, list):
        return ["answer_requests is missing"]
    if not isinstance(supplied, dict):
        return ["supplied_values is missing"]

    errors = _request_errors(requests, context=context)
    errors.extend(_candidate_origin_errors(requests, context=context))
    values = _supplied_values(supplied)
    errors.extend(_supplied_value_errors(values, context=context))
    errors.extend(_entity_reference_group_errors(values, context=context))
    errors.extend(_semantic_coverage_errors(arguments, context=context))
    return errors


def _candidate_origin_errors(
    requests: list[object],
    *,
    context: dict[str, Any],
) -> list[str]:
    expected = context.get("expected_candidate_origin_kinds") or ()
    errors: list[str] = []
    for index, expected_kind in enumerate(expected):
        if index >= len(requests) or not isinstance(requests[index], dict):
            continue
        result = _result(requests[index])
        qualifying_rows = _row_source(result)
        origin = (
            qualifying_rows.get("origin") if isinstance(qualifying_rows, dict) else None
        )
        actual_kind = origin.get("kind") if isinstance(origin, dict) else None
        if actual_kind != expected_kind:
            errors.append(
                f"answer request {index + 1} candidate origin is "
                f"{actual_kind!r}; expected {expected_kind!r}"
            )
    return errors


def _request_errors(
    requests: list[object],
    *,
    context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_count = context.get("expected_request_count")
    accepted_counts = context.get("accepted_request_counts") or ()
    if accepted_counts and len(requests) not in accepted_counts:
        errors.append(
            f"answer request count is {len(requests)}; "
            f"expected one of {accepted_counts!r}"
        )
    elif (
        not accepted_counts
        and isinstance(expected_count, int)
        and len(requests) != expected_count
    ):
        errors.append(
            f"answer request count is {len(requests)}; expected {expected_count}"
        )
    accepted_shapes = context.get("accepted_request_shapes") or ()
    if accepted_shapes:
        actual_shapes = [_request_shape(item) for item in requests]
        if not any(actual_shapes == shapes for shapes in accepted_shapes):
            errors.append(
                f"answer request shapes are {actual_shapes!r}; "
                f"expected one of {accepted_shapes!r}"
            )

    expected_kinds = context.get("expected_result_kinds") or ()
    expected_grouping_counts = context.get("expected_grouping_counts") or ()
    expected_grouping_kinds = context.get("expected_grouping_kinds") or ()
    expected_grouping_values = context.get("expected_grouping_values") or ()
    expected_output_counts = context.get("expected_output_counts") or ()
    expected_output_terms = context.get("expected_output_contains_any") or ()
    expected_selections = context.get("expected_selections") or ()
    for index, raw_request in enumerate(requests):
        if not isinstance(raw_request, dict):
            errors.append(f"answer request {index + 1} is not an object")
            continue
        result_rows = _result(raw_request)
        result_kind = _result_kind(result_rows)
        if index < len(expected_kinds) and result_kind != expected_kinds[index]:
            errors.append(
                f"answer request {index + 1} result kind is {result_kind!r}; "
                f"expected {expected_kinds[index]!r}"
            )

        grouping = _grouping_meanings(result_rows)
        grouping_count = len(grouping) if isinstance(grouping, list) else -1
        if index < len(expected_grouping_values) and isinstance(grouping, list):
            actual_values = [item.get("grouping_value") for item in grouping]
            expected_values = expected_grouping_values[index]
            if Counter(json.dumps(item, sort_keys=True) for item in actual_values) != Counter(
                json.dumps(item, sort_keys=True) for item in expected_values
            ):
                errors.append(f"answer request {index + 1} grouping values are {actual_values!r}; expected {expected_values!r}")
        if (
            index < len(expected_grouping_counts)
            and grouping_count != expected_grouping_counts[index]
        ):
            errors.append(
                f"answer request {index + 1} grouping count is {grouping_count}; "
                f"expected {expected_grouping_counts[index]}"
            )
        if index < len(expected_grouping_kinds) and isinstance(grouping, list):
            actual_kinds = [
                item.get("grouping_kind") for item in grouping if isinstance(item, dict)
            ]
            if actual_kinds != expected_grouping_kinds[index]:
                errors.append(
                    f"answer request {index + 1} grouping kinds are "
                    f"{actual_kinds!r}; expected {expected_grouping_kinds[index]!r}"
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

        result_order = (
            result_rows.get("result_order") if isinstance(result_rows, dict) else None
        )
        selection = (
            result_order.get("selection") if isinstance(result_order, dict) else None
        )
        selection_kind = selection.get("kind") if isinstance(selection, dict) else None
        if (
            index < len(expected_selections)
            and selection_kind != expected_selections[index]
        ):
            errors.append(
                f"answer request {index + 1} selection is {selection_kind!r}; "
                f"expected {expected_selections[index]!r}"
            )
    return errors


def _request_shape(request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        return {"result_kind": None, "grouping_kinds": None}
    result_rows = _result(request)
    grouping = _grouping_meanings(result_rows)
    return {
        "result_kind": _result_kind(result_rows),
        "grouping_kinds": (
            [item.get("grouping_kind") for item in grouping if isinstance(item, dict)]
            if isinstance(grouping, list)
            else None
        ),
    }


def _result_kind(result_rows: object) -> str | None:
    if not isinstance(result_rows, dict):
        return None
    return {
        "one_value_for_population": "one_for_population",
        "one_result_per_qualifying_row": "one_per_candidate",
        "one_result_per_group": "one_per_group",
    }.get(result_rows.get("kind"))


def _grouping_meanings(result_rows: object) -> list[object] | None:
    if not isinstance(result_rows, dict):
        return None
    grouping = result_rows.get("grouping_meanings")
    if isinstance(grouping, list):
        return grouping
    return [] if result_rows.get("kind") != "one_result_per_group" else None


def _returned_meanings(request: dict[str, Any]) -> tuple[str, ...]:
    result = _result(request)
    if not isinstance(result, dict):
        return ()
    scalar_meanings = tuple(
        str(item["meaning"])
        for item in result.get("returned_meanings") or ()
        if isinstance(item, dict) and isinstance(item.get("meaning"), str)
    )
    if scalar_meanings:
        return scalar_meanings
    projection = result.get("projection")
    if not isinstance(projection, dict):
        return ()
    meanings: list[str] = []
    grouping = _meaning_texts(_grouping_meanings(result))
    if grouping:
        meanings.extend(grouping)
    elif projection.get("candidate_identity") == "returned":
        candidate = _row_source(result)
        if isinstance(candidate, dict) and isinstance(
            candidate.get("instance_kind"), str
        ):
            meanings.append(candidate["instance_kind"])
    meanings.extend(_meaning_texts(projection.get("explicitly_requested_values")))
    return tuple(meanings)


def _supplied_values(values: dict[str, object]) -> tuple[SuppliedValue, ...]:
    supplied: list[SuppliedValue] = []
    raw_values = values.get("operands")
    if isinstance(raw_values, list):
        for raw_value in raw_values:
            parsed = _operand_value(raw_value)
            if parsed is not None and not _contains_same_value(supplied, parsed):
                supplied.append(parsed)
    raw_limits = values.get("selection_limits")
    if isinstance(raw_limits, list):
        for raw_limit in raw_limits:
            if not isinstance(raw_limit, dict):
                continue
            non_entity = raw_limit.get("non_entity_value")
            if not isinstance(non_entity, dict):
                continue
            parsed = _supplied_value(
                {
                    "meaning": raw_limit.get("meaning"),
                    "value": non_entity.get("value"),
                },
                denotation="scalar",
            )
            if parsed is not None and not _contains_same_value(supplied, parsed):
                supplied.append(parsed)
    return tuple(supplied)


def _operand_value(raw_value: object) -> SuppliedValue | None:
    if not isinstance(raw_value, dict):
        return None
    common = {"meaning": raw_value.get("meaning")}
    entity_reference = raw_value.get("entity_reference")
    if isinstance(entity_reference, dict):
        identity_value = entity_reference.get("value")
        if not isinstance(identity_value, dict):
            return None
        operands = (
            [identity_value.get("identity_value")]
            if identity_value.get("kind") == "single_identity"
            else identity_value.get("identity_values")
        )
        # The production parser has already validated literal/description objects.
        operands = ([item.get("value") if isinstance(item, dict) else None for item in operands]
                    if isinstance(operands, list) else None)
        return _supplied_value(
            {
                **common,
                "instance_kind": entity_reference.get("instance_kind"),
                "value": {"operands": operands},
            },
            denotation="identity_reference",
        )
    non_entity_value = raw_value.get("non_entity_value")
    if isinstance(non_entity_value, dict):
        return _supplied_value({**common, **non_entity_value}, denotation="scalar")
    return None


def _contains_same_value(
    supplied: list[SuppliedValue], candidate: SuppliedValue
) -> bool:
    return any(
        item.operands == candidate.operands
        and item.denotation == candidate.denotation
        and item.instance_kind == candidate.instance_kind
        and (
            item.value_type == candidate.value_type
            or {item.value_type, candidate.value_type} == {"number", "integer"}
        )
        for item in supplied
    )


def _supplied_value(
    raw_value: object,
    *,
    denotation: str,
) -> SuppliedValue | None:
    if not isinstance(raw_value, dict):
        return None
    value = raw_value.get("value")
    value_type = value.get("value_type") if isinstance(value, dict) else None
    declared_kind = raw_value.get("kind")
    operands = value.get("operands") if isinstance(value, dict) else None
    if not (
        isinstance(raw_value.get("meaning"), str)
        and isinstance(operands, list)
        and all(isinstance(item, str) for item in operands)
    ):
        return None
    if denotation == "identity_reference":
        kind = "identity_name_or_code"
    elif isinstance(declared_kind, str):
        kind = (
            "integer"
            if declared_kind == "number"
            and all(
                input_operand_matches_value_type(operand, IntegerType())
                for operand in operands
            )
            else declared_kind
        )
    elif isinstance(value_type, dict) and isinstance(value_type.get("kind"), str):
        kind = value_type["kind"]
    else:
        return None
    return SuppliedValue(
        meaning=raw_value["meaning"],
        operands=tuple(operands),
        denotation=denotation,
        instance_kind=(
            str(raw_value["instance_kind"])
            if denotation == "identity_reference"
            and raw_value.get("instance_kind") is not None
            else None
        ),
        value_type=kind,
    )


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
        _same_inventory(actual_inventory, tuple(expected)) for expected in inventories
    ):
        errors.append(
            f"supplied-value inventory is {actual_inventory!r}; "
            f"expected one of {inventories!r}"
        )

    required = context.get("required_supplied_values") or {}
    for expected_text, expected_shape in required.items():
        accepted_texts = (
            tuple(str(item) for item in expected_shape.get("accepted_operands", ()))
            if isinstance(expected_shape, dict)
            else ()
        )
        matches = [
            value
            for value in supplied
            if value.represents(str(expected_text))
            or any(value.represents(item) for item in accepted_texts)
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


def _entity_reference_group_errors(
    supplied: tuple[SuppliedValue, ...],
    *,
    context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for expected_group in context.get("required_entity_reference_operand_groups") or ():
        expected_operands = tuple(str(operand) for operand in expected_group)
        exact_matches = [
            value
            for value in supplied
            if value.denotation == "identity_reference"
            and _same_inventory(value.operands, expected_operands)
        ]
        atomic_operands = tuple(
            value.operands[0]
            for value in supplied
            if value.denotation == "identity_reference"
            and len(value.operands) == 1
            and value.operands[0] in expected_operands
        )
        if len(exact_matches) != 1 and not _same_inventory(
            atomic_operands, expected_operands
        ):
            errors.append(
                "entity-reference operand group "
                f"{expected_operands!r} is not represented"
            )
    return errors


def _result(request: object) -> dict[str, Any] | None:
    if not isinstance(request, dict):
        return None
    body = request.get("request")
    result = body.get("result") if isinstance(body, dict) else None
    return result if isinstance(result, dict) else None


def _row_source(result: object) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    for field in (
        "population_rows",
        "result_candidates",
        "grouped_observation_rows",
        "coverage_candidates",
    ):
        value = result.get(field)
        if isinstance(value, dict):
            return value
    return None


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
        return tuple(text for item in value.values() for text in _strings(item))
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value for text in _strings(item))
    return ()
