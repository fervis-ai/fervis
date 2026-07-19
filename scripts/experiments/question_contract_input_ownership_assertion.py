from __future__ import annotations

from typing import Any

from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
    ResolvedLiteralQuestionInput,
)
from fervis.lookup.question_contract.parser import parse_question_contract
from fervis.lookup.question_contract.tools import QUESTION_CONTRACT_TOOL_NAME
from fervis.lookup.question_inputs import LiteralInputRole


_QUESTIONS = {
    "pair_grouped": "How many sales did the staff members with ids 51515151-0000-0000-0002-000000000001 and 51515151-0000-0000-0002-000000000002 sell each today?",
    "single_staff": "How many sales did the staff with staff_id: 51515151-0000-0000-0002-000000000001 sell today?",
    "choice_and_time": "How many in-person sales happened this month?",
    "one_input_two_tests": "How many records have both owner and reviewer equal to her?",
    "result_limit": "Which 3 stores had the most sales this month?",
}


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    outcome = arguments.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("kind") != "question_contract":
        return ["outcome is not a question contract"]
    inputs = outcome.get("question_inputs")
    requests = outcome.get("answer_requests")
    if not isinstance(inputs, list) or not isinstance(requests, list) or len(requests) != 1:
        return ["expected question inputs and exactly one answer request"]
    request = requests[0]
    if not isinstance(request, dict):
        return ["answer request is not an object"]
    uses = request.get("question_input_uses")
    if not isinstance(uses, list):
        return ["question_input_uses is not an array"]
    refs = [item.get("input_ref") for item in uses if isinstance(item, dict)]
    if len(refs) != len(uses) or len(refs) != len(set(refs)):
        errors.append("question input refs are not owned exactly once")
    input_by_ref = {
        item.get("input_ref"): item for item in inputs if isinstance(item, dict)
    }
    if set(refs) != set(input_by_ref):
        errors.append("fact-local uses do not cover the declared inputs")
    if _contains_legacy_owner(request):
        errors.append("output contains a legacy ownership field")

    sources_by_ref = {
        ref: str(
            item.get("operand_text")
            or item.get("value_source_text")
            or item.get("reference_text")
            or ""
        )
        for ref, item in input_by_ref.items()
    }
    owners_by_source = {
        sources_by_ref.get(item.get("input_ref")): item
        for item in uses
        if isinstance(item, dict)
    }
    label = context.get("label")
    question = _QUESTIONS.get(str(label))
    if question is not None:
        try:
            parse_question_contract(
                tool_name=QUESTION_CONTRACT_TOOL_NAME,
                payload=arguments,
                question_context=question,
                conversation_resolution=_conversation_resolution(str(label)),
            )
        except ValueError as exc:
            errors.append(f"production parser rejected output: {exc}")
    if label == "pair_grouped":
        expected_groups = {
            "51515151-0000-0000-0002-000000000001",
            "51515151-0000-0000-0002-000000000002",
        }
        actual_groups = {
            source
            for source, use in owners_by_source.items()
            if use.get("owner_kind") == "GROUP_KEY"
        }
        if actual_groups != expected_groups:
            errors.append("staff inputs are not exclusively GROUP_KEY-owned")
        if owners_by_source.get("today", {}).get("owner_kind") != "POPULATION_TESTS":
            errors.append("today is not population-test-owned")
        expression = request.get("answer_expression")
        group_key = expression.get("group_key") if isinstance(expression, dict) else None
        if not isinstance(group_key, dict) or group_key.get("domain") != "SPECIFIED_QUESTION_INPUTS":
            errors.append("pair question lacks its specified-input group key")
    elif label == "single_staff":
        expected = {
            "51515151-0000-0000-0002-000000000001",
            "today",
        }
        if set(owners_by_source) != expected or any(
            use.get("owner_kind") != "POPULATION_TESTS"
            for use in owners_by_source.values()
        ):
            errors.append("single-staff operands are not population-test-owned")
    elif label == "choice_and_time":
        if set(owners_by_source) != {"in-person", "this month"} or any(
            use.get("owner_kind") != "POPULATION_TESTS"
            for use in owners_by_source.values()
        ):
            errors.append("choice/time operands are not population-test-owned")
    elif label == "one_input_two_tests":
        use = owners_by_source.get("Azraah")
        use_id = use.get("use_id") if isinstance(use, dict) else None
        consuming_tests = [
            test
            for test in request.get("answer_population", {}).get(
                "membership_tests", []
            )
            if isinstance(test, dict)
            and use_id in test.get("question_input_use_refs", [])
        ]
        if (
            not isinstance(use, dict)
            or use.get("owner_kind") != "POPULATION_TESTS"
            or len(consuming_tests) != 2
        ):
            errors.append("Azraah does not supply both population tests")
    elif label == "result_limit":
        limit_use = owners_by_source.get("3")
        time_use = owners_by_source.get("this month")
        expression = request.get("answer_expression")
        if (
            not isinstance(limit_use, dict)
            or limit_use.get("owner_kind") != "RESULT_LIMIT"
        ):
            errors.append("3 is not result-limit-owned")
        if (
            not isinstance(time_use, dict)
            or time_use.get("owner_kind") != "POPULATION_TESTS"
        ):
            errors.append("this month is not population-test-owned")
        if not isinstance(expression, dict) or expression.get("family") != "ranked_selection":
            errors.append("result-limit question is not ranked_selection")
    else:
        errors.append(f"unknown assertion label: {label}")
    return errors


def _contains_legacy_owner(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) & {
            "used_question_inputs",
            "owned_question_input_refs",
            "question_input_refs",
        }:
            return True
        return any(_contains_legacy_owner(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_legacy_owner(item) for item in value)
    return False


def _conversation_resolution(
    label: str,
) -> CompiledConversationResolution | None:
    if label != "one_input_two_tests":
        return None
    question = _QUESTIONS[label]
    return CompiledConversationResolution(
        current_question_text=question,
        contextualized_question=question,
        clauses=(),
        inputs=(
            ResolvedLiteralQuestionInput(
                input_ref="shared_person",
                value_source_text="her",
                resolved_value_text="Azraah",
                role=LiteralInputRole.REFERENCE_VALUE,
                field_label_text="owner and reviewer",
                value_meaning_hint="person",
            ),
        ),
        frame_call=None,
        used_source_card_ids=(),
        used_memory_ids=(),
    )
