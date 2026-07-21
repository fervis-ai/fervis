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
    "normal_sale_predicates": "List all completed in-person sales from March 2026.",
    "raw_sale_audit": "For an audit, list every persisted sales record from March 2026 for completed in-person sales.",
    "raw_sale_audit_inclusive": "For an audit, list every persisted sales record from March 2026 for completed in-person sales, including records marked as deleted.",
    "normal_sale_deleted_only": "List all deleted sales from March 2026.",
    "normal_sale_including_deleted": "List all sales from March 2026, including deleted sales.",
    "one_input_two_tests": "How many records have both owner and reviewer equal to her?",
    "result_limit": "Which 3 stores had the most sales this month?",
    "percent_formula": "What is 10% of sales revenue from March 2026?",
    "subtraction_formula": "What is sales revenue from March 2026 minus 500?",
    "unresolved_prior_reference": "What were her sales last week?",
}


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = context.get("label")
    outcome = arguments.get("outcome")
    if label == "unresolved_prior_reference":
        if not isinstance(outcome, dict):
            return ["outcome is not an object"]
        if outcome.get("kind") != "unresolved_prior_turn_references":
            return ["unresolved pronoun did not require prior-turn clarification"]
        references = outcome.get("references")
        if not isinstance(references, list) or not any(
            isinstance(item, dict) and item.get("source_text") == "her"
            for item in references
        ):
            return ["clarification does not identify the unresolved pronoun"]
        return []
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
    question = _QUESTIONS.get(str(label))
    if question is not None and label not in {
        "normal_sale_deleted_only",
        "normal_sale_including_deleted",
    }:
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
    elif label in {
        "normal_sale_predicates",
        "raw_sale_audit",
        "raw_sale_audit_inclusive",
    }:
        if any(
            use.get("owner_kind") != "POPULATION_TESTS"
            for use in owners_by_source.values()
        ):
            errors.append("sale predicate operands are not population-test-owned")
        subject = request.get("answer_subject")
        interpretation = (
            subject.get("instance_interpretation")
            if isinstance(subject, dict)
            else None
        )
        expected_interpretation = (
            "RAW_DATA_RECORD"
            if label in {"raw_sale_audit", "raw_sale_audit_inclusive"}
            else "NORMAL_BUSINESS_INSTANCE"
        )
        if (
            not isinstance(interpretation, dict)
            or interpretation.get("kind") != expected_interpretation
        ):
            errors.append(
                f"sale request is not represented as {expected_interpretation}"
            )
        if label in {"raw_sale_audit", "raw_sale_audit_inclusive"} and any(
            "deleted" in source.casefold() for source in sources_by_ref.values()
        ):
            errors.append("inclusive deleted-state wording became a predicate input")
        tests = request.get("answer_population", {}).get("membership_tests", [])
        explicit_tests = [
            test
            for test in tests
            if isinstance(test, dict)
            and test.get("kind") == "EXPLICIT_USER_CONSTRAINT"
        ]
        for required_text in ("completed", "in-person", "march 2026"):
            matching_tests = [
                test
                for test in explicit_tests
                if required_text in str(test.get("test_question") or "").casefold()
            ]
            test_refs = [
                input_ref
                for test in matching_tests
                for input_ref in test.get("question_input_refs", [])
            ]
            test_operands = " ".join(
                sources_by_ref.get(input_ref, "").casefold()
                for input_ref in test_refs
            )
            if not matching_tests or required_text not in test_operands:
                errors.append(f"{required_text} lacks its supplied predicate operand")
    elif label in {"normal_sale_deleted_only", "normal_sale_including_deleted"}:
        tests = request.get("answer_population", {}).get("membership_tests", [])
        explicit_tests = [
            test
            for test in tests
            if isinstance(test, dict)
            and test.get("kind") == "EXPLICIT_USER_CONSTRAINT"
        ]
        deleted_input_refs = {
            ref
            for ref, source in sources_by_ref.items()
            if "deleted" in source.casefold()
        }
        deleted_test_refs = {
            input_ref
            for test in explicit_tests
            for input_ref in test.get("question_input_refs", [])
            if input_ref in deleted_input_refs
        }
        qualification = request.get("answer_population", {}).get("qualification")
        any_of = (
            qualification.get("any_of")
            if isinstance(qualification, dict)
            else None
        )
        clauses = [
            {
                str(requirement.get("test_id") or ""): str(
                    requirement.get("required_result") or ""
                )
                for requirement in item.get("all_of", [])
                if isinstance(requirement, dict)
            }
            for item in any_of or []
            if isinstance(item, dict)
        ]
        tests_by_id = {
            str(test.get("test_id") or ""): test
            for test in tests
            if isinstance(test, dict)
        }
        deleted_test_ids = {
            test_id
            for test_id, test in tests_by_id.items()
            if "deleted" in str(test.get("test_question") or "").casefold()
        }
        normal_test_ids = {
            test_id
            for test_id, test in tests_by_id.items()
            if test.get("kind") == "NORMAL_INSTANCE_GUARD"
        }
        subject_test_ids = {
            test_id
            for test_id, test in tests_by_id.items()
            if test.get("kind") == "SUBJECT_IDENTITY"
        }
        march_test_ids = {
            test_id
            for test_id, test in tests_by_id.items()
            if "march 2026" in str(test.get("test_question") or "").casefold()
        }
        if not clauses or any(not clause for clause in clauses):
            errors.append("population qualification is not a non-empty any-of/all-of")
        if any(
            polarity
            for test in tests
            if isinstance(test, dict)
            for polarity in [test.get("polarity")]
        ):
            errors.append("atomic membership test still owns composition polarity")
        if any(
            not subject_test_ids.issubset(clause)
            or not march_test_ids.issubset(clause)
            or any(result not in {"PASS", "FAIL"} for result in clause.values())
            for clause in clauses
        ):
            errors.append("qualification clause omits shared requirements or has invalid result")
        if label == "normal_sale_deleted_only":
            if not deleted_input_refs or deleted_test_refs != deleted_input_refs:
                errors.append("required deleted state is not a predicate operand")
            if not clauses or any(
                not any(clause.get(test_id) == "PASS" for test_id in deleted_test_ids)
                for clause in clauses
            ):
                errors.append("deleted-only qualification admits a non-deleted clause")
        else:
            if not deleted_input_refs or deleted_test_refs != deleted_input_refs:
                errors.append("included deleted state lacks its atomic predicate")
            if not any(
                any(clause.get(test_id) == "PASS" for test_id in normal_test_ids)
                and not any(test_id in clause for test_id in deleted_test_ids)
                for clause in clauses
            ):
                errors.append("inclusive qualification omits the normal population")
            if not any(
                any(clause.get(test_id) == "PASS" for test_id in deleted_test_ids)
                and not any(test_id in clause for test_id in normal_test_ids)
                for clause in clauses
            ):
                errors.append("inclusive qualification omits the deleted population")
    elif label == "one_input_two_tests":
        use = owners_by_source.get("Azraah")
        input_ref = use.get("input_ref") if isinstance(use, dict) else None
        consuming_tests = [
            test
            for test in request.get("answer_population", {}).get(
                "membership_tests", []
            )
            if isinstance(test, dict)
            and input_ref in test.get("question_input_refs", [])
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
        if not isinstance(expression, dict) or expression.get("selection") != {
            "kind": "take"
        }:
            errors.append("result-limit question does not use take selection")
    elif label in {"percent_formula", "subtraction_formula"}:
        if set(owners_by_source) not in (
            {"10%", "March 2026"},
            {"500", "March 2026"},
        ):
            errors.append("formula inputs are not all owned by the answer request")
        if owners_by_source.get("March 2026", {}).get("owner_kind") != "POPULATION_TESTS":
            errors.append("formula period is not population-test-owned")
        formula_source = "10%" if label == "percent_formula" else "500"
        if owners_by_source.get(formula_source, {}).get("owner_kind") != "COMPUTE_EXPRESSION":
            errors.append("formula operand is not compute-expression-owned")
    else:
        errors.append(f"unknown assertion label: {label}")
    return errors


def _contains_legacy_owner(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) & {
            "used_question_inputs",
            "owned_question_input_refs",
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
                field_label_text="person",
                value_meaning_hint="person",
            ),
        ),
        frame_call=None,
        used_source_card_ids=(),
        used_memory_ids=(),
    )
