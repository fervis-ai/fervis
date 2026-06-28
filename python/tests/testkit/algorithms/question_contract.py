from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator, ValidationError

from fervis.lookup.question_contract import (
    ANSWER_REQUEST_CONTRACT_TOOL_NAME,
    MISSING_INPUT_CLARIFICATION_TOOL_NAME,
    QuestionContractRequest,
    QuestionContractTurnPrompt,
    build_question_contract_decisions_schema,
    parse_question_contract,
)
from fervis.lookup.conversation_resolution import (
    ConversationDependencyOverlay,
    ConversationResolutionOverlay,
    ConversationValueFrameOverlay,
    conversation_resolution_question_contract_prompt_payload,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context

from tests.testkit.assertions import subset_mismatches


def run_question_contract_parse_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    model_payload = dict(
        input_payload.get("payload")
        or _model_payload_from_case_input(input_payload)
    )
    tool_name = str(input_payload.get("tool_name") or "")
    if not tool_name:
        tool_name = (
            MISSING_INPUT_CLARIFICATION_TOOL_NAME
            if model_payload.get("kind") == "needs_clarification"
            else ANSWER_REQUEST_CONTRACT_TOOL_NAME
        )
    try:
        result = parse_question_contract(
            tool_name=tool_name,
            payload=model_payload,
            question_context=str(input_payload["question_context"]),
            question_context_texts=tuple(input_payload.get("question_context_texts") or ()),
        )
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    actual = {
        "question_inputs": [
            {
                "id": item.id,
                "kind": item.kind.value,
                "source": item.source.value,
                "text": item.text,
                "description": item.description,
                "numeric_value": item.numeric_value,
                "value_source_text": item.value_source_text,
                "lookup_text": item.lookup_text,
                "resolved_input_ref": item.resolved_input_ref,
            }
            for item in result.outcome.question_inputs
        ],
        "requested_facts": [
            {
                "id": fact.id,
                "description": fact.description,
                "answer_expression_family": (
                    fact.answer_expression.family.value
                    if fact.answer_expression is not None
                    else ""
                ),
                "answer_subject_text": (
                    fact.answer_subject.subject_text
                    if fact.answer_subject is not None
                    else ""
                ),
                "input_refs": list(fact.input_refs),
                "known_inputs": [
                    {
                        "id": item.id,
                        "kind": item.kind.value,
                        "text": item.text,
                        "numeric_value": item.numeric_value,
                        "lookup_text": item.lookup_text,
                    }
                    for item in fact.known_inputs
                ],
                "answer_outputs": [
                    {"id": output.id, "description": output.description}
                    for output in fact.answer_outputs
                ],
                "answer_request": fact.answer_request_model_dict(),
            }
            for fact in result.outcome.requested_facts
        ],
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_question_contract_schema_case(payload: dict[str, Any]) -> list[str]:
    schema = build_question_contract_decisions_schema()
    branches = {
        branch["properties"]["kind"]["enum"][0]: branch
        for branch in schema["oneOf"]
    }
    answer_contract_schema = branches["question_contract"]
    clarification_schema = branches["needs_clarification"]
    answer_request_schema = answer_contract_schema["properties"]["answer_requests"][
        "items"
    ]
    question_input_item = answer_contract_schema["properties"]["question_inputs"][
        "items"
    ]
    variants = {
        variant["properties"]["kind"]["enum"][0]: variant
        for variant in question_input_item["oneOf"]
    }
    actual = {
        "has_root_one_of": "oneOf" in schema,
        "branch_kinds": sorted(branches),
        "answer_contract_required": list(answer_contract_schema["required"]),
        "answer_contract_properties": list(answer_contract_schema["properties"]),
        "answer_contract_kind_values": answer_contract_schema["properties"]["kind"][
            "enum"
        ],
        "answer_contract_has_clarification_fields": any(
            field in answer_contract_schema["properties"]
            for field in ("missing", "clarification_question")
        ),
        "clarification_required": list(clarification_schema["required"]),
        "clarification_properties": list(clarification_schema["properties"]),
        "clarification_kind_values": clarification_schema["properties"]["kind"][
            "enum"
        ],
        "clarification_has_answer_fields": any(
            field in clarification_schema["properties"]
            for field in (
                "answer_requests_count",
                "question_inputs",
                "answer_requests",
                "question_input_inventory_check",
            )
        ),
        "answer_request_required": answer_request_schema["required"],
        "answer_output_required": answer_request_schema["properties"][
            "answer_outputs"
        ]["items"]["required"],
        "answer_output_properties": sorted(
            answer_request_schema["properties"]["answer_outputs"]["items"][
                "properties"
            ]
        ),
        "question_input_kinds": sorted(variants),
        "question_input_kind_membership": {
            "time_text": "time_text" in variants,
            "explicit_numeric_limit_text": "explicit_numeric_limit_text" in variants,
            "named_reference_text": "named_reference_text" in variants,
            "number_text": "number_text" in variants,
            "row_set_reference": "row_set_reference" in variants,
        },
        "row_set_reference_properties": sorted(
            variants["row_set_reference"]["properties"]
        ),
        "row_set_reference_property_membership": {
            name: name in variants["row_set_reference"]["properties"]
            for name in (
                "input_ref",
                "source",
                "reference_text",
                "occurrence",
                "resolved_input_ref",
                "inventory_check",
                "kind",
                "target_meaning",
            )
        },
        "row_set_reference_required": variants["row_set_reference"]["required"],
        "schema_text": repr(schema),
    }
    errors = subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"].get("result_contains") or {},
    )
    for text in payload["expect"].get("text_excludes") or ():
        if text in actual["schema_text"]:
            errors.append(f"unexpected text present: {text!r}")
    return errors


def run_question_contract_schema_validate_case(payload: dict[str, Any]) -> list[str]:
    instance = dict(
        payload["input"].get("payload")
        or _model_payload_from_case_input(payload["input"])
    )
    schema = build_question_contract_decisions_schema()
    errors = list(Draft7Validator(schema).iter_errors(instance))
    if errors:
        expected_error = payload["expect"].get("error_contains")
        error_text = " | ".join(_validation_error_text(error) for error in errors)
        if expected_error and expected_error in error_text:
            return []
        return [f"unexpected validation error: {error_text}"]
    if "error_contains" in payload["expect"]:
        return [f"expected validation error containing {payload['expect']['error_contains']!r}"]
    return []


def _validation_error_text(error: ValidationError) -> str:
    messages: list[str] = []

    def collect(item: ValidationError) -> None:
        messages.append(item.message)
        for child in item.context:
            collect(child)

    collect(error)
    return " | ".join(messages)


def run_question_contract_prompt_case(payload: dict[str, Any]) -> list[str]:
    request = QuestionContractRequest(
        current_question=str(payload["input"]["current_question"]),
        conversation_context=dict(payload["input"].get("conversation_context") or {}),
    )
    conversation_resolution_overlay = payload["input"].get(
        "conversation_resolution_overlay"
    )
    invocation = QuestionContractTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.current_question,
            conversation_context=request.conversation_context,
            conversation_resolution_overlay=(
                conversation_resolution_question_contract_prompt_payload(
                    _conversation_overlay(conversation_resolution_overlay)
                )
                if isinstance(conversation_resolution_overlay, dict)
                else None
            ),
        )
    )
    actual = {
        "prompt_text": invocation.prompt_text,
        "provider_schema_text": repr(invocation.provider_schema),
        "current_question_present": request.current_question in invocation.prompt_text,
        "contains": {
            text: text in invocation.prompt_text
            for text in payload["input"].get("contains") or ()
        },
        "excludes": {
            text: text not in invocation.prompt_text
            for text in payload["input"].get("excludes") or ()
        },
    }
    errors: list[str] = []
    for field, excluded_values in (
        payload["expect"].get("text_excludes_from") or {}
    ).items():
        text = str(actual.get(field) or "")
        for value in excluded_values:
            if value in text:
                errors.append(f"{field} contains excluded text: {value!r}")
    expected_subset = payload["expect"].get("result_contains") or {}
    if expected_subset:
        errors.extend(subset_mismatches(actual=actual, expected_subset=expected_subset))
    return errors


def _conversation_overlay(payload: dict[str, Any]) -> ConversationResolutionOverlay:
    return ConversationResolutionOverlay(
        current_question=str(payload["current_question"]),
        value_frames=tuple(
            ConversationValueFrameOverlay(
                current_clause_text=str(item["current_clause_text"]),
                current_value_text=str(item["current_value_text"]),
                current_value_kind=str(item["current_value_kind"]),
                resolved_frame_text=str(item["resolved_frame_text"]),
                must_preserve_terms=tuple(item.get("must_preserve_terms") or ()),
                used_context_frame_ids=tuple(item.get("used_context_frame_ids") or ()),
            )
            for item in payload.get("value_frames") or ()
        ),
        references=tuple(
            ConversationDependencyOverlay(
                current_clause_text=str(item["current_clause_text"]),
                anchor_text=str(item["anchor_text"]),
                occurrence=int(item.get("occurrence") or 1),
                resolved_text=str(item["resolved_text"]),
                must_preserve_terms=tuple(item.get("must_preserve_terms") or ()),
                source_ids=tuple(item.get("source_ids") or ()),
            )
            for item in payload.get("references") or ()
        ),
        scopes=(),
        activated_memory_ids=(),
        used_source_card_ids=(),
    )


def _model_payload_from_case_input(input_payload: dict[str, Any]) -> dict[str, object]:
    question_inputs = list(input_payload.get("question_inputs") or ())
    answer_requests = list(input_payload.get("answer_requests") or ())
    if not answer_requests:
        answer_requests = [_answer_request(input_payload)]
    return {
        "kind": "question_contract",
        "answer_requests_count": int(
            input_payload.get("answer_requests_count") or len(answer_requests)
        ),
        "question_inputs": question_inputs,
        "answer_requests": answer_requests,
        "question_input_inventory_check": dict(
            input_payload.get("question_input_inventory_check")
            or {"all_input_like_phrases_declared": True}
        ),
    }


def _answer_request(input_payload: dict[str, Any]) -> dict[str, object]:
    question_inputs = list(input_payload.get("question_inputs") or ())
    used_input_refs = set(input_payload.get("used_input_refs") or ())
    request = {
        "answer_fact": str(input_payload.get("answer_fact") or "sales at ABC Mall"),
        "answer_expression": dict(
            input_payload.get("answer_expression") or {"family": "scalar_aggregate"}
        ),
        "answer_subject": dict(
            input_payload.get("answer_subject")
            or {
                "subject_text": "sales",
                "instance_interpretation": {"kind": "NORMAL_BUSINESS_INSTANCE"},
            }
        ),
        "input_requirements": dict(
            input_payload.get("input_requirements")
            or {"time_requirements": []}
        ),
        "answer_population": dict(
            input_payload.get("answer_population")
            or {
                "population_label": "sales",
                "counted_unit": "sales",
                "membership_tests": [
                    {
                        "test_id": "pop_test_1",
                        "kind": "SUBJECT_IDENTITY",
                        "polarity": "MUST_PASS",
                        "test_question": "Does the row/value represent sales?",
                    }
                ],
            }
        ),
        "answer_outputs": list(
            input_payload.get("answer_outputs") or [{"description": "sales total"}]
        ),
        "input_decisions": [
            {
                "input_ref": str(item["input_ref"]),
                "use_input": str(item["input_ref"]) in used_input_refs,
            }
            for item in question_inputs
            if isinstance(item, dict) and str(item.get("input_ref") or "").strip()
        ],
    }
    request.update(dict(input_payload.get("answer_request_overrides") or {}))
    return request
