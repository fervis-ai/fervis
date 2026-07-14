from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator, ValidationError

from fervis.lookup.question_contract import (
    QUESTION_CONTRACT_TOOL_NAME,
    QuestionContractRequest,
    QuestionContractTurnPrompt,
    RequestedFactKnownInput,
    build_answer_request_contract_schema,
    build_question_contract_decisions_schema,
    parse_question_contract,
)
from fervis.lookup.question_inputs import KnownInputKind
from fervis.lookup.conversation_resolution.compilation import ResolvedIdentityInput
from fervis.lookup.turn_prompts import build_turn_prompt_context
from tests.testkit.algorithms.conversation_resolution import (
    compiled_conversation_resolution_from_payload,
)

from tests.testkit.assertions import (
    exact_mismatches,
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def _conversation_identity_payload(
    item: ResolvedIdentityInput,
) -> dict[str, object]:
    identity = item.canonical_identity
    payload: dict[str, object] = {
        "input_ref": item.input_ref,
        "entity_kind": identity.key.entity_kind,
        "key_id": identity.key.key_id,
    }
    if len(identity.key.components) == 1:
        component = identity.key.components[0]
        payload["key_component_id"] = component.component_id
        payload["value"] = component.value
    else:
        payload["components"] = identity.key.component_values()
    return payload


def run_question_contract_parse_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    model_payload = dict(
        input_payload.get("payload") or _model_payload_from_case_input(input_payload)
    )
    conversation_resolution = compiled_conversation_resolution_from_payload(
        input_payload.get("conversation_resolution")
    )
    question_context_texts = list(input_payload.get("question_context_texts") or ())
    if conversation_resolution is not None:
        question_context_texts.extend(conversation_resolution.context_texts())
    tool_name = str(input_payload.get("tool_name") or "")
    if not tool_name:
        tool_name = QUESTION_CONTRACT_TOOL_NAME
    decision_payload = {
        "decision_basis": str(
            input_payload.get("decision_basis")
            or "The current wording supports the selected outcome."
        ),
        "outcome": model_payload,
    }
    try:
        result = parse_question_contract(
            tool_name=tool_name,
            payload=decision_payload,
            question_context=str(input_payload["question_context"]),
            question_context_texts=tuple(question_context_texts),
            conversation_resolution=conversation_resolution,
        )
    except ValueError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    actual = {
        "question_inputs": [
            _known_input_actual(item) for item in result.outcome.question_inputs
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
                    _known_input_actual(item) for item in fact.known_inputs
                ],
                "answer_outputs": [
                    output.to_model_dict() for output in fact.answer_outputs
                ],
                "answer_request": fact.answer_request_model_dict(),
            }
            for fact in result.outcome.requested_facts
        ],
    }
    expected_result = (
        payload["expect"].get("result_equals")
        or payload["expect"].get("result_contains")
        or {}
    )
    if "conversation_identity_inputs" in expected_result:
        actual["conversation_identity_inputs"] = [
            _conversation_identity_payload(item)
            for item in (
                conversation_resolution.identity_inputs()
                if conversation_resolution is not None
                else ()
            )
        ]
    if "decision_basis" in expected_result:
        actual["decision_basis"] = result.decision_basis
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    actual["requested_fact_input_handoff"] = [
        {
            "id": fact["id"],
            "input_refs": fact["input_refs"],
            "known_input_refs": [item["id"] for item in fact["known_inputs"]],
        }
        for fact in actual["requested_facts"]
    ]
    errors = subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"].get("result_contains") or {},
    )
    for field, expected_value in (
        payload["expect"].get("result_exact_fields") or {}
    ).items():
        errors.extend(
            exact_mismatches(
                actual=actual.get(field),
                expected=expected_value,
                path=field,
            )
        )
    return errors


def _known_input_actual(item: RequestedFactKnownInput) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": item.id,
        "kind": item.kind.value,
        "source": item.source.value,
        "text": item.text,
        "resolved_input_ref": item.resolved_input_ref,
    }
    if item.kind == KnownInputKind.ROW_SET_REFERENCE:
        payload["occurrence"] = item.occurrence
        return payload
    payload["resolved_value_text"] = item.resolved_value_text
    payload["field_label_text"] = item.field_label_text
    payload["value_meaning_hint"] = item.value_meaning_hint
    payload["role"] = item.role.value
    return payload


def run_question_contract_schema_case(payload: dict[str, Any]) -> list[str]:
    if payload.get("input", {}).get("projection") == "declared_conversation_inputs":
        return _declared_conversation_input_schema_case(payload)
    schema = build_question_contract_decisions_schema()
    outcome_schema = schema["properties"]["outcome"]
    branches = {
        branch["properties"]["kind"]["enum"][0]: branch
        for branch in outcome_schema["oneOf"]
    }
    answer_contract_schema = branches["question_contract"]
    missing_fact_schema = branches["missing_requested_fact"]
    unresolved_schema = branches["unresolved_prior_turn_references"]
    answer_request_schema = answer_contract_schema["properties"]["answer_requests"][
        "items"
    ]
    answer_output_item_schema = answer_request_schema["properties"]["answer_outputs"][
        "items"
    ]
    answer_expression_schema = answer_request_schema["properties"]["answer_expression"]
    question_input_item = answer_contract_schema["properties"]["question_inputs"][
        "items"
    ]
    question_input_variants = tuple(question_input_item["oneOf"])
    variants = {
        variant["properties"]["kind"]["enum"][0]: variant
        for variant in question_input_variants
        if variant["properties"]["kind"]["enum"][0] != KnownInputKind.LITERAL.value
    }
    literal_variants = tuple(
        variant
        for variant in question_input_variants
        if variant["properties"]["kind"]["enum"][0] == KnownInputKind.LITERAL.value
    )
    literal_properties = sorted(
        {
            property_name
            for variant in literal_variants
            for property_name in variant["properties"]
        }
    )
    literal_role_values = sorted(
        {
            role
            for variant in literal_variants
            for role in variant["properties"]["role"]["enum"]
        }
    )
    row_set_variant = variants.get(KnownInputKind.ROW_SET_REFERENCE.value)
    actual = {
        "root_required": list(schema["required"]),
        "root_properties": list(schema["properties"]),
        "has_outcome_one_of": "oneOf" in outcome_schema,
        "branch_kinds": sorted(branches),
        "answer_contract_required": list(answer_contract_schema["required"]),
        "answer_contract_properties": list(answer_contract_schema["properties"]),
        "answer_contract_kind_values": answer_contract_schema["properties"]["kind"][
            "enum"
        ],
        "answer_contract_has_clarification_fields": any(
            field in answer_contract_schema["properties"] for field in ("missing",)
        ),
        "missing_fact_required": list(missing_fact_schema["required"]),
        "missing_fact_properties": list(missing_fact_schema["properties"]),
        "unresolved_required": list(unresolved_schema["required"]),
        "unresolved_properties": list(unresolved_schema["properties"]),
        "unresolved_reference_properties": list(
            unresolved_schema["properties"]["references"]["items"]["properties"]
        ),
        "clarifications_have_answer_fields": any(
            field in branch["properties"]
            for branch in (missing_fact_schema, unresolved_schema)
            for field in (
                "answer_requests_count",
                "question_inputs",
                "answer_requests",
                "question_input_inventory_check",
            )
        ),
        "answer_request_required": answer_request_schema["required"],
        "answer_expression_schema_kind": _branching_schema_kind(
            answer_expression_schema
        ),
        "answer_expression_one_of_branch_count": len(
            answer_expression_schema.get("oneOf") or ()
        ),
        "grouped_answer_expression_branch": _answer_expression_branch_summary(
            _answer_expression_grouped_branch(answer_expression_schema)
        ),
        "ordinary_answer_expression_branch": _answer_expression_branch_summary(
            _answer_expression_ordinary_branch(answer_expression_schema)
        ),
        "answer_output_schema_kind": _answer_output_schema_kind(
            answer_output_item_schema
        ),
        "answer_output_one_of_branch_count": len(
            answer_output_item_schema.get("oneOf") or ()
        ),
        "answer_output_branch": _answer_output_branch_summary(
            answer_output_item_schema
        ),
        "question_input_kinds": sorted(
            {
                variant["properties"]["kind"]["enum"][0]
                for variant in question_input_variants
            }
        ),
        "literal_text_role_values": literal_role_values,
        "literal_text_properties": literal_properties,
        "literal_text_property_membership": {
            name: name in literal_properties
            for name in (
                "input_ref",
                "kind",
                "source",
                "value_source_text",
                "resolved_value_text",
                "field_label_text",
                "value_meaning_hint",
                "role",
                "inventory_check",
            )
        },
        "row_set_reference_properties": sorted(
            row_set_variant["properties"] if row_set_variant is not None else ()
        ),
        "row_set_reference_property_membership": {
            name: (
                row_set_variant is not None and name in row_set_variant["properties"]
            )
            for name in (
                "input_ref",
                "source",
                "reference_text",
                "occurrence",
                "resolved_input_ref",
                "inventory_check",
                "kind",
            )
        },
        "row_set_reference_required": (
            row_set_variant["required"] if row_set_variant is not None else []
        ),
        "schema_text": repr(schema),
    }
    if payload.get("input", {}).get("projection") == "question_input_contract":
        actual = {
            "question_input_kinds": actual["question_input_kinds"],
            "literal_text_role_values": actual["literal_text_role_values"],
            "literal_text_properties": actual["literal_text_properties"],
            "row_set_reference_properties": actual["row_set_reference_properties"],
            "row_set_reference_required": actual["row_set_reference_required"],
        }
    if "result_equals" in payload["expect"]:
        errors = exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    else:
        errors = subset_mismatches(
            actual=actual,
            expected_subset=payload["expect"].get("result_contains") or {},
        )
    for field, expected_value in (
        payload["expect"].get("result_exact_fields") or {}
    ).items():
        errors.extend(
            exact_mismatches(
                actual=actual.get(field),
                expected=expected_value,
                path=field,
            )
        )
    return errors


def _declared_conversation_input_schema_case(payload: dict[str, Any]) -> list[str]:
    conversation_resolution = compiled_conversation_resolution_from_payload(
        payload["input"].get("conversation_resolution")
    )
    schema = build_answer_request_contract_schema(
        conversation_inputs=(
            conversation_resolution.inputs
            if conversation_resolution is not None
            else ()
        )
    )
    variants = schema["properties"]["question_inputs"]["items"]["oneOf"]
    declared_variants = [
        variant
        for variant in variants
        if variant["properties"]["source"]["enum"] == ["conversation_resolution"]
    ]
    actual = {
        "declared_variants": [
            {
                "kind": variant["properties"]["kind"]["enum"],
                "source": variant["properties"]["source"]["enum"],
                "value_source_text": variant["properties"].get(
                    "value_source_text",
                    variant["properties"].get("reference_text"),
                )["enum"],
                "resolved_value_text": variant["properties"]
                .get("resolved_value_text", {})
                .get("enum", []),
                "role": variant["properties"].get("role", {}).get("enum", []),
                "resolved_input_ref": variant["properties"]["resolved_input_ref"][
                    "enum"
                ],
            }
            for variant in declared_variants
        ]
    }
    return exact_mismatches(
        actual=actual,
        expected=payload["expect"]["result_equals"],
    )


def _answer_output_schema_kind(schema: dict[str, Any]) -> str:
    return _branching_schema_kind(schema)


def _branching_schema_kind(schema: dict[str, Any]) -> str:
    if "oneOf" in schema:
        return "oneOf"
    return "object"


def _answer_expression_grouped_branch(schema: dict[str, Any]) -> dict[str, Any]:
    return next(
        (
            branch
            for branch in schema.get("oneOf") or ()
            if branch.get("properties", {}).get("family", {}).get("enum")
            == ["grouped_aggregate"]
        ),
        {},
    )


def _answer_expression_ordinary_branch(schema: dict[str, Any]) -> dict[str, Any]:
    return next(
        (
            branch
            for branch in schema.get("oneOf") or ()
            if branch.get("properties", {}).get("family", {}).get("enum")
            != ["grouped_aggregate"]
        ),
        {},
    )


def _answer_expression_branch_summary(branch: dict[str, Any]) -> dict[str, object]:
    properties = branch.get("properties") or {}
    group_key_schema = properties.get("group_key") or {}
    return {
        "required": list(branch.get("required") or ()),
        "properties": sorted(properties),
        "family_enum": list((properties.get("family") or {}).get("enum") or ()),
        "allows_group_key": "group_key" in properties,
        "additional_properties": bool(branch.get("additionalProperties", True)),
        "group_key_schema_kind": _branching_schema_kind(group_key_schema),
        "group_key_branch_count": len(group_key_schema.get("oneOf") or ()),
    }


def _answer_output_branch_summary(branch: dict[str, Any]) -> dict[str, object]:
    properties = branch.get("properties") or {}
    role_schema = properties.get("role") or {}
    return {
        "required": list(branch.get("required") or ()),
        "properties": sorted(properties),
        "role_enum": list(role_schema.get("enum") or ()),
        "additional_properties": bool(branch.get("additionalProperties", True)),
    }


def run_question_contract_schema_validate_case(payload: dict[str, Any]) -> list[str]:
    instance = dict(
        payload["input"].get("payload")
        or _model_payload_from_case_input(payload["input"])
    )
    schema = build_question_contract_decisions_schema()
    errors = list(Draft7Validator(schema).iter_errors(instance))
    if errors:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        error_text = " | ".join(_validation_error_text(error) for error in errors)
        return [f"unexpected validation error: {error_text}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
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
        conversation_resolution=compiled_conversation_resolution_from_payload(
            payload["input"].get("conversation_resolution")
        ),
    )
    invocation = QuestionContractTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.current_question,
            conversation_context=request.conversation_context,
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
    expected_subset = payload["expect"].get("result_contains") or {}
    if expected_subset:
        errors.extend(subset_mismatches(actual=actual, expected_subset=expected_subset))
    return errors


def _model_payload_from_case_input(input_payload: dict[str, Any]) -> dict[str, object]:
    question_inputs = _question_inputs_from_case_input(input_payload)
    answer_requests = list(input_payload.get("answer_requests") or ())
    if not answer_requests:
        answer_requests = [
            _answer_request(input_payload, question_inputs=question_inputs)
        ]
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


def _question_inputs_from_case_input(
    input_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _provider_question_input(dict(item))
        for item in input_payload.get("question_inputs") or ()
    ]


def _provider_question_input(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("kind") != KnownInputKind.LITERAL.value:
        return payload
    payload["operand_text"] = payload.pop("resolved_value_text")
    return payload


def _answer_request(
    input_payload: dict[str, Any],
    *,
    question_inputs: list[dict[str, Any]],
) -> dict[str, object]:
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
                        "owned_question_input_refs": [],
                    }
                ],
            }
        ),
        "answer_outputs": list(
            input_payload.get("answer_outputs")
            or [{"description": "sales total", "role": "MEASURED_VALUE"}]
        ),
        "used_question_inputs": [
            input_ref
            for item in question_inputs
            if isinstance(item, dict)
            and (input_ref := str(item.get("input_ref") or "").strip())
            and input_ref in used_input_refs
        ],
    }
    request.update(dict(input_payload.get("answer_request_overrides") or {}))
    return request
