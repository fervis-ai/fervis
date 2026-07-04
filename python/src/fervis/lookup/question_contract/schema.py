"""Provider schema projection for catalog-blind question-contract decisions."""

from __future__ import annotations


def _strict_object(
    properties: dict[str, object],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


def build_question_contract_decisions_schema() -> dict[str, object]:
    return {
        "oneOf": [
            build_answer_request_contract_schema(),
            build_missing_input_clarification_schema(),
        ]
    }


def build_answer_request_contract_schema() -> dict[str, object]:
    return _strict_object(
        {
            "kind": {"enum": ["question_contract"]},
            "answer_requests_count": {"type": "integer", "minimum": 1},
            "question_inputs": {
                "type": "array",
                "items": _question_input_schema(),
            },
            "answer_requests": {
                "type": "array",
                "minItems": 1,
                "items": _answer_request_schema(),
            },
            "question_input_inventory_check": _strict_object(
                {
                    "all_input_like_phrases_declared": {"type": "boolean"},
                },
                required=("all_input_like_phrases_declared",),
            ),
        },
        required=(
            "kind",
            "answer_requests_count",
            "question_inputs",
            "answer_requests",
            "question_input_inventory_check",
        ),
    )


def build_missing_input_clarification_schema() -> dict[str, object]:
    return _strict_object(
        {
            "kind": {"enum": ["needs_clarification"]},
            "missing": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": _missing_question_input_schema(),
            },
            "clarification_question": {"type": "string", "minLength": 1},
        },
        required=("kind", "missing", "clarification_question"),
    )


def _missing_question_input_schema() -> dict[str, object]:
    return _strict_object(
        {
            "type": {
                "enum": [
                    "target_reference",
                    "answer_definition",
                ]
            },
            "source_text": {"type": "string", "minLength": 1},
            "entity_type": {"type": "string"},
            "why_context_is_insufficient": {
                "type": "string",
                "minLength": 1,
            },
        },
        required=(
            "type",
            "source_text",
            "entity_type",
            "why_context_is_insufficient",
        ),
    )


def _answer_request_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "answer_fact": {"type": "string", "minLength": 1},
        "answer_expression": _answer_expression_schema(),
        "answer_subject": _answer_subject_schema(),
        "input_requirements": _input_requirements_schema(),
        "answer_population": _answer_population_schema(),
        "answer_outputs": {
            "type": "array",
            "minItems": 1,
            "items": _answer_output_schema(),
        },
        "input_decisions": {
            "type": "array",
            "items": _input_decision_schema(),
        },
    }
    required = [
        "answer_fact",
        "answer_expression",
        "answer_subject",
        "input_requirements",
        "answer_population",
        "answer_outputs",
        "input_decisions",
    ]
    return _strict_object(
        properties,
        required=tuple(required),
    )


def _answer_expression_schema() -> dict[str, object]:
    return _strict_object(
        {
            "family": {
                "enum": [
                    "list_rows",
                    "scalar_value",
                    "scalar_aggregate",
                    "grouped_aggregate",
                    "ranked_selection",
                    "computed_scalar",
                    "set_difference",
                    "coverage_check",
                    "existence_check",
                    "comparison_check",
                ]
            },
        },
        required=("family",),
    )


def _input_requirements_schema() -> dict[str, object]:
    return _strict_object(
        {
            "time_requirements": {
                "type": "array",
                "items": _time_requirement_schema(),
            },
        },
        required=("time_requirements",),
    )


def _time_requirement_schema() -> dict[str, object]:
    return _strict_object(
        {
            "requirement_id": {"type": "string", "minLength": 1},
            "source_text": {"type": "string", "minLength": 1},
            "why_required": {"type": "string", "minLength": 1},
        },
        required=("requirement_id", "source_text", "why_required"),
    )


def _answer_subject_schema() -> dict[str, object]:
    return _strict_object(
        {
            "subject_text": {"type": "string", "minLength": 1},
            "instance_interpretation": _strict_object(
                {
                    "kind": {
                        "enum": [
                            "NORMAL_BUSINESS_INSTANCE",
                            "RAW_DATA_RECORD",
                        ]
                    },
                },
                required=("kind",),
            ),
        },
        required=("subject_text", "instance_interpretation"),
    )


def _answer_population_schema() -> dict[str, object]:
    return _strict_object(
        {
            "population_label": {"type": "string", "minLength": 1},
            "counted_unit": {"type": "string", "minLength": 1},
            "membership_tests": {
                "type": "array",
                "minItems": 1,
                "items": _answer_population_membership_test_schema(),
            },
        },
        required=("population_label", "counted_unit", "membership_tests"),
    )


def _answer_population_membership_test_schema() -> dict[str, object]:
    return _strict_object(
        {
            "test_id": {"type": "string", "minLength": 1},
            "kind": {
                "enum": [
                    "SUBJECT_IDENTITY",
                    "EXPLICIT_USER_CONSTRAINT",
                    "NORMAL_INSTANCE_GUARD",
                    "RAW_RECORD_GUARD",
                ]
            },
            "polarity": {"enum": ["MUST_PASS", "MUST_FAIL"]},
            "test_question": {"type": "string", "minLength": 1},
        },
        required=("test_id", "kind", "polarity", "test_question"),
    )


def _input_decision_schema() -> dict[str, object]:
    return _strict_object(
        {
            "input_ref": {"type": "string", "minLength": 1},
            "use_input": {"type": "boolean"},
        },
        required=("input_ref", "use_input"),
    )


def _answer_output_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "description": {"type": "string", "minLength": 1},
    }
    required = ["description"]
    return _strict_object(
        properties,
        required=tuple(required),
    )


def _question_input_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _literal_text_input_schema(),
            _row_set_reference_input_schema(),
        ]
    }


def _literal_text_input_schema() -> dict[str, object]:
    return _strict_object(
        {
            "input_ref": {"type": "string", "minLength": 1},
            "source": {"enum": ["question_context", "conversation_resolution"]},
            "source_text": {"type": "string", "minLength": 1},
            "resolved_value_text": {"type": "string", "minLength": 1},
            "field_label_text": {"type": "string", "minLength": 1},
            "value_meaning_hint": {"type": "string", "minLength": 1},
            "role": {
                "enum": [
                    "reference_value",
                    "result_limit",
                    "time_value",
                ]
            },
            "satisfies_requirement_id": {"type": "string", "minLength": 1},
            "resolved_input_ref": {"type": "string", "minLength": 1},
            "inventory_check": _question_input_inventory_check_schema(),
            "kind": {"enum": ["literal_text"]},
        },
        required=(
            "input_ref",
            "source",
            "source_text",
            "resolved_value_text",
            "role",
            "inventory_check",
            "kind",
        ),
    )


def _question_input_inventory_check_schema() -> dict[str, object]:
    return _strict_object(
        {
            "why_this_is_an_input": {"type": "string", "minLength": 1},
        },
        required=("why_this_is_an_input",),
    )


def _row_set_reference_input_schema() -> dict[str, object]:
    return _strict_object(
        {
            "input_ref": {"type": "string", "minLength": 1},
            "source": {"enum": ["conversation_resolution"]},
            "reference_text": {"type": "string", "minLength": 1},
            "occurrence": {"type": "integer", "minimum": 1},
            "resolved_input_ref": {"type": "string", "minLength": 1},
            "inventory_check": _question_input_inventory_check_schema(),
            "kind": {"enum": ["row_set_reference"]},
        },
        required=(
            "input_ref",
            "source",
            "reference_text",
            "occurrence",
            "resolved_input_ref",
            "inventory_check",
            "kind",
        ),
    )
