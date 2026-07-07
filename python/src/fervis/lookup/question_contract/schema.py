"""Provider schema projection for catalog-blind question-contract decisions."""

from __future__ import annotations

from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.question_contract import provider_contract as provider_output
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole


def build_question_contract_decisions_schema() -> dict[str, object]:
    return {
        "oneOf": [
            build_answer_request_contract_schema(),
            build_missing_input_clarification_schema(),
        ]
    }


def build_answer_request_contract_schema(
    *,
    include_conversation_resolution_inputs: bool = True,
) -> dict[str, object]:
    return provider_output.QuestionContractOutput.schema(
        {
            "kind": {"enum": ["question_contract"]},
            "answer_requests_count": {"type": "integer", "minimum": 1},
            "question_inputs": {
                "type": "array",
                "items": _question_input_schema(
                    include_conversation_resolution_inputs=(
                        include_conversation_resolution_inputs
                    ),
                ),
            },
            "answer_requests": {
                "type": "array",
                "minItems": 1,
                "items": _answer_request_schema(),
            },
            "question_input_inventory_check": (
                provider_output.QuestionInputInventoryCheckOutput.schema(
                {
                    "all_input_like_phrases_declared": {"type": "boolean"},
                },
                )
            ),
        },
    )


def build_missing_input_clarification_schema() -> dict[str, object]:
    return provider_output.MissingInputClarificationOutput.schema(
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
    )


def _missing_question_input_schema() -> dict[str, object]:
    return provider_output.MissingQuestionInputOutput.schema(
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
    )


def _answer_request_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "answer_fact": {"type": "string", "minLength": 1},
        "answer_expression": _answer_expression_schema(),
        "answer_subject": _answer_subject_schema(),
        "answer_population": _answer_population_schema(),
        "answer_outputs": {
            "type": "array",
            "minItems": 1,
            "items": _answer_output_schema(),
        },
        "used_question_inputs": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    }
    return provider_output.AnswerRequestOutput.schema(
        properties,
    )


def _answer_expression_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _grouped_answer_expression_schema(),
            _ordinary_answer_expression_schema(),
        ]
    }


def _grouped_answer_expression_schema() -> dict[str, object]:
    schema = provider_output.AnswerExpressionOutput.schema(
        {
            "family": {"enum": ["grouped_aggregate"]},
            "group_key": _group_key_schema(),
        },
    )
    schema["required"] = ["family", "group_key"]
    return schema


def _ordinary_answer_expression_schema() -> dict[str, object]:
    return provider_output.AnswerExpressionOutput.schema(
        {
            "family": {
                "enum": [
                    "list_rows",
                    "scalar_value",
                    "scalar_aggregate",
                    "ranked_selection",
                    "computed_scalar",
                    "set_difference",
                    "coverage_check",
                    "existence_check",
                    "comparison_check",
                ]
            },
        },
    )


def _answer_subject_schema() -> dict[str, object]:
    return provider_output.AnswerSubjectOutput.schema(
        {
            "subject_text": {"type": "string", "minLength": 1},
            "instance_interpretation": (
                provider_output.AnswerSubjectInstanceInterpretationOutput.schema(
                {
                    "kind": {
                        "enum": [
                            "NORMAL_BUSINESS_INSTANCE",
                            "RAW_DATA_RECORD",
                        ]
                    },
                },
                )
            ),
        },
    )


def _answer_population_schema() -> dict[str, object]:
    return provider_output.AnswerPopulationOutput.schema(
        {
            "population_label": {"type": "string", "minLength": 1},
            "counted_unit": {"type": "string", "minLength": 1},
            "membership_tests": {
                "type": "array",
                "minItems": 1,
                "items": _answer_population_membership_test_schema(),
            },
        },
    )


def _answer_population_membership_test_schema() -> dict[str, object]:
    return provider_output.AnswerPopulationMembershipTestOutput.schema(
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
            "owned_question_input_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    )


def _answer_output_schema() -> dict[str, object]:
    return provider_output.AnswerOutputOutput.schema(
        {
            "description": {"type": "string", "minLength": 1},
            "role": {
                "enum": [
                    role for role in ANSWER_OUTPUT_SUPPORT_ROLE_VALUES
                    if role != "GROUP_KEY"
                ]
            },
        },
    )


def _group_key_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _specified_question_inputs_group_key_schema(),
            _source_result_values_group_key_schema(),
        ]
    }


def _specified_question_inputs_group_key_schema() -> dict[str, object]:
    schema = provider_output.GroupKeyOutput.schema(
        {
            "description": {"type": "string", "minLength": 1},
            "domain": {"enum": ["SPECIFIED_QUESTION_INPUTS"]},
            "question_input_refs": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
    )
    schema["required"] = ["description", "domain", "question_input_refs"]
    return schema


def _source_result_values_group_key_schema() -> dict[str, object]:
    return provider_output.GroupKeyOutput.schema(
        {
            "description": {"type": "string", "minLength": 1},
            "domain": {"enum": ["SOURCE_RESULT_VALUES"]},
        },
    )


def _question_input_schema(
    *,
    include_conversation_resolution_inputs: bool,
) -> dict[str, object]:
    branches = [
        _literal_text_input_role_schema(
            role=LiteralInputRole.REFERENCE_VALUE,
            include_conversation_resolution_inputs=(
                include_conversation_resolution_inputs
            ),
        ),
        _literal_text_input_role_schema(
            role=LiteralInputRole.RESULT_LIMIT,
            include_conversation_resolution_inputs=(
                include_conversation_resolution_inputs
            ),
        ),
        _literal_text_input_role_schema(
            role=LiteralInputRole.TIME_VALUE,
            include_conversation_resolution_inputs=(
                include_conversation_resolution_inputs
            ),
        ),
    ]
    if include_conversation_resolution_inputs:
        branches.append(_row_set_reference_input_schema())
    return {"oneOf": branches}


def _literal_text_input_role_schema(
    *,
    role: LiteralInputRole,
    include_conversation_resolution_inputs: bool,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "input_ref": {"type": "string", "minLength": 1},
        "source": {
            "enum": (
                ["question_context", "conversation_resolution"]
                if include_conversation_resolution_inputs
                else ["question_context"]
            )
        },
        "value_source_text": {"type": "string", "minLength": 1},
        "resolved_value_text": {"type": "string", "minLength": 1},
        "field_label_text": {"type": "string", "minLength": 1},
        "value_meaning_hint": {"type": "string", "minLength": 1},
        "role": {"enum": [role.value]},
        "occurrence": {"type": "integer", "minimum": 1},
        "inventory_check": _question_input_inventory_check_schema(),
        "kind": {"enum": [KnownInputKind.LITERAL.value]},
    }
    if include_conversation_resolution_inputs:
        properties["resolved_input_ref"] = {"type": "string", "minLength": 1}
    return provider_output.LiteralTextInputOutput.schema(
        properties,
    )


def _question_input_inventory_check_schema() -> dict[str, object]:
    return provider_output.QuestionInputItemInventoryCheckOutput.schema(
        {
            "why_this_is_an_input": {"type": "string", "minLength": 1},
        },
    )


def _row_set_reference_input_schema() -> dict[str, object]:
    return provider_output.RowSetReferenceInputOutput.schema(
        {
            "input_ref": {"type": "string", "minLength": 1},
            "source": {"enum": ["conversation_resolution"]},
            "reference_text": {"type": "string", "minLength": 1},
            "occurrence": {"type": "integer", "minimum": 1},
            "resolved_input_ref": {"type": "string", "minLength": 1},
            "inventory_check": _question_input_inventory_check_schema(),
            "kind": {"enum": [KnownInputKind.ROW_SET_REFERENCE.value]},
        },
    )
