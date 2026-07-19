"""Provider schema projection for catalog-blind question-contract decisions."""

from __future__ import annotations

from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.question_contract import provider_contract as provider_output
from fervis.lookup.question_inputs import (
    KnownInputKind,
    LiteralInputRole,
)
from fervis.lookup.conversation_resolution.compilation import (
    ResolvedLiteralQuestionInput,
    ResolvedQuestionInput,
    ResolvedRowSetQuestionInput,
)


def build_question_contract_decisions_schema(
    *,
    conversation_inputs: tuple[ResolvedQuestionInput, ...] = (),
) -> dict[str, object]:
    return provider_output.QuestionContractDecisionOutput.schema(
        {
            "decision_basis": {"type": "string", "minLength": 1},
            "outcome": {
                "oneOf": [
                    build_answer_request_contract_schema(
                        conversation_inputs=conversation_inputs,
                    ),
                    *_incomplete_factual_request_schemas(),
                ]
            },
        }
    )


def build_answer_request_contract_schema(
    *,
    conversation_inputs: tuple[ResolvedQuestionInput, ...] = (),
) -> dict[str, object]:
    return provider_output.QuestionContractOutput.schema(
        {
            "kind": {"enum": ["question_contract"]},
            "answer_requests_count": {"type": "integer", "minimum": 1},
            "question_inputs": {
                "type": "array",
                "items": _question_input_schema(
                    conversation_inputs=conversation_inputs,
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


def build_incomplete_factual_request_schema() -> dict[str, object]:
    return {"oneOf": list(_incomplete_factual_request_schemas())}


def _incomplete_factual_request_schemas() -> tuple[dict[str, object], ...]:
    return (
        _missing_requested_fact_schema(),
        _unresolved_prior_turn_references_schema(),
    )


def _missing_requested_fact_schema() -> dict[str, object]:
    return provider_output.MissingRequestedFactOutput.schema(
        {
            "kind": {"enum": ["missing_requested_fact"]},
            "source_text": {"type": "string", "minLength": 1},
            "why_question_is_incomplete": {
                "type": "string",
                "minLength": 1,
            },
        },
    )


def _unresolved_prior_turn_references_schema() -> dict[str, object]:
    return provider_output.UnresolvedPriorTurnReferencesOutput.schema(
        {
            "kind": {"enum": ["unresolved_prior_turn_references"]},
            "references": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": _unresolved_prior_turn_reference_schema(),
            },
        },
    )


def _unresolved_prior_turn_reference_schema() -> dict[str, object]:
    return provider_output.UnresolvedPriorTurnReferenceOutput.schema(
        {
            "source_text": {"type": "string", "minLength": 1},
            "target_label": {"type": "string", "minLength": 1},
            "why_question_is_incomplete": {
                "type": "string",
                "minLength": 1,
            },
        },
    )


def _answer_request_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "answer_fact": {"type": "string", "minLength": 1},
        "answer_expression": _answer_expression_schema(),
        "question_input_uses": {
            "type": "array",
            "items": _question_input_use_schema(),
        },
        "answer_subject": _answer_subject_schema(),
        "answer_population": _answer_population_schema(),
        "answer_outputs": {
            "type": "array",
            "minItems": 1,
            "items": _answer_output_schema(),
        },
    }
    return provider_output.AnswerRequestOutput.schema(
        properties,
    )


def _answer_expression_schema() -> dict[str, object]:
    return {
        "oneOf": [
            *(
                _relation_answer_expression_schema(
                    family=family,
                    selection=selection,
                    ordered=ordered,
                )
                for family in ("list_rows", "grouped_aggregate")
                for selection, ordered in (
                    ("all_results", False),
                    ("all_results", True),
                    ("take_one", True),
                    ("take", True),
                )
            ),
            _scalar_or_set_answer_expression_schema(),
        ]
    }


def _relation_answer_expression_schema(
    *, family: str, selection: str, ordered: bool
) -> dict[str, object]:
    properties: dict[str, object] = {
        "family": {"enum": [family]},
        "selection": provider_output.ResultSelectionOutput.schema(
            {"kind": {"enum": [selection]}}
        ),
    }
    required = ["family", "selection"]
    if family == "grouped_aggregate":
        properties["group_key"] = _group_key_schema()
        required.append("group_key")
    if ordered:
        properties["ordering"] = provider_output.OrderingOutput.schema(
            {
                "basis": {"type": "string", "minLength": 1},
                "direction": {"enum": ["ascending", "descending"]},
            }
        )
        required.append("ordering")
    schema = provider_output.AnswerExpressionOutput.schema(properties)
    schema["required"] = required
    return schema


def _scalar_or_set_answer_expression_schema() -> dict[str, object]:
    return provider_output.AnswerExpressionOutput.schema(
        {
            "family": {
                "enum": [
                    "scalar_value",
                    "scalar_aggregate",
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
    return {
        "oneOf": [
            _answer_population_membership_test_variant(
                kind="EXPLICIT_USER_CONSTRAINT",
            ),
            *(
                _answer_population_membership_test_variant(
                    kind=kind,
                )
                for kind in (
                    "SUBJECT_IDENTITY",
                    "NORMAL_INSTANCE_GUARD",
                    "RAW_RECORD_GUARD",
                )
            ),
        ]
    }


def _answer_population_membership_test_variant(
    *,
    kind: str,
) -> dict[str, object]:
    use_refs: dict[str, object] = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
    }
    if kind == "EXPLICIT_USER_CONSTRAINT":
        use_refs["minItems"] = 1
    else:
        use_refs["maxItems"] = 0
    return provider_output.AnswerPopulationMembershipTestOutput.schema(
        {
            "question_input_use_refs": use_refs,
            "test_id": {"type": "string", "minLength": 1},
            "kind": {"enum": [kind]},
            "polarity": {"enum": ["MUST_PASS", "MUST_FAIL"]},
            "test_question": {"type": "string", "minLength": 1},
        }
    )


def _answer_output_schema() -> dict[str, object]:
    return provider_output.AnswerOutputOutput.schema(
        {
            "description": {"type": "string", "minLength": 1},
            "role": {
                "enum": [
                    role
                    for role in ANSWER_OUTPUT_SUPPORT_ROLE_VALUES
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
    return provider_output.GroupKeyOutput.schema(
        {
            "description": {"type": "string", "minLength": 1},
            "domain": {"enum": ["SPECIFIED_QUESTION_INPUTS"]},
        },
    )


def _source_result_values_group_key_schema() -> dict[str, object]:
    return provider_output.GroupKeyOutput.schema(
        {
            "description": {"type": "string", "minLength": 1},
            "domain": {"enum": ["SOURCE_RESULT_VALUES"]},
        },
    )


def _question_input_use_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _question_input_use_variant(
                provider_output.QuestionInputOwnerKind.GROUP_KEY,
            ),
            _question_input_use_variant(
                provider_output.QuestionInputOwnerKind.POPULATION_TESTS,
                include_use_id=True,
            ),
            _question_input_use_variant(
                provider_output.QuestionInputOwnerKind.COMPUTE_EXPRESSION,
            ),
            _question_input_use_variant(
                provider_output.QuestionInputOwnerKind.RESULT_LIMIT,
            ),
        ]
    }


def _question_input_use_variant(
    owner_kind: provider_output.QuestionInputOwnerKind,
    *,
    include_use_id: bool = False,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "input_ref": {"type": "string", "minLength": 1},
        "owner_kind": {"enum": [owner_kind.value]},
    }
    if include_use_id:
        properties["use_id"] = {"type": "string", "minLength": 1}
    return provider_output.QuestionInputUseOutput.schema(properties)


def _question_input_schema(
    *,
    conversation_inputs: tuple[ResolvedQuestionInput, ...],
) -> dict[str, object]:
    branches = [
        _literal_text_input_role_schema(
            role=LiteralInputRole.REFERENCE_VALUE,
            include_conversation_resolution_inputs=False,
        ),
        _literal_text_input_role_schema(
            role=LiteralInputRole.TIME_VALUE,
            include_conversation_resolution_inputs=False,
        ),
        _literal_text_input_role_schema(
            role=LiteralInputRole.FORMULA_VALUE,
            include_conversation_resolution_inputs=False,
        ),
        _literal_text_input_role_schema(
            role=LiteralInputRole.RESULT_LIMIT,
            include_conversation_resolution_inputs=False,
        ),
    ]
    if conversation_inputs:
        branches.extend(
            _declared_conversation_input_schema(item) for item in conversation_inputs
        )
    return {"oneOf": branches}


def _declared_conversation_input_schema(
    item: ResolvedQuestionInput,
) -> dict[str, object]:
    match item:
        case ResolvedLiteralQuestionInput():
            return _declared_literal_input_schema(item)
        case ResolvedRowSetQuestionInput():
            return _declared_row_set_input_schema(item)


def _declared_literal_input_schema(
    item: ResolvedLiteralQuestionInput,
) -> dict[str, object]:
    if item.role is None:
        raise ValueError("literal conversation input requires role")
    properties: dict[str, object] = {
        "input_ref": {"type": "string", "minLength": 1},
        "source": {"enum": ["conversation_resolution"]},
        "value_source_text": {"enum": [item.value_source_text]},
        "operand_text": {"enum": [item.resolved_value_text]},
        "role": {"enum": [item.role.value]},
        "occurrence": {"type": "integer", "minimum": 1},
        "resolved_input_ref": {"enum": [item.input_ref]},
        "inventory_check": _question_input_inventory_check_schema(),
        "kind": {"enum": [KnownInputKind.LITERAL.value]},
    }
    required_optional_fields = ["resolved_input_ref"]
    if item.field_label_text:
        properties["field_label_text"] = {"enum": [item.field_label_text]}
        required_optional_fields.append("field_label_text")
    if item.value_meaning_hint:
        properties["value_meaning_hint"] = {"enum": [item.value_meaning_hint]}
        required_optional_fields.append("value_meaning_hint")
    schema = provider_output.LiteralTextInputOutput.schema(properties)
    required = schema["required"]
    if not isinstance(required, list):
        raise ValueError("provider schema required fields must be an array")
    schema["required"] = [*required, *required_optional_fields]
    return schema


def _declared_row_set_input_schema(
    item: ResolvedRowSetQuestionInput,
) -> dict[str, object]:
    return provider_output.RowSetReferenceInputOutput.schema(
        {
            "input_ref": {"type": "string", "minLength": 1},
            "source": {"enum": ["conversation_resolution"]},
            "reference_text": {"enum": [item.reference_text]},
            "occurrence": {"type": "integer", "minimum": 1},
            "resolved_input_ref": {"enum": [item.input_ref]},
            "inventory_check": _question_input_inventory_check_schema(),
            "kind": {"enum": [KnownInputKind.ROW_SET_REFERENCE.value]},
        }
    )


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
        "operand_text": {"type": "string", "minLength": 1},
        "field_label_text": {"type": "string", "minLength": 1},
        "value_meaning_hint": {"type": "string", "minLength": 1},
        "role": {"enum": [role.value]},
        "occurrence": {"type": "integer", "minimum": 1},
        "inventory_check": _question_input_inventory_check_schema(),
        "kind": {"enum": [KnownInputKind.LITERAL.value]},
    }
    if include_conversation_resolution_inputs:
        properties["resolved_input_ref"] = {"type": "string", "minLength": 1}
    schema = provider_output.LiteralTextInputOutput.schema(properties)
    return schema


def _question_input_inventory_check_schema() -> dict[str, object]:
    return provider_output.QuestionInputItemInventoryCheckOutput.schema(
        {
            "why_this_is_an_input": {"type": "string", "minLength": 1},
        },
    )
