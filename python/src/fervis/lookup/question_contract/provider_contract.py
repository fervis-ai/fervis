"""Provider-output DTOs for question contract."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


QuestionContractOutput = provider_output_type(
    "QuestionContractOutput",
    (
        "kind",
        "answer_requests_count",
        "question_inputs",
        "answer_requests",
        "question_input_inventory_check",
    ),
)
MissingInputClarificationOutput = provider_output_type(
    "MissingInputClarificationOutput",
    ("kind", "missing", "clarification_question"),
)
MissingQuestionInputOutput = provider_output_type(
    "MissingQuestionInputOutput",
    ("type", "source_text", "entity_type", "why_context_is_insufficient"),
)
AnswerRequestOutput = provider_output_type(
    "AnswerRequestOutput",
    (
        "answer_fact",
        "answer_expression",
        "answer_subject",
        "answer_population",
        "answer_outputs",
        "used_question_inputs",
    ),
)
AnswerExpressionOutput = provider_output_type(
    "AnswerExpressionOutput",
    ("family", "group_key"),
    optional_fields=("group_key",),
)
GroupKeyOutput = provider_output_type(
    "GroupKeyOutput",
    ("description", "domain", "question_input_refs"),
    optional_fields=("question_input_refs",),
)
AnswerSubjectOutput = provider_output_type(
    "AnswerSubjectOutput",
    ("subject_text", "instance_interpretation"),
)
AnswerSubjectInstanceInterpretationOutput = provider_output_type(
    "AnswerSubjectInstanceInterpretationOutput",
    ("kind",),
)
AnswerPopulationOutput = provider_output_type(
    "AnswerPopulationOutput",
    ("population_label", "counted_unit", "membership_tests"),
)
AnswerPopulationMembershipTestOutput = provider_output_type(
    "AnswerPopulationMembershipTestOutput",
    ("test_id", "kind", "polarity", "test_question", "owned_question_input_refs"),
)
AnswerOutputOutput = provider_output_type(
    "AnswerOutputOutput",
    ("description", "role"),
    optional_fields=("role",),
)
LiteralTextInputOutput = provider_output_type(
    "LiteralTextInputOutput",
    (
        "input_ref",
        "source",
        "value_source_text",
        "resolved_value_text",
        "field_label_text",
        "value_meaning_hint",
        "role",
        "occurrence",
        "resolved_input_ref",
        "inventory_check",
        "kind",
    ),
    optional_fields=(
        "field_label_text",
        "value_meaning_hint",
        "resolved_input_ref",
        "occurrence",
    ),
)
RowSetReferenceInputOutput = provider_output_type(
    "RowSetReferenceInputOutput",
    (
        "input_ref",
        "source",
        "reference_text",
        "occurrence",
        "resolved_input_ref",
        "inventory_check",
        "kind",
    ),
)
QuestionInputInventoryCheckOutput = provider_output_type(
    "QuestionInputInventoryCheckOutput",
    ("all_input_like_phrases_declared",),
)
QuestionInputItemInventoryCheckOutput = provider_output_type(
    "QuestionInputItemInventoryCheckOutput",
    ("why_this_is_an_input",),
)
