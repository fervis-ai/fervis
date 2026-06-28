"""Catalog-blind prompt projection for question-contract decisions."""

from __future__ import annotations

from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.question_contract.model import QuestionContractRequest
from fervis.lookup.question_contract.schema import (
    build_answer_request_contract_schema,
    build_missing_input_clarification_schema,
)
from fervis.lookup.question_contract.tools import (
    ANSWER_REQUEST_CONTRACT_TOOL_NAME,
    MISSING_INPUT_CLARIFICATION_TOOL_NAME,
)
from fervis.model_io.structured_output.specs import required_tool_spec


class QuestionContractTurnPrompt(TurnPromptBase):
    turn_name = "question contract"
    turn_task = (
        "author the catalog-blind answer request contract for the factual API question"
    )
    include_active_clarification = True

    def __init__(self, request: QuestionContractRequest) -> None:
        self.request = request

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        del builder
        return ()

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Decision Scope",
                (
                    "Interpret the user's factual question intent from the current question and conversation resolution annotations.",
                    "Author the requested facts and the exact question inputs that apply to each requested fact.",
                    "Set answer_requests_count to the number of complete requested facts in the current question plus annotations.",
                    "Each answer_request describes exactly one complete requested fact.",
                    "Do not output implementation IDs, API details, calculations, or execution plans.",
                    "Do not decide API feasibility, data availability, safety, endpoints, fields, operation decomposition, or execution.",
                ),
            ),
            builder.instruction_block(
                "Question Boundary",
                (
                    "Author the contract for the factual intent expressed by the current question in its resolved conversation context.",
                    "Use conversation resolution annotations as the resolved context for current-question words that depend on prior turns.",
                    "If active clarification context is shown, it is also part of the allowed question context for this turn.",
                    "Use only the current question, conversation resolution annotations, and shown active clarification context as question context.",
                    "Return needs_clarification only when visible context is insufficient to author one complete factual question contract.",
                ),
            ),
            builder.instruction_block(
                "Clarification Boundary",
                (
                    "Do not use needs_clarification when visible context is sufficient to author a complete factual question contract.",
                    "Use missing.type=target_reference when the question points to an unresolved person, place, object, row set, period, or other target reference.",
                    "Use missing.type=answer_definition when the user has not supplied the answer definition, metric, comparison baseline, or factual request needed to know what answer to produce.",
                    "For each missing item, copy source_text exactly from the current question or visible context.",
                    "Set why_context_is_insufficient to the specific reason the visible context cannot resolve that missing item.",
                    "Set clarification_question to one direct question that asks only for the missing information.",
                ),
            ),
            builder.instruction_block(
                "Answer Requests",
                (
                    "answer_fact is a complete normalized label for one requested fact.",
                    "answer_expression.family is required and classifies the catalog-blind answer shape, not API execution.",
                    "Use: list_rows for rows/records/details; scalar_value for one direct value; scalar_aggregate for one computed row aggregate; grouped_aggregate for grouped aggregate values; ranked_selection for object(s) selected by ranking/order/optimization; computed_scalar for arithmetic over facts or values; set_difference for members of A not evidenced in B; coverage_check for required coverage present/missing; existence_check for any-match questions; comparison_check for comparing facts, sets, or values.",
                    "Use scalar_aggregate for count answers such as how many X, number of X, or count of X, because the answer is computed from row/population cardinality even when no numeric field is named.",
                    "Use scalar_value only for one direct requested value, not for row/population counts.",
                    "Choose answer_expression.family from the requested answer shape, not endpoints, fields, APIs, or a single keyword.",
                    "answer_subject is required and names the base business subject whose instances the answer will count, list, rank, group, total, or describe.",
                    "Write answer_subject.subject_text as the copied head noun phrase from the current question or conversation resolution, without modifiers that narrow the requested instances.",
                    "answer_subject.instance_interpretation.kind is required.",
                    "Use NORMAL_BUSINESS_INSTANCE for ordinary business reporting questions over the subject as business users normally understand it.",
                    "Use RAW_DATA_RECORD only when the user explicitly asks for persisted records, rows, logs, audit entries, raw data, database entries, or another data artifact.",
                    "input_requirements is required and comes before answer_population.",
                    "input_requirements.time_requirements must have one item for each time word or phrase that constrains this answer_request.",
                    "Use an empty time_requirements array only when no time word or phrase constrains this answer_request.",
                    "Copy each time_requirements.source_text exactly from the current question or conversation resolution annotations.",
                    "Write why_required to explain how that copied time word or phrase constrains this answer_request.",
                    "answer_population is required and defines the final answer population as testable membership rules.",
                    "answer_population.population_label is a concise phrase for the exact population being counted, listed, ranked, grouped, totaled, or described.",
                    "answer_population.counted_unit names one business unit in that population.",
                    "answer_population.membership_tests must include one SUBJECT_IDENTITY test.",
                    "Add one EXPLICIT_USER_CONSTRAINT test for each user-stated state, outcome, lifecycle, time, threshold, channel, type, owner, or filter that changes which subject instances count.",
                    "For NORMAL_BUSINESS_INSTANCE, include a NORMAL_INSTANCE_GUARD test; the backend attaches the standard ORDINARY_BUSINESS_INSTANCE_V1 profile with typed excluded-state roles.",
                    "For RAW_DATA_RECORD, include a RAW_RECORD_GUARD test.",
                    "Each membership test has polarity MUST_PASS unless the user explicitly asks to exclude matching instances, in which case use MUST_FAIL.",
                    "Do not decide which API values, enum options, endpoints, fields, or params pass answer_population tests in this turn.",
                    "answer_outputs contain the values or facts the user asked to receive for that answer_fact.",
                    "Conversation resolution resolved_question_inputs clarify referenced inputs, not answer outputs.",
                    "answer_requests_count must equal the number of answer_requests.",
                    "input_decisions contains one true/false decision for every already-declared question_inputs item.",
                    "Set use_input=true when that question input constrains the answer_request.",
                    "Set use_input=false when that question input does not constrain the answer_request.",
                    "For list or table questions with multiple requested columns about the same rows or groups, use one answer_request and put each requested column in answer_outputs.",
                    "Do not put API details, endpoint names, field names, params, enum values, or execution operations in answer_subject.",
                    "Do not include caveats, proof, data availability checks, endpoint/API terms, execution instructions, or underlying calculation support unless the user explicitly asks for that support as an answer part.",
                ),
            ),
            builder.instruction_block(
                "Question Inputs",
                (
                    "question_inputs is declared before answer_requests.",
                    "question_inputs declares each named reference, resolved row-set reference, time scope, numeric value, threshold, or explicit numeric rank/row limit once.",
                    "Create question_inputs only for lookup values that need grounding, time scopes, numeric thresholds, or limits applied to the answer subject, such as London, today, top five, or 20%.",
                    "Do not create a question_inputs item for answer_subject.subject_text itself.",
                    "Do not create question_inputs for state, type, channel, or lifecycle modifiers that define answer_population membership tests, such as unverified, in-person, open, or canceled.",
                    "Before finalizing question_inputs, actively inventory every word or phrase that is a named reference, resolved row-set reference, time scope, numeric value, threshold, or explicit numeric rank/row limit.",
                    "Declare exactly one question_inputs item for every inventoried phrase.",
                    "Each question_inputs item must represent exactly one input span from the current question or conversation resolution annotations.",
                    "Each question_inputs item must include inventory_check.why_this_is_an_input explaining which input category it belongs to and why it constrains an answer request or supplies a value.",
                    "Use source=question_context for inputs copied directly from the current question.",
                    "Use source=conversation_resolution only for a resolved_question_inputs item whose kind is row_set_reference.",
                    "Every reference_text must be copied exactly from the current question or conversation resolution annotations.",
                    "When conversation resolution annotations include resolved_question_inputs, use those items as the allowed resolved lookup meaning for matching current-question references.",
                    "When resolved_question_inputs includes kind=row_set_reference and that current-question phrase constrains an answer_request, copy it as a row_set_reference input with the same reference_text, occurrence, and resolved_input_ref.",
                    "Do not convert a row_set_reference into named_reference_text.",
                    "Conversation resolution can supply the requested value frame, but time phrases still require matching time_text question inputs.",
                    "For every time_requirements item, declare one matching time_text question input and copy its requirement_id into satisfies_requirement_id.",
                    "A time_text question input satisfies a requirement only when reference_text exactly matches the requirement source_text.",
                    "When one input constrains multiple requested facts, declare it once in question_inputs and set use_input=true for that input on each applicable answer_request.",
                    "After declaring question_inputs and input_decisions, set question_input_inventory_check.all_input_like_phrases_declared=true only when every input-like word or phrase has a question_inputs item.",
                    "Do not compile entity IDs, dates, date ranges, resolver choices, endpoint params, or API values in this turn.",
                ),
            ),
            builder.instruction_block(
                "Named Reference Inputs",
                (
                    "Use kind=named_reference_text for a proper name, code, identifier, or other specific named lookup value in the question context.",
                    "Do not use named_reference_text for answer_subject.subject_text, a generic resource class, answer category, grouping label, pronoun, or question word.",
                    "If the phrase is the same as answer_subject.subject_text, answer_population.counted_unit, or answer_population.population_label, it is not named_reference_text.",
                    "Use one named_reference_text item per named lookup value.",
                    "target_meaning describes the business object type the lookup value refers to.",
                    "lookup_text is the lookup-ready phrase for the same named value.",
                    "If a resolved_question_inputs item matches the reference_text and occurrence, copy that item's lookup_text and target_meaning.",
                    "Keep the actual name, code, or identifier words in lookup_text.",
                    "Remove possessive endings, surrounding punctuation, and grammar that is not part of the name, code, or identifier.",
                    "Do not replace lookup_text with an entity ID, resolver result, API value, synonym, or different business object.",
                    "If reference_text is already lookup-ready, lookup_text can equal reference_text.",
                ),
            ),
            builder.instruction_block(
                "Time Inputs",
                (
                    "Use kind=time_text for calendar dates, calendar date ranges, relative time, calendar periods, quarters, months, years, rolling windows, and open calendar ranges.",
                    "For each time input, copy only the exact reference_text from the question context.",
                    "Set satisfies_requirement_id to the matching input_requirements.time_requirements requirement_id.",
                    "Do not compile date ranges, calendar dates, relative offsets, or time shapes in this turn.",
                    "Use separate time inputs when the user asks for separate dates or periods. Use one range input when the user asks for one combined range.",
                ),
            ),
            builder.instruction_block(
                "Literal Inputs",
                (
                    "Use kind=explicit_numeric_limit_text only for explicit rank or row limits with a copied number expression, such as Top 5, top five, first 10, or bottom three.",
                    "For explicit_numeric_limit_text, set value_source_text to the exact copied word or digit span that expresses the number, such as 5, five, 10, or three.",
                    "Do not use a question input for ranking words without an explicit number, such as most, highest, least, best, or top item.",
                    "Use kind=number_text for numeric thresholds or values such as 20%.",
                    "Set numeric_value to the numeric interpretation of the copied value_source_text.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Return exactly one provider-native tool call.",
                    "Use submit_answer_request_contract when visible context is sufficient to author complete answer requests.",
                    "Use submit_missing_input_clarification only when visible context is insufficient to author one complete factual question contract.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                ANSWER_REQUEST_CONTRACT_TOOL_NAME: (
                    build_answer_request_contract_schema()
                ),
                MISSING_INPUT_CLARIFICATION_TOOL_NAME: (
                    build_missing_input_clarification_schema()
                ),
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=ANSWER_REQUEST_CONTRACT_TOOL_NAME,
                    tool_description="Submit complete catalog-blind answer request contracts.",
                    input_schema=build_answer_request_contract_schema(),
                ),
                required_tool_spec(
                    tool_name=MISSING_INPUT_CLARIFICATION_TOOL_NAME,
                    tool_description=(
                        "Submit a missing-input clarification request for the question-contract turn."
                    ),
                    input_schema=build_missing_input_clarification_schema(),
                ),
            )
        )
