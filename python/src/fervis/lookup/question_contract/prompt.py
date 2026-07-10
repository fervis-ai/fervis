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
        provenance_payload = (
            self.request.conversation_input_provenance.to_prompt_payload()
        )
        if not provenance_payload:
            return ()
        return (
            builder.json_section(
                "Conversation input provenance:",
                provenance_payload,
                indent=2,
            ),
        )

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
                    "A repeated measure over a specified key set is one grouped requested fact, not one fact per key.",
                    "Do not output implementation IDs, API details, calculations, or execution plans.",
                    "Do not decide API feasibility, data availability, safety, endpoints, fields, operation decomposition, or execution.",
                ),
            ),
            builder.instruction_block(
                "Question Boundary",
                (
                    "Author the contract for the factual intent expressed by the current question in its resolved conversation context.",
                    "When conversation input provenance includes resolved_request_text, use it as the factual meaning of the current question.",
                    "Use conversation input provenance as authoritative input provenance; do not recreate carried prior inputs from resolved_request_text.",
                    "Use conversation resolution annotations as the resolved context for current-question words that depend on prior turns.",
                    "If active clarification context is shown, it is also part of the allowed question context for this turn.",
                    "Use only the current question, conversation input provenance, conversation resolution annotations, and shown active clarification context as question context.",
                    "Return needs_clarification only when visible context is insufficient to author one complete factual question contract.",
                ),
            ),
            builder.instruction_block(
                "Clarification Boundary",
                (
                    "Do not use needs_clarification when visible context is sufficient to author a complete factual question contract.",
                    "Use missing.type=target_reference when the question points to an unresolved person, place, object, row set, period, or other target reference.",
                    "Use missing.type=answer_definition when the user has not supplied the answer definition, metric, comparison baseline, or factual request needed to know what answer to produce.",
                    "For each missing item, copy source_text verbatim from the current question or visible context.",
                    "Set why_context_is_insufficient to the specific reason the visible context cannot resolve that missing item.",
                ),
            ),
            builder.instruction_block(
                "Answer Requests",
                (
                    "answer_fact is a complete normalized label for one requested fact.",
                    "answer_expression.family is required and classifies the catalog-blind answer shape, not API execution.",
                    "Use: list_rows for rows/records/details; scalar_value for one direct value; scalar_aggregate for one computed row aggregate; grouped_aggregate for grouped aggregate values; ranked_selection for object(s) selected by ranking/order/optimization; computed_scalar for arithmetic over facts or values; set_difference for members of A not evidenced in B; coverage_check for required coverage present/missing; existence_check for any-match questions; comparison_check for comparing facts, sets, or values.",
                    "Use scalar_aggregate for count answers only when the requested result is one scalar count for the whole requested population, such as how many X, number of X, or count of X.",
                    "If the question asks for counts per group, by group, or for each specified key, use grouped_aggregate.",
                    "For grouped_aggregate, set answer_expression.group_key.",
                    "answer_expression.group_key.description names the result key or grouping dimension, such as region, period, category, or supplied key.",
                    "Use answer_expression.group_key.domain=SPECIFIED_QUESTION_INPUTS when groups are exactly declared question_inputs; otherwise use SOURCE_RESULT_VALUES.",
                    "With SPECIFIED_QUESTION_INPUTS, question_input_refs must list only inputs that define the result key axis, not filter-only inputs such as time, status, lifecycle, threshold, or channel.",
                    "For a repeated measure over specified inputs, set answer_expression.group_key with domain=SPECIFIED_QUESTION_INPUTS and put one measure/count result column in answer_outputs, not one output per key value.",
                    "Use scalar_value only for one direct requested value, not for row/population counts.",
                    "Choose answer_expression.family from the requested answer shape, not endpoints, fields, APIs, or a single keyword.",
                    "answer_subject is required and names the base business subject whose instances the answer will count, list, rank, group, total, or describe.",
                    "Write answer_subject.subject_text as the copied head noun phrase from the current question or conversation resolution, without modifiers that narrow the requested instances.",
                    "answer_subject.instance_interpretation.kind is required.",
                    "Use NORMAL_BUSINESS_INSTANCE for ordinary business reporting questions over the subject as business users normally understand it.",
                    "Use RAW_DATA_RECORD only when the user explicitly asks for persisted records, rows, logs, audit entries, raw data, database entries, or another data artifact.",
                    "answer_population is required and defines the final answer population as testable membership rules.",
                    "answer_population.population_label is a concise phrase for the exact population being counted, listed, ranked, grouped, totaled, or described.",
                    "answer_population.counted_unit names one business unit in that population.",
                    "answer_population.membership_tests must include one SUBJECT_IDENTITY test.",
                    "Add one EXPLICIT_USER_CONSTRAINT test for each user-stated state, outcome, lifecycle, time, threshold, channel, type, owner, or filter that narrows the subject instances before the requested result is produced.",
                    "For each membership test, set owned_question_input_refs to the question_input ids whose concrete values the test consumes.",
                    "Use owned_question_input_refs=[] when a membership test is not owned by a concrete question_input, including normal instance and raw record guards.",
                    "Do not use used_question_inputs as test-level ownership proof; owned_question_input_refs is the membership-test owner edge.",
                    "For NORMAL_BUSINESS_INSTANCE, include a NORMAL_INSTANCE_GUARD test; the backend attaches the standard ORDINARY_BUSINESS_INSTANCE_V1 profile with typed excluded-state roles.",
                    "For RAW_DATA_RECORD, include a RAW_RECORD_GUARD test.",
                    "Each membership test has polarity MUST_PASS unless the user explicitly asks to exclude matching instances, in which case use MUST_FAIL.",
                    "Do not decide which API values, enum options, endpoints, fields, or params pass answer_population tests in this turn.",
                    "answer_requests_count must equal the number of answer_requests.",
                    "used_question_inputs lists only the question_inputs that constrain this answer_request.",
                    "Do not list question_inputs that do not constrain this answer_request.",
                    "Do not put API details, endpoint names, field names, params, enum values, or execution operations in answer_subject.",
                    "Do not include caveats, proof, data availability checks, endpoint/API terms, execution instructions, or underlying calculation support unless the user explicitly asks for that support as an answer part.",
                ),
            ),
            builder.instruction_block(
                "Answer Outputs",
                (
                    "answer_outputs contain the values or facts the user asked to receive for that answer_fact.",
                    "Each answer_output describes one requested result output, not one output per result instance.",
                    "Set answer_output.role whenever the requested output matches one of these roles.",
                    "Use ROW_POPULATION for a count/cardinality output over the requested subject instances, such as sales count or number of orders.",
                    "Use MEASURED_VALUE for a numeric measured output, such as sales total, average amount, max duration, or payroll total.",
                    "Use ANSWER_VALUE for a direct requested value that is not a row count or measured numeric aggregate.",
                    "Use POPULATION_SCOPE only when the user explicitly asks to return the population or scope itself as an answer output.",
                    "Conversation input provenance clarifies referenced inputs, not answer outputs.",
                    "For list or table questions with multiple requested columns about the same rows or groups, use one answer_request and put each requested column in answer_outputs.",
                ),
            ),
            builder.instruction_block(
                "Question Inputs Overview",
                (
                    "question_inputs is declared before answer_requests.",
                    "question_inputs declares each literal value or resolved row-set reference once.",
                    "Create question_inputs only for values that need grounding, time compilation, or result limiting, such as London, today, or top five.",
                    "Do not create a question_inputs item for answer_subject.subject_text itself.",
                    "Do not create question_inputs for state, type, channel, or lifecycle modifiers that define answer_population membership tests, such as unverified, in-person, open, or canceled.",
                ),
            ),
            builder.instruction_block(
                "Question Input Inventory",
                (
                    "Before finalizing question_inputs, actively inventory every word or phrase that is a reference value, time value, result limit, or resolved row-set reference.",
                    "Declare exactly one question_inputs item for every inventoried phrase.",
                    "Question inputs are atomic value rows. Each question_input represents one value, time, limit, or reference that the answer contract may use. If the question names multiple values, create one question_input per value. Put labels such as \"staff ids\" or \"store\" in field_label_text; do not combine several values into one value_source_text or resolved_value_text.",
                    "Each question_inputs item must include inventory_check.why_this_is_an_input explaining which input category it belongs to and why it constrains an answer request or supplies a value.",
                    "Each question_input must have one primary contract role: population predicate operand, result key, time constraint, result limit, or computation operand.",
                    "Inputs that define result shape or output axes belong to answer_expression.group_key or answer_outputs, not answer_population membership tests.",
                    "Use answer_population membership_tests only for predicates that narrow the subject instances before the requested result is produced.",
                    "question_inputs declares concrete user/context values that those predicates, time predicates, result limits, or requested computations depend on and that downstream stages must ground, compile, verify, or bind.",
                    "Do not use answer_fact, population_label, or membership-test prose as the only carrier for a concrete value that affects the requested fact.",
                    "When one input constrains multiple requested facts, declare it once in question_inputs and include its input_ref in used_question_inputs on each applicable answer_request.",
                    "After declaring question_inputs and used_question_inputs, set question_input_inventory_check.all_input_like_phrases_declared=true only when every input-like word or phrase has a question_inputs item.",
                ),
            ),
            builder.instruction_block(
                "Question Input Sources",
                (
                    "Use source=question_context for inputs copied directly from the current question.",
                    "Use source=question_context for conversation input provenance items marked question_input_source=question_context.",
                    "Use source=conversation_resolution only for conversation input provenance items marked question_input_source=conversation_resolution.",
                    "Every value_source_text or reference_text must be copied verbatim from the current question, conversation input provenance, or conversation resolution annotations.",
                ),
            ),
            builder.instruction_block(
                "Conversation Resolution Inputs",
                (
                    "When conversation input provenance includes kind=row_set_reference and that input constrains an answer_request, copy it as a row_set_reference input with the same value_source_text and input_ref as resolved_input_ref.",
                    "When conversation input provenance includes kind=literal_text and that input constrains an answer_request, copy value_source_text, resolved_value_text, role, input_ref as resolved_input_ref, and any field_label_text or value_meaning_hint.",
                    "Only new replacement text marked question_input_source=question_context should be treated as current-turn input evidence.",
                    "Do not convert a row_set_reference into literal_text, and do not convert a literal_text into row_set_reference.",
                ),
            ),
            builder.instruction_block(
                "Literal Reference Inputs",
                (
                    "Use kind=literal_text with role=reference_value for each separately addressable user/context value that identifies, names, keys, codes, or otherwise refers to a concrete entity or business value.",
                    "A reference_value is required when the requested fact depends on that concrete value being grounded or directly verified before compilation.",
                    "Do not use reference_value for answer_subject.subject_text, a generic resource class, answer category, grouping label, pronoun, or question word unless conversation resolution emits that pronoun as a resolved literal_text input.",
                    "If the phrase is the same as answer_subject.subject_text, answer_population.counted_unit, or answer_population.population_label, it is not a reference_value.",
                    "Use one literal_text reference_value item per separately addressable value, even when multiple values appear in one coordinated phrase.",
                    "value_source_text is the verbatim copied question span that supplies the value and may include surrounding qualifier text.",
                    "resolved_value_text is the question-level value after language/context resolution, not a Fervis-verified canonical identity.",
                    "When value_source_text includes a qualifier, punctuation, or grammar that is not part of the value, keep that context out of resolved_value_text.",
                    "For user-supplied names, codes, UUIDs, IDs, or other identifiers, copy the supplied value itself; grounding decides whether it is a verified canonical identity, resolver lookup, direct binding, or clarification.",
                    "When the question or conversation-resolution context gives an attribute-like qualifier for the value, set field_label_text to the closest catalog-blind approximation of that attribute name; omit it only when no such qualifier exists.",
                    "field_label_text helps grounding choose or verify the intended attribute; it is not a catalog field decision.",
                    "value_meaning_hint briefly describes what kind of value this is, such as location, account, or code.",
                    "Do not replace resolved_value_text with a resolver result, API value, synonym, or different business object that was not supplied by the user or conversation context.",
                ),
            ),
            builder.instruction_block(
                "Literal Time Inputs",
                (
                    "Use kind=literal_text with role=time_value for calendar dates, calendar date ranges, relative time, calendar periods, quarters, months, years, rolling windows, and open calendar ranges.",
                    "For each time input, copy only the exact value span into value_source_text from the question context or conversation input provenance.",
                    "Set resolved_value_text to the copied time phrase or resolved conversation text, without compiling it into dates.",
                    "When a time input constrains an answer_request, include its input_ref in that answer_request's used_question_inputs.",
                    "Do not compile date ranges, calendar dates, relative offsets, or time shapes in this turn.",
                    "Use separate time inputs when the user asks for separate dates or periods. Use one range input when the user asks for one combined range.",
                ),
            ),
            builder.instruction_block(
                "Literal Limits",
                (
                    "Use kind=literal_text with role=result_limit only for explicit rank or row limits with a copied number expression, such as Top 5, top five, first 10, or bottom three.",
                    "Set value_source_text to the verbatim copied limit phrase and resolved_value_text to canonical positive integer digits, such as 5, 10, or 3.",
                    "Do not use a question input for ranking words without an explicit number, such as most, highest, least, best, or top item.",
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
        answer_contract_schema = self._answer_request_contract_schema()
        return ProviderResponseContract(
            provider_schema={
                ANSWER_REQUEST_CONTRACT_TOOL_NAME: answer_contract_schema,
                MISSING_INPUT_CLARIFICATION_TOOL_NAME: (
                    build_missing_input_clarification_schema()
                ),
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        answer_contract_schema = self._answer_request_contract_schema()
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=ANSWER_REQUEST_CONTRACT_TOOL_NAME,
                    tool_description="Submit complete catalog-blind answer request contracts.",
                    input_schema=answer_contract_schema,
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

    def _answer_request_contract_schema(self) -> dict[str, object]:
        return build_answer_request_contract_schema(
            include_conversation_resolution_inputs=(
                _has_conversation_resolution_inputs(self.request)
            )
        )


def _has_conversation_resolution_inputs(request: QuestionContractRequest) -> bool:
    return request.conversation_input_provenance.has_conversation_resolution_inputs
