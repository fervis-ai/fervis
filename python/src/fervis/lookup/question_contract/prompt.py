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
    build_question_contract_decisions_schema,
)
from fervis.lookup.question_contract.tools import (
    QUESTION_CONTRACT_TOOL_NAME,
)
from fervis.model_io.structured_output.specs import required_tool_spec


class QuestionContractTurnPrompt(TurnPromptBase):
    turn_name = "question contract"
    turn_task = (
        "author the catalog-blind answer request contract for the factual API question"
    )

    def __init__(self, request: QuestionContractRequest) -> None:
        self.request = request

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        resolution_payload = (
            self.request.conversation_resolution.to_prompt_payload()
            if self.request.conversation_resolution is not None
            else {}
        )
        sections: list[PromptSection] = []
        if resolution_payload:
            sections.append(builder.json_section(
                "Conversation resolution context:",
                resolution_payload,
                indent=2,
            ))
        responses = self.request.clarification_responses
        if responses:
            sections.append(
                builder.json_section(
                    "Attributed clarification responses:",
                    {
                        "responses": [
                            {
                                "response_id": response.source.response_id,
                                "clarification_id": response.source.clarification_id,
                                "exact_user_text": response.source.exact_user_text,
                                "missing_item_id": response.missing_item_id,
                                "expected_value_kind": response.expected_value_kind,
                            }
                            for response in responses
                        ]
                    },
                    indent=2,
                )
            )
        return tuple(sections)

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Decision Scope",
                (
                    "Interpret the factual intent expressed by the current question and its typed conversation-resolution context.",
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
                    "Author the contract for the complete factual intent in the current question.",
                    "The current question preserves the user's demand and discourse structure; conversation-resolution values supply context-dependent meaning.",
                    "When active_clarification is present, interpret its original_question and ordered exchanges together, then author a new question contract from scratch.",
                    "An active clarification supplies question context, not a prior question contract.",
                    "Treat each resolved value as a binding meaning commitment for its current clause.",
                    "Declared resolved question inputs are authoritative; copy them exactly when they constrain an answer request.",
                    "Do not reconstruct additional prior-turn inputs from conversation history.",
                    "Use only the current question and typed conversation-resolution context as question context.",
                    "Return a clarification outcome only when visible context is insufficient to author one complete factual question contract.",
                ),
            ),
            builder.instruction_block(
                "Relational Ownership",
                (
                    "answer_subject: Kind of candidate instance to which answer_expression applies.",
                    "answer_population: Candidate instances qualifying independently, before cross-instance operations.",
                    "answer_expression: The base operation over qualifying candidates, plus any requested ordering and result selection.",
                    "answer_outputs: Values or facts projected from the result.",
                ),
            ),
            builder.instruction_block(
                "Answer Requests",
                (
                    "answer_fact concisely and completely describes the requested factual result, including any user-stated ordering, comparison, or selection.",
                    "answer_expression.family is required and classifies the catalog-blind answer shape, not API execution.",
                    "Choose family for the base result.",
                    "Use list_rows for qualifying rows; scalar_value for one direct value; scalar_aggregate for one aggregate over all qualifying candidates; grouped_aggregate for one aggregate per group; computed_scalar for arithmetic over facts or values; and set_difference, coverage_check, existence_check, and comparison_check for their stated set or comparison operations.",
                    "Ordering and result selection are separate from family.",
                    "Use scalar_aggregate for count answers only when the requested result is one scalar count for the whole requested population, such as how many X, number of X, or count of X.",
                    "If the question asks for counts per group, by group, or for each specified key, use grouped_aggregate.",
                    "For grouped_aggregate, set answer_expression.group_key.",
                    "answer_expression.group_key.description names the result key or grouping dimension, such as region, period, category, or supplied key.",
                    "Use answer_expression.group_key.domain=SPECIFIED_QUESTION_INPUTS when groups are exactly declared question_inputs; otherwise use SOURCE_RESULT_VALUES.",
                    "For a repeated measure over specified inputs, set answer_expression.group_key with domain=SPECIFIED_QUESTION_INPUTS and put one measure/count result column in answer_outputs, not one output per key value.",
                    "Use scalar_value only for one direct requested value, not for row/population counts.",
                    "Choose answer_expression.family from the requested answer shape, not endpoints, fields, APIs, or a single keyword.",
                    "answer_subject is required. It names the kind of candidate instance to which answer_expression applies, not the grammatical subject, a concrete entity restricting those instances, or a property returned through answer_outputs.",
                    "Write answer_subject.subject_text as a concise catalog-blind kind of candidate instance established by the current question or conversation resolution, without modifiers that narrow the requested instances.",
                    "answer_subject.instance_interpretation.kind is required.",
                    "Use NORMAL_BUSINESS_INSTANCE for ordinary business reporting questions over the subject as business users normally understand it.",
                    "Use RAW_DATA_RECORD only when the user explicitly asks for persisted records, rows, logs, audit entries, raw data, database entries, or another data artifact.",
                    "answer_population is required. It defines candidate instances qualifying independently, before cross-instance ordering, comparison, selection, or aggregation.",
                    "answer_population.population_label is a concise phrase for those independently qualifying candidate instances.",
                    "answer_population.counted_unit names one business unit in that population.",
                    "answer_population.membership_tests must include one SUBJECT_IDENTITY test.",
                    "Each membership test asks one predicate about one candidate property. Create separate EXPLICIT_USER_CONSTRAINT tests for different properties, even when they consume the same input; omit predicates already enforced by answer_expression.",
                    "Each concrete value named by an EXPLICIT_USER_CONSTRAINT is its operand and must have its own POPULATION_TESTS question input.",
                    "Within answer_population, question inputs are predicate operands, not separate tests; the number of inputs does not determine the number of tests.",
                    "Within answer_population, when multiple inputs are alternative values for the same predicate, create one membership test for that predicate.",
                    "Multiple MUST_PASS tests mean the candidate must satisfy every test. Create separate MUST_PASS tests only when that conjunction matches the question.",
                    "For NORMAL_BUSINESS_INSTANCE, include a NORMAL_INSTANCE_GUARD test; the backend attaches the standard ORDINARY_BUSINESS_INSTANCE_V1 profile with typed excluded-state roles.",
                    "For RAW_DATA_RECORD, include a RAW_RECORD_GUARD test.",
                    "Each membership test has polarity MUST_PASS unless the user explicitly asks to exclude matching instances, in which case use MUST_FAIL.",
                    "Do not decide which API values, enum options, endpoints, fields, or params pass answer_population tests in this turn.",
                    "answer_requests_count must equal the number of answer_requests.",
                    "Do not put API details, endpoint names, field names, params, enum values, or execution operations in answer_subject.",
                    "Do not include caveats, proof, data availability checks, endpoint/API terms, execution instructions, or underlying calculation support unless the user explicitly asks for that support as an answer part.",
                ),
            ),
            builder.instruction_block(
                "Ordering And Selection",
                (
                    "ordering states what result value determines order and which direction it uses.",
                    "ordering.basis describes that value without naming an API field.",
                    "Use direction=ascending when smaller or earlier values come first.",
                    "Use direction=descending when larger or later values come first.",
                    "For list_rows and grouped_aggregate, selection states which results survive.",
                    "all_results keeps every result; take_one keeps exactly one result; take keeps the explicit number supplied by one result_limit input.",
                    "take_one and take require ordering. all_results may be ordered or unordered.",
                    "Use take_one for a singular first, last, highest, or lowest result. Do not create a result_limit input for take_one.",
                    "Use take only when the question explicitly supplies a positive result count.",
                    "The result_limit input owns that count through RESULT_LIMIT; do not copy its input_ref into answer_expression.",
                ),
            ),
            builder.instruction_block(
                "Answer Outputs",
                (
                    "answer_outputs contain the values or facts projected from the result that the user asked to receive for that answer_fact.",
                    "Each answer_output describes one requested result output, not one output per result instance.",
                    "Set answer_output.role whenever the requested output matches one of these roles.",
                    "Use ROW_COUNT for a count/cardinality output over the requested subject instances, such as sales count or number of orders.",
                    "Use MEASURED_VALUE for a numeric measured output, such as sales total, average amount, max duration, or payroll total.",
                    "Use ANSWER_VALUE for a direct requested value that is not a row count or measured numeric aggregate.",
                    "Use POPULATION_SCOPE only when the user explicitly asks to return the population or scope itself as an answer output.",
                    "Declared resolved inputs clarify referenced inputs, not answer_outputs.",
                    "For list or table questions with multiple requested columns about the same rows or groups, use one answer_request and put each requested column in answer_outputs.",
                ),
            ),
            builder.instruction_block(
                "Question Inputs Overview",
                (
                    "question_inputs is declared before answer_requests.",
                    "question_inputs declares each literal value or resolved row-set reference once.",
                    "Create question_inputs for concrete values supplied by the question or conversation resolution that a population predicate, result key, time constraint, compute expression, or result limit consumes.",
                    "Do not create a question_inputs item for answer_subject.subject_text itself.",
                ),
            ),
            builder.instruction_block(
                "Literal Reference Inputs",
                (
                    "A reference_value is a concrete identity, property, category, status, channel, or other predicate value, such as completed or in-person; create one kind=literal_text with role=reference_value input for each and, when it modifies a candidate kind, copy only the property-value span.",
                    "The qualification test must make sense for one candidate without inspecting any other candidate.",
                    "Values used to compare, order, rank, or select candidates by position belong to answer_expression, not reference_value.",
                    "Question Contract does not decide whether a reference resolves to a canonical entity or a scalar field value; catalog-aware grounding owns that decision.",
                    "A reference_value is required when the requested fact depends on that concrete value being grounded or directly verified before compilation.",
                    "Do not use reference_value for answer_subject.subject_text, a generic resource class, answer category, grouping label, pronoun, or question word unless conversation resolution emits that pronoun as a resolved literal_text input.",
                    "Use one literal_text reference_value item per separately addressable value, even when multiple values appear in one coordinated phrase.",
                    "value_source_text is the smallest verbatim question span that supplies the value; exclude the subject and surrounding grammar.",
                    "operand_text is the question-level operand after language/context resolution, not a Fervis-verified catalog value or canonical identity.",
                    "operand_text contains only the operand. Remove subject words and grammatical material that states how the operand constrains the subject.",
                    "For user-supplied names, codes, UUIDs, IDs, or other identifiers, copy the supplied value itself; grounding decides whether it is a verified canonical identity, resolver lookup, direct binding, or clarification.",
                    "When the question or conversation-resolution context gives an attribute-like qualifier for the value, set field_label_text to the closest catalog-blind approximation of that attribute name; omit it only when no such qualifier exists.",
                    "field_label_text helps grounding choose or verify the intended attribute; it is not a catalog field decision.",
                    "value_meaning_hint briefly describes what kind of value this is, such as location, account, or code.",
                    "Do not replace operand_text with a resolver result, API value, synonym, or different business object that was not supplied by the user or conversation context.",
                ),
            ),
            builder.instruction_block(
                "Literal Time Inputs",
                (
                    "Use kind=literal_text with role=time_value only for values that identify a calendar or clock instant, interval, or relative period; an ordinal position in an ordered result set is not a time value.",
                    "For each time input, copy only the exact value span into value_source_text from the question context or declared resolved inputs.",
                    "Set operand_text to the copied time phrase or resolved conversation text, without compiling it into dates.",
                    "When a time input constrains an answer_request, assign it through POPULATION_TESTS to the applicable time membership test.",
                    "Do not compile date ranges, calendar dates, relative offsets, or time shapes in this turn.",
                    "Use separate time inputs when the user asks for separate dates or periods. Use one range input when the user asks for one combined range.",
                ),
            ),
            builder.instruction_block(
                "Formula Values",
                (
                    "Use formula_value when a supplied literal is an arithmetic operand used to compute the returned answer, such as 10% in '10% of the total measured value.'",
                    "A formula_value is not a population predicate and does not determine which candidates qualify.",
                ),
            ),
            builder.instruction_block(
                "Result Limits",
                (
                    "A result_limit is an explicit positive integer stating how many ordered results to return.",
                    "The positive integer in 'which N', 'top N', or 'first N' is a result_limit.",
                    "Use kind=literal_text with role=result_limit for every supplied result_limit; it is a question input, not merely answer-shape wording.",
                    "Set operand_text to canonical positive integer digits for that copied integer.",
                    "Assign a result_limit through one RESULT_LIMIT question_input_uses record on the answer_request.",
                    "Do not infer a result limit from singular or plural grammar, ordering, or superlative language.",
                ),
            ),
            builder.instruction_block(
                "Question Input Inventory",
                (
                    "Before finalizing question_inputs, actively inventory every word or phrase that is a reference value, time value, formula value, result limit, or resolved row-set reference.",
                    "Declare exactly one question_inputs item for every inventoried phrase.",
                    "Question inputs are atomic value rows. Each question_input represents one value, time, formula operand, limit, or reference that the answer contract may use. If the question names multiple values, create one question_input per value. Put the input's semantic role in field_label_text; do not combine several values into one value_source_text or operand_text.",
                    "Question-input identity comes from the copied occurrence, not from the predicate or field that consumes it. One copied occurrence remains one input when several predicates consume it.",
                    "Each question_inputs item must include inventory_check.why_this_is_an_input explaining which input category it belongs to and why it constrains an answer request or supplies a value.",
                    "Each question_input must have one primary contract role: population predicate operand, result key, time constraint, compute-expression operand, or result limit.",
                    "Result-shape and result-axis inputs belong to answer_expression, not answer_population membership tests.",
                    "Use answer_population membership_tests only for predicates that narrow subject instances independently of answer_expression's result axis.",
                    "question_inputs declares concrete user/context values that those predicates, time predicates, or result limits depend on and that downstream stages must ground, compile, verify, or bind.",
                    "Do not use answer_fact, population_label, or membership-test prose as the only carrier for a concrete value that affects the requested fact.",
                    "When one input constrains multiple requested facts, declare it once in question_inputs; each applicable answer_request gives it one fact-local question_input_uses record.",
                    "Set question_input_inventory_check.all_input_like_phrases_declared=true only when every input-like word or phrase has a question_inputs item.",
                ),
            ),
            builder.instruction_block(
                "Clarification Boundary",
                (
                    "Write decision_basis first. First state whether the current wording identifies a requested fact and whether any required referent can only be identified from an earlier utterance.",
                    "Then list every reference_value, time_value, formula_value, and result_limit the question contains and outcome must declare without assigning owners or predicates.",
                    "Relational structure belongs in outcome; do not assess grounding, time compilation, or execution in decision_basis.",
                    "Do not use a clarification outcome when visible context is sufficient to author a complete factual question contract.",
                    "Use kind=missing_requested_fact only when explicit wording states no business fact, property, measure, relationship, comparison, or row set to return.",
                    "Use missing_requested_fact only when no answer_fact can be authored from explicit question wording; an unresolved subject or input does not erase a stated answer_fact.",
                    "If a required referent can only be identified from an earlier utterance and typed conversation resolution does not supply it, return kind=unresolved_prior_turn_references instead of a question_contract.",
                    "A named property requested for a subject is a complete answer definition; its unknown value is the requested answer, not missing context.",
                    "An explicitly named factual measure or business result is a sufficient answer definition; do not request a narrower metric merely because several API fields or calculations might later implement it.",
                    "An explicit name, code, key, date, number, or other value is sufficient to author a question input; grounding determines whether it exists or resolves uniquely.",
                    "A self-contained relative time expression is an explicit time value, not an unresolved conversation reference.",
                    "For each unresolved prior-turn reference, copy source_text verbatim from the current question or visible context and set target_label to a concise catalog-blind category without copying or paraphrasing source_text.",
                    "For missing_requested_fact, copy the incomplete request text into source_text.",
                    "Set why_question_is_incomplete to the specific information needed to form a factual request.",
                ),
            ),
            builder.instruction_block(
                "Question Input Ownership",
                (
                    "Within each answer_request, author answer_expression, then question_input_uses, then answer_population.",
                    "question_input_uses assigns each input used by this answer_request to exactly one semantic owner kind.",
                    "Create exactly one question_input_uses record for each fact-local input_ref. When several population tests consume that input, they reuse its one use_id; do not create one use record per test.",
                    "Use GROUP_KEY when the input defines a requested result group under SPECIFIED_QUESTION_INPUTS.",
                    "A SPECIFIED_QUESTION_INPUTS group key restricts results to its GROUP_KEY inputs and groups by them.",
                    "GROUP_KEY inputs are not candidate-row predicates; create no EXPLICIT_USER_CONSTRAINT for them.",
                    "Use POPULATION_TESTS when the input is an operand of one or more EXPLICIT_USER_CONSTRAINT membership tests, and give that use a unique use_id.",
                    "Use COMPUTE_EXPRESSION when a formula_value is an operand in the requested computed_scalar expression.",
                    "Use RESULT_LIMIT when the input supplies answer_expression's requested result limit.",
                    "One population test may reference several POPULATION_TESTS use_ids.",
                    "Build EXPLICIT_USER_CONSTRAINT tests only from prior POPULATION_TESTS uses; an input without use_id cannot be a question_input_use_ref.",
                    "Each EXPLICIT_USER_CONSTRAINT lists the use_id of every POPULATION_TESTS operand it consumes in question_input_use_refs.",
                    "question_input_use_refs may name only POPULATION_TESTS uses. GROUP_KEY, COMPUTE_EXPRESSION, and RESULT_LIMIT inputs are never membership-test operands.",
                    "SUBJECT_IDENTITY asks only whether the candidate is an instance of answer_subject and has no input operands.",
                    "An input identifying a related entity uses POPULATION_TESTS unless it is a SPECIFIED_QUESTION_INPUTS group member, in which case it uses GROUP_KEY only.",
                    "SUBJECT_IDENTITY, NORMAL_INSTANCE_GUARD, and RAW_RECORD_GUARD use an empty question_input_use_refs array.",
                ),
            ),
            builder.instruction_block(
                "Question Input Sources",
                (
                    "Use source=question_context for inputs copied directly from the current question.",
                    "Use source=conversation_resolution only for declared resolved question inputs.",
                    "Every value_source_text or reference_text must be copied verbatim from the current question or declared resolved inputs.",
                ),
            ),
            builder.instruction_block(
                "Conversation Resolution Inputs",
                (
                    "Conversation resolution has already classified each declared input. "
                    "There is no input-kind or role decision in this turn.",
                    "For every resolved input used by an answer_request, copy its "
                    "declared kind, role, value text, resolved operand, and input_ref "
                    "exactly into the corresponding question_input fields.",
                    "When a declared resolved input has kind=row_set_reference and constrains an answer_request, copy it as a row_set_reference input with the same value_source_text and input_ref as resolved_input_ref.",
                    "When a declared resolved input has kind=literal_text and constrains an answer_request, copy value_source_text, its resolved value as operand_text, role, input_ref as resolved_input_ref, and any field_label_text or value_meaning_hint.",
                ),
            ),
            builder.instruction_block(
                "Retained Prior Shape",
                (
                    "retained_frame_parts are fixed prior question meanings that "
                    "conversation resolution selected for this clause.",
                    "Use their typed kind and answer_shape together with the raw current "
                    "question. Explicit current meaning remains authoritative; text in "
                    "a retained part does not restore a subject or grouping that the "
                    "current question replaced.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Before returning a complete contract, verify that each copied question occurrence is declared once, every fact-local input appears once in question_input_uses, group-key and compute-expression inputs appear in no population test, and every EXPLICIT_USER_CONSTRAINT references its POPULATION_TESTS operands.",
                    "Return exactly one provider-native tool call.",
                    "Set kind=question_contract when visible context is sufficient to author complete answer requests.",
                    "Set kind=missing_requested_fact only when no complete factual result is identifiable.",
                    "Set kind=unresolved_prior_turn_references only when a complete factual result is identifiable but required prior-turn references remain unresolved.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                QUESTION_CONTRACT_TOOL_NAME: self._question_contract_outcome_schema(),
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=QUESTION_CONTRACT_TOOL_NAME,
                    tool_description=(
                        "Submit the catalog-blind question-contract outcome."
                    ),
                    input_schema=self._question_contract_outcome_schema(),
                ),
            )
        )

    def _question_contract_outcome_schema(self) -> dict[str, object]:
        return build_question_contract_decisions_schema(
            conversation_inputs=(
                self.request.conversation_resolution.inputs
                if self.request.conversation_resolution is not None
                else ()
            ),
        )
