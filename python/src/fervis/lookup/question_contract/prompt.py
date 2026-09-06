"""Normative model-facing semantics for the relational Question Contract."""

from __future__ import annotations

from fervis.lookup.question_contract.request import QuestionContractRequest
from fervis.lookup.question_contract.schema import (
    build_semantic_question_contract_schema_for_meaning,
    build_semantic_question_frame_schema,
)
from fervis.lookup.question_contract.parser import (
    ParsedSemanticQuestionMeaning,
)
from fervis.lookup.semantic_types import (
    CollectionType,
    SourceOrigin,
    ValueType,
    value_type_kind,
)
from fervis.lookup.question_contract.tools import QUESTION_CONTRACT_TOOL_NAME
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.context import TurnPromptContext
from fervis.model_io.structured_output.specs import required_tool_spec


SEMANTIC_QUESTION_CONTRACT_INSTRUCTIONS = """\
Definitions

A set is one question-local kind of candidate instance or occurrence.

A fact is one value observed for a set instance or through a declared
association. Its meaning names that observed value, rather than the row or
relationship containing it. The expression using the fact states the capability
required of its future source field: sum and average require summable values;
minimum and maximum require orderable values; within requires temporal values;
Boolean conditions require Boolean values. Source Binding later verifies that
the selected catalog field provides that capability.

An identifier fact means which exact instance of a declared set is involved.
It is semantic identity even when the supplied input is a human-readable name;
a value shared by many instances is a property fact.

An association is the question-local relation of occurrences connecting
instances of two sets. Its cardinality is unspecified. It is independent of
physical sources, fields, and joins.

An input_ref names one fixed supplied value shown with the question meaning.
Expressions use that ref wherever the supplied value participates.

When a supplied time period limits which occurrences qualify, declare a Date
or DateTime fact observed for the qualifying set and compare that fact with the
TemporalScope input using within.

Authoring order

Write one answer_requests item for every shown requested_fact_ref. The shown
result kind fixes its result grain. The shown grouping, ordering, and output
meanings fix their exact counts, order, and origins. Within each item, write
candidate_set and set_graph before grouping.

Requested-fact boundaries

Each shown requested_fact_ref is one fixed answer-request boundary.

Semantic structure

Use s1 for the shown candidate set. In set_graph, each identity input first
chooses null when unused by this request or declares its related association
and entity set. Each related output declares its association and entity set.
other_related_sets contains only relationships owned by neither fixed role.

Each identity_input_relations and requested_output_relations entry starts at
s1. At the top level of other_related_sets, the parent set is s1. Each
related_sets item
declares a relationship from that parent set to the item's set. Inside an item's
nested related_sets, the parent is that item's set; each nested relationship
starts there, never at an earlier ancestor. Every graph set writes its id,
instance_kind, and origin. A related-identity grouping copies the referenced
graph set's id as set_ref.

Declare each non-candidate set and each association exactly once in set_graph.
A related-identity grouping references its graph set by set_ref. The graph alone
declares the set and the association that reaches it. Facts, qualifications,
ordering expressions, and outputs reference declared graph terms. In a fact,
observed_for_ref names the set or association where the value is observed.
Set-valued expressions use set_ref.
Structurally identical fact leaves denote one fact.

A relationship used to assign qualifying rows to a shown grouping is fully
represented by set_graph and grouping. It contributes no separate qualification
or aggregate filter.

The fixed candidate set is the set whose instances are tested by qualification
and then grouped or aggregated. A requested group label is separate from it
when qualifying instances are aggregated by that label.

NORMAL_BUSINESS_INSTANCE means the candidate instances as business users
normally understand them. RAW_DATA_RECORD means persisted records, rows, logs,
audit entries, raw data, database entries, or another explicitly requested data
artifact. Use RAW_DATA_RECORD only when the question explicitly requests that
data artifact.

A value defined over a supplied time scope is derived from observations in
that scope. The observations are qualifying rows, the supplied period is an
input, and the scoped aggregate is a result expression.

When returned entities are compared by an aggregate over related occurrences,
use the occurrences as the qualifying set, connect the grouped entity at its
actual depth in set_graph, reference that entity's graph set from the
related_instance_identity grouping, and aggregate an occurrence-owned fact.

Reference candidate_set when the result returns its instances. When a
result groups exact candidate instances, use candidate_instance_identity. When
it groups exact related instances, use related_instance_identity with set_ref
naming the entity set declared by set_graph. The
grouped output references the grouping entry by its group_ref. A set may also
be the argument of count; it is the row-domain reference used by count.

Qualification is exactly the user-stated true-or-false conditions that decide
whether a candidate qualifies. Identity inputs filling one role as alternatives
combine with OR. Identity inputs filling different simultaneous roles combine
with AND. instance_kind is a question-local nominal entity type. Equal
instance_kind values mean the same entity type; different values mean different
entity types. A fixed entity reference is represented by its input_ref. Compare
that input_ref with an Identifier fact of the same instance_kind. Each identity
input_comparison copies its input_ref, operand_meaning, and instance_kind. When
it uses related_instance, write association_ref naming the declared relationship
from the current row to the identified entity. candidate_instance identifies
the current row itself. In a quantifier condition the current row is the
quantified set, so the relationship must start at that set. One supplied identity
can be compared through different relationships at different use sites; each
comparison names its own relationship. Declare additional relationships in
other_related_sets as needed. Each related entity output copies its fixed
instance_kind and references a graph set of that same type.
An aggregate's filter applies only to
that aggregate. A Boolean output contains the requested Boolean expression.
value_comparison compares two observed or computed values.

A quantifier traverses declared associations from the current row to the
related row. Use forall for every related row. Use coverage when every
candidate must have an observation for every member of a separate
required-dimension set.

related_row means that one row from set_ref is connected to the current row
through every association declared on that set's graph node. condition is
evaluated once for that related row, or is null when those relationships alone
are the condition.

Coverage asks whether, for one candidate, every qualifying required member has
at least one qualifying observation. candidate_condition applies to the
candidate row, or is null when every candidate is eligible.
required_member_condition applies to one required-member row, or is null when
the whole required-member set applies. observation declares its set_ref and
condition together; its condition applies to one observation row. The
association graph supplies one unambiguous path from the candidate and one
from the required member to the observation.

aggregate summarizes all rows retained by qualification. filtered_aggregate
adds one condition that applies only to that aggregate's argument rows.
count returns the number of qualifying argument rows. sum returns the total of
an observed value. distinct_argument states whether repeated argument values
count once; neither aggregate-local choice changes the outer candidate
population.

Each ordering entry writes ordering_basis before expression. ordering_basis
states exactly what value is compared across result rows and whether it is
obtained by counting rows or aggregating an observed value. Then write the
expression that computes that value.

A requested proportion, ratio, share, or percentage is divide(part, whole).
When part and whole summarize rows, give each its own
aggregate and filter only the part.

An aggregate summarizes qualifying-set rows. With grouping, it produces
one value per grouping tuple; without grouping, it produces one value for the
whole qualifying population.

A Percentage input is normalized to its ratio value before arithmetic. Use
the supplied Percentage directly as the arithmetic operand.

Each grouping entry writes grouping_basis before kind. grouping_basis states
what value one qualifying row contributes to the group. Use
candidate_instance_identity for shown qualifying_row_identity,
related_instance_identity for shown related_entity_identity, and value for
shown non_identity_value. A value grouping writes the expression fixed by the
shown grouping_value: observed_value writes the recorded fact directly, computed_value writes per-row arithmetic, condition writes a Boolean condition, and temporal_bucket
writes a temporal_bucket over a fact and copies the shown grain. Grouped
outputs and ordering use group_ref for grouping values; aggregate expressions
compute values over each group. Grouping assigns a key to every qualifying row.
Qualification selects the requested keys. When supplied identity inputs name
requested groups, qualification compares the grouping identity fact with those
input refs, using OR for alternatives. The per-row identity remains the grouping
value.

Each result_key_meaning identifies a result row. Write its expression in
result_key_outputs. A grouped result key returns its group_ref; an ungrouped
result key returns a set_ref or an identifier fact with identity_path.

Each requested_value_meaning is another result the answer must state. Copy its
value_ref to output_ref in requested_value_outputs. For value_kind=value, write
its value expression. For value_kind=related_entity, the output writes
output_ref; its fixed requested_output_relations entry owns the returned entity
set.
Preserve the shown order within each output list. Ordering names the
value that determines order. Selection states which ordered rows survive.

Express a highest, lowest, first, or last result through ordering and selection.
Qualification retains its population meaning. FirstRankWithTies keeps every row
tied at the first ordered value.

distinct_by is empty unless the question explicitly requests distinct result
rows. When present, it equals the complete output-reference tuple. Grouping,
ranking, and ordinary result identity remain in their dedicated structures.

Inputs

Every shown input_ref is used wherever its supplied value changes
qualification, grouping, computation, ordering, or selection. The shown
operand, type, meaning, and denotation remain fixed.

Authority boundary

This contract contains semantic sets, associations, facts, inputs, and
expressions. Preserve the requested meaning independently of available APIs,
resources, endpoints, fields, parameters, tables, resolver routes, or
executable operations.
"""


SEMANTIC_QUESTION_FRAME_TOOL_NAME = "submit_question_frame"

_MISSING_REQUESTED_FACT_INSTRUCTION = (
    "Return kind=missing_requested_fact when the requested factual result "
    "itself is not identifiable. The kind of thing counted or returned is stated "
    "by the question or supplied by prior context. A population or set named by "
    "its business role is identifiable; its member rows are retrieved as data."
)
_UNRESOLVED_PRIOR_REFERENCE_INSTRUCTION = (
    "Return kind=unresolved_prior_turn_references when the factual result is "
    "identifiable and a required person, object, time, or value is expressed "
    "only by a pronoun or dependent phrase whose antecedent is absent."
)


SEMANTIC_QUESTION_FRAME_INSTRUCTIONS = """\
Authoring order

Write decision_basis first. For a complete question, write every answer_request,
then supplied_values.

decision_basis inventories the independent requested results and the concrete
supplied operands that constrain or compute them. Candidate-population nouns
belong to requested meaning. Concrete names, codes, identifiers, times, and
property values that restrict those candidates belong to supplied_values.

Answer requests

Write one answer_request for each independent factual result. Values that the
question asks the answer to state and that describe the same result row or group
under one ordering and selection are columns of that one answer_request.

A row or group request writes return_request_basis first. It states exactly
what the answer must state, independently of relational mechanics. The result
branch later writes projection from that basis. For one-per-candidate results,
projection writes projection_basis, then candidate_identity=returned when the
answer identifies each candidate row or candidate_identity=omitted when it
states only requested related or property values. For grouped results, write
returned_grouping_keys=all. explicitly_requested_values contains only answer
values besides those returned grouping keys. Ordering
and selection do not make a value part of the answer. Candidate identity is
part of the answer when the question asks which candidates or pairs requested
values with them. Candidate rows used only to produce one requested related or
property value per row omit candidate identity. A result key is the candidate
identity for one row per qualifying candidate, or the grouping tuple for one
row per group. grouping_kind states whether each group key is an entity identity
or a non-identity value. Each explicitly_requested_values item writes value_ref,
then value_kind_basis, value_kind, meaning, and origin once. value_kind_basis
states whether the requested answer states a related entity or a value.
related_entity means the answer states which related person, organization,
place, product, or other entity is involved. value means the answer states an
attribute, measurement, status, time, quantity, or computed value without
identifying an entity. A population scalar retains
return_request_basis and its one returned meaning. Then write
relational_shape_basis, then request. Inside request write relational_shape,
result_grain_basis, then result. Each result branch owns its row source. Inside
result, result_order writes ordering_request_basis, ordering, then selection.
ordering_request_basis states whether the question asks to arrange or rank the
result rows. ordering is no_ordering_requested when it does not, or ordered_by
with the values that determine the requested order. When an existing grouping
key determines order, use group_ref and copy that grouping's group_ref.
supplied_values later
declares each concrete non-selection operand once.

population_rows names rows aggregated into one scalar. result_candidates names
the instances that identify one-per-candidate result rows.
grouped_observation_rows names occurrences aggregated into groups. Each field's
instance_kind is unqualified. Concrete identities, properties, and times that
restrict those rows are supplied values.

result_grain_basis states what one result row represents after qualification
and grouping. Then result selects one closed grain branch. Use
one_value_for_population for one value over all qualifying rows,
one_result_per_qualifying_row for one result per qualifying row, and
one_result_per_group with grouping_meanings for one result per grouping tuple.

A request for the first, last, or top N qualifying occurrences uses
one_result_per_qualifying_row. When entities are ranked by an aggregate over
related occurrences, grouped_observation_rows names those occurrences and
result is one_result_per_group with the entity as a grouping meaning.

ordering lists meanings compared to arrange candidate or group rows before
selection, in priority order. An ordering meaning varies across those rows; a
shared time scope constrains them through supplied_values. Write
ownership_basis stating whether the ordering meaning is one of projection's
returned values. When it is, use requested_value_ref and copy value_ref.
unreturned_ordering_meaning declares a meaning absent from projection.
Ranking words such as first, last, highest, lowest, and top N select result
rows.
“Which A has the greatest B?” returns the group or candidate identity and uses
an unreturned ordering meaning for B. “Which A has the greatest B, and what is
B?” returns the identity and B, then orders by B's value_ref.

For result kind one_value_for_population, returned_meanings contains exactly
the one unknown value requested by the question and ordering is
no_ordering_requested.
Relationships used to qualify rows retain one_result_per_qualifying_row.

For result_rows kind one_result_per_group, each grouping_meaning names one
value that varies across result rows and defines one grouping dimension. It
writes grouping_basis, meaning, origin, then grouping_kind. A
non_identity_value grouping then writes grouping_value. Use observed_value for a recorded grouping value. Use computed_value for a per-row arithmetic grouping value. Use condition for a Boolean grouping condition. Use temporal_bucket with day, week,
month, quarter, or year when rows are grouped into calendar periods.
related_entity_identity identifies another entity related to each qualifying
row, and non_identity_value identifies no entity. A result per qualifying row
uses one_result_per_qualifying_row instead of grouping by that row's identity.
A grouping dimension is a value carried by each qualifying row and
identifies one requested result group. Shared qualifications and time scopes
constrain the rows. An ordinal such as first two belongs to selection over
ordered candidate rows.

selection is all_results when every qualifying result is requested,
first_rank_with_ties for a singular first, last, highest, or lowest request,
and take_with_boundary_ties for an explicit positive number of ordered results.
supplied_values.selection_limits declares that positive integer once and writes
the one-based answer_request_number of the request it limits.

relational_shape is ordinary for ordinary qualification,
every_related_row when every related row must satisfy a condition,
every_required_member_has_observation when each candidate is tested for whether every member of
one required set has a matching observation from a different set. For this
shape, result.coverage_candidates.instance_kind is the kind being tested and
returned; required members and observations are later relational sets.
same_related_row when one related row must participate in two or more stated
relationships to the candidate.

Each requested scalar value is one answer request. Sharing a candidate set or
time scope does not merge scalar values. A requested row or group result is one
answer request and may contain several columns or selected rows at that result
grain. One request owns the returned meanings, ordering, and selection for a
bounded subset. A repeated measure over a specified key set is one grouped
requested fact, not one fact per key.

Supplied values

supplied_values contains only concrete operand values already given by the
question. Candidate kinds, requested unknowns, grouping meanings, and observed
or computed ordering meanings remain in answer_requests. A selection limit is
declared only in supplied_values.selection_limits.

A business subject or requested unknown belongs to the answer request.
Entity references and values used by a condition or computation each belong to
one supplied_values.operands item. Result counts belong to selection_limits.

A business modifier may define the result row source when it names the
business population, or a supplied scalar when it is independently compared.
That meaning has one owner.

A supplied value is a concrete operand already provided by the question for
a comparison, arithmetic, time scope, or supplied collection. Qualifying rows
and required sets are owned by the result row-source field or relational_shape.
A bounded result count is declared once in selection_limits for its answer
request. Requested unknowns are owned by the answer request.

One supplied value item owns one independent operand role. Alternatives filling
the same role share one item. A value shared by several answer requests appears
once.

After supplied_values, set
question_input_inventory_check.all_input_like_phrases_declared=true only when
every condition or computation operand has exactly one supplied_values.operands item and
every bounded result count has exactly one selection_limits item.

supplied_values.operands contains one item for each independent supplied operand
role. Each item writes meaning and denotation_basis, then chooses exactly one
closed branch.

entity_reference is a supplied name, code, or identifier that denotes a
person, organization, place, product, or other entity whose exact identity the
question qualifies or returns. A name, code, or identifier used to locate that
entity through a name, code, or identifier field remains an entity reference
and requires canonical identity resolution or validation. Write instance_kind, then value. value is
single_identity with one identity_value, or identity_alternatives with distinct
identity_values that fill the same role. For conversation_resolution origin,
identity_value copies the shown resolved_value_text; resolved_input_ref is copied
only into origin.

non_entity_value is a supplied category, status, time, quantity, Boolean,
duration, or shared classification. It describes qualifying rows without
naming, coding, or identifying an entity. Write kind, then value. kind is
categorical_value for a category, status, or shared classification;
temporal_scope for a date, time, interval, or relative period; number for a
numeric operand; boolean for true or false; or duration for an elapsed amount
with a unit. Each value contains only the copied operands and their origin.
"""


class SemanticQuestionFrameTurnPrompt(TurnPromptBase):
    turn_name = "question frame"
    turn_task = "author requested results and supplied values"

    def __init__(self, request: QuestionContractRequest) -> None:
        self.request = request

    def system_prompt(self, context: TurnPromptContext) -> str:
        return _semantic_question_system_prompt(context)

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return _conversation_resolution_sections(self.request, builder=builder)

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.text_section(
                "Question frame rules:", SEMANTIC_QUESTION_FRAME_INSTRUCTIONS
            ),
            builder.instruction_block(
                "Outcome",
                (
                    "Return kind=question_meaning when the visible context specifies a complete factual request.",
                    _MISSING_REQUESTED_FACT_INSTRUCTION,
                    _UNRESOLVED_PRIOR_REFERENCE_INSTRUCTION,
                    "Return exactly one provider-native tool call.",
                ),
            ),
            builder.instruction_block('Grouping ownership', (
                'A restriction shared by all groups remains a qualification unless the question also explicitly requests it as a grouping dimension.',
                'Comparison operators and arithmetic operations are structural relations, not supplied text operands. Copy their operand values only; a word is an operand when the question uses it as data.',
            )),
            builder.instruction_block('Temporal operands', (
                'A temporal_scope operand is one complete interval expression, including both boundaries when supplied. Copy the whole interval as one operand; its endpoints are not alternative values. Separate temporal scopes have separate supplied-value items.',
            )),
            builder.instruction_block('Quantified relationship', (
                'every_required_member_has_observation requires two different related row sets: every member of an independently required set must have at least one matching row from the observation set.',
                'An absence condition asks whether matching observations do not exist. It uses ordinary relational shape even when the search domain includes all stores or locations.',
                'every_related_row tests a property directly on each existing related row. An amount or another field on that row is not a separate observation set. Determine the logical requirement rather than treating a broad search domain as positive coverage.',
            )),
            builder.instruction_block('Ordinal selection', (
                'Use position_with_ties when the question requests one explicit ordered position, such as second, third, or position five. It retains only rows tied at that one-based position, excluding rows before that boundary.',
                'Declare that positive integer position once in supplied_values.selection_limits, with the answer_request_number it belongs to. The selection kind distinguishes an ordinal position from a requested number of results.',
                'Use take_with_boundary_ties for the first specified number of ordered rows, and first_rank_with_ties for a highest or lowest result without another explicit position.',
            )),
            builder.instruction_block('Group result grain', (
                'A grouped result returns its grouping keys and aggregate values. An individual related entity is a grouping key or a row-level result, not an aggregate value.',
                'When a question lists individual entities with their related entities or attributes, retain one result per qualifying row. Organizing a list by a related entity does not require aggregating away the listed entities.',
            )),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                SEMANTIC_QUESTION_FRAME_TOOL_NAME: (
                    build_semantic_question_frame_schema(
                        conversation_input_refs=self._conversation_input_refs()
                    )
                )
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        schema = build_semantic_question_frame_schema(
            conversation_input_refs=self._conversation_input_refs()
        )
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SEMANTIC_QUESTION_FRAME_TOOL_NAME,
                    tool_description=(
                        "Submit the requested-result frame and supplied values."
                    ),
                    input_schema=schema,
                ),
            )
        )

    def _conversation_input_refs(self) -> tuple[str, ...]:
        return _conversation_input_refs(self.request)


class SemanticQuestionContractTurnPrompt(TurnPromptBase):
    """Author the relational contract from the canonical question frame."""

    turn_name = "question contract"
    turn_task = "author the semantic relational contract"

    def __init__(
        self,
        request: QuestionContractRequest,
        *,
        meaning: ParsedSemanticQuestionMeaning,
    ) -> None:
        self.request = request
        self.meaning = meaning

    def system_prompt(self, context: TurnPromptContext) -> str:
        return _semantic_question_system_prompt(context)

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            *_conversation_resolution_sections(self.request, builder=builder),
            builder.json_section(
                "Fixed question meaning:",
                _question_meaning_payload(self.meaning),
                indent=2,
            ),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        origin_rules = [
            "For every semantic origin, meaning states the local semantic meaning and source cites where that meaning came from.",
            "Use source=question_context with resolved_input_ref=null for current-question meaning.",
        ]
        if self._conversation_input_refs():
            origin_rules.append(
                "Use source=conversation_resolution with its shown resolved_input_ref for resolved context text."
            )
        return (
            builder.instruction_block(
                "Origins",
                tuple(origin_rules),
            ),
            builder.text_section(
                "Question Contract rules:",
                SEMANTIC_QUESTION_CONTRACT_INSTRUCTIONS,
            ),
            builder.instruction_block(
                "Outcome",
                (
                    "Return kind=question_contract when the visible context specifies a complete factual request.",
                    _MISSING_REQUESTED_FACT_INSTRUCTION,
                    _UNRESOLVED_PRIOR_REFERENCE_INSTRUCTION,
                    "Return exactly one provider-native tool call.",
                ),
            ),
            builder.instruction_block('Operand ownership', (
                'An input_comparison is fact OPERATOR input, in that order. The operator states how the observed fact compares with the supplied input.',
                'Write the requested arithmetic directly. Do not add neutral, cancelling, or repeated operations. A constant uses a supplied input_ref; it is never an observed fact.',
            )),
            builder.instruction_block('Input references', (
                'Every supplied input must be referenced by an input_ref in the executable semantic structure. Repeating its meaning in instance_kind or origin text does not use that input.',
                'When a supplied value identifies a class, category, or state of the candidate population, express that restriction as a qualification using the supplied input_ref. Retain it even when the population description already mentions the same class.',
            )),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={QUESTION_CONTRACT_TOOL_NAME: self._schema()}
        )

    def tool_contract(self) -> ProviderToolContract:
        schema = self._schema()
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=QUESTION_CONTRACT_TOOL_NAME,
                    tool_description="Submit the semantic relational Question Contract outcome.",
                    input_schema=schema,
                ),
            )
        )

    def _conversation_input_refs(self) -> tuple[str, ...]:
        return _conversation_input_refs(self.request)

    def _schema(self) -> dict[str, object]:
        return build_semantic_question_contract_schema_for_meaning(
            self.meaning,
            conversation_input_refs=self._conversation_input_refs(),
        )


def _semantic_question_system_prompt(context: TurnPromptContext) -> str:
    organization_name = context.host.organization_name.strip()
    organization_context = f" for {organization_name}" if organization_name else ""
    return (
        "You are authoring the semantic meaning of a factual "
        f"business or operational question{organization_context}. The question "
        "and declared conversation resolution are the complete semantic authority. "
        "The result is independent of APIs, storage, fields, sources, and executable "
        "operations."
    )


def _question_meaning_payload(
    meaning: ParsedSemanticQuestionMeaning,
) -> dict[str, object]:
    denotation_by_input_ref = {
        item.input_ref: item for item in meaning.input_denotations
    }
    return {
        "answer_requests": [
            {
                "requested_fact_ref": item.requested_fact_id,
                "return_request_basis": item.return_request_basis,
                "relational_shape_basis": item.relational_shape_basis,
                "result_grain_basis": item.result_grain_basis,
                "ordering_request_basis": item.ordering_request_basis,
                "result_kind": item.result_kind,
                "candidate_kind": _origin_payload(item.candidate_set_origin),
                "grouping_meanings": [
                    {
                        "group_ref": group_ref,
                        **_origin_payload(origin),
                        "grouping_kind": grouping_kind,
                        **(
                            {"grouping_value": _grouping_value_payload(value_shape)}
                            if value_shape is not None
                            else {}
                        ),
                    }
                    for group_ref, origin, grouping_kind, value_shape in zip(
                        item.grouping_refs,
                        item.grouping_origins,
                        item.grouping_kinds,
                        item.grouping_value_shapes,
                        strict=True,
                    )
                ],
                "ordering_meanings": [
                    {
                        **_origin_payload(origin),
                        **(
                            {"group_ref": group_ref}
                            if group_ref is not None
                            else {}
                        ),
                        **({"value_ref": value_ref} if value_ref is not None else {}),
                    }
                    for origin, group_ref, value_ref in zip(
                        item.ordering_origins,
                        item.ordering_group_refs,
                        item.ordering_value_refs,
                        strict=True,
                    )
                ],
                "result_key_meanings": [
                    {
                        **_origin_payload(origin),
                        "key_kind": output_kind,
                    }
                    for origin, output_kind in zip(
                        item.output_origins[: item.result_key_count],
                        item.output_kinds[: item.result_key_count],
                        strict=True,
                    )
                ],
                "requested_value_meanings": [
                    {
                        "value_ref": value_ref,
                        "value_kind": output_kind,
                        **_origin_payload(origin),
                    }
                    for value_ref, origin, output_kind in zip(
                        item.requested_value_refs,
                        item.output_origins[item.result_key_count :],
                        item.output_kinds[item.result_key_count :],
                        strict=True,
                    )
                ],
                "selection": {
                    "kind": item.selection_kind,
                    "limit_input_ref": item.selection_limit_input_ref,
                },
                "relational_shape": item.relational_shape,
            }
            for item in meaning.answer_requests
        ],
        "inputs": [
            {
                "input_ref": item.id,
                "operand": item.operand,
                "value_type": _value_type_label(item.value_type),
                "origin": _origin_payload(item.origin),
                "operand_meaning": denotation_by_input_ref[item.id].operand_meaning,
                "denotation_basis": denotation_by_input_ref[item.id].denotation_basis,
                "denoted_instance_kind": denotation_by_input_ref[
                    item.id
                ].denoted_instance_kind,
                "input_kind": denotation_by_input_ref[item.id].kind.value,
            }
            for item in meaning.inputs
        ],
    }


def _origin_payload(origin: SourceOrigin) -> dict[str, object]:
    return {
        "source": origin.source.value,
        "meaning": origin.meaning,
        "resolved_input_ref": origin.resolved_input_ref,
    }


def _grouping_value_payload(
    shape: tuple[str, str | None],
) -> dict[str, str]:
    kind, grain = shape
    if kind in {"observed_value", "computed_value", "condition"} and grain is None:
        return {"kind": kind}
    if kind == "temporal_bucket" and grain is not None:
        return {"kind": kind, "grain": grain}
    raise ValueError("invalid grouping value shape")


def _value_type_label(value_type: ValueType) -> object:
    if isinstance(value_type, CollectionType):
        return {
            "kind": "collection",
            "element_kind": value_type_kind(value_type.element_type),
        }
    return {"kind": value_type_kind(value_type)}


def _conversation_resolution_sections(
    request: QuestionContractRequest,
    *,
    builder: TurnPromptBuilder,
) -> tuple[PromptSection, ...]:
    resolution = request.conversation_resolution
    if resolution is None:
        return ()
    payload = resolution.to_prompt_payload()
    if not payload:
        return ()
    return (
        builder.json_section("Conversation resolution context:", payload, indent=2),
    )


def _conversation_input_refs(request: QuestionContractRequest) -> tuple[str, ...]:
    resolution = request.conversation_resolution
    if resolution is None:
        return ()
    return tuple(item.input_ref for item in resolution.inputs)


__all__ = [
    "SEMANTIC_QUESTION_CONTRACT_INSTRUCTIONS",
    "SEMANTIC_QUESTION_FRAME_INSTRUCTIONS",
    "SEMANTIC_QUESTION_FRAME_TOOL_NAME",
    "SemanticQuestionContractTurnPrompt",
    "SemanticQuestionFrameTurnPrompt",
]
