"""Normative model-facing semantics for the relational Question Contract."""

from __future__ import annotations

from fervis.lookup.question_contract.request import QuestionContractRequest
from fervis.lookup.question_contract.semantic_schema import (
    build_semantic_question_contract_schema_for_meaning,
    build_semantic_question_frame_schema,
)
from fervis.lookup.question_contract.semantic_parser import (
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

A fact is one typed value observed for a set instance or through a declared
association.

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
candidate_set and grouping before other_sets and other_associations.

Requested-fact boundaries

Each shown requested_fact_ref is one fixed answer-request boundary.

Semantic structure

Use s1 for the shown candidate set. A related-instance identity grouping owns
its identified_set declaration and its direct association declaration from s1. other_sets
contains independently needed set refs not already owned by grouping, declared
once as s2, s3, and so on.
Association endpoints use those set refs. Declare each remaining independently used association once
as a1, a2, and so on. In a fact, observed_for_ref names the set or association
where the value is observed. other_associations contains independently needed
associations not already owned by grouping. Set-valued expressions use set_ref. Structurally
identical fact leaves denote one fact.

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
use the occurrences as the qualifying set, use related_instance_identity for
the returned entity grouping, and aggregate an occurrence-owned fact.

Reference candidate_set when the result returns its instances. When a
result groups exact candidate instances, use candidate_instance_identity. When
it groups exact related instances, use related_instance_identity with one
identified_set and association. The parser derives the Identifier fact,
identified set, and direct association from that single declaration. The grouped
output references the grouping entry by its group_ref. A set may also be the
argument of count; it is the row-domain reference used by count.

Qualification contains every user-stated true-or-false condition that decides
whether a candidate qualifies. Represent each shown IDENTITY_REFERENCE as its
input_ref in an input_comparison with one Identifier fact for its denoted set.
When that set differs from the candidate set, declare their association and
observe the Identifier fact through that association. An aggregate's filter
applies only to that aggregate. A Boolean output contains the requested Boolean
expression. value_comparison compares two observed or computed values.

Use forall only for every row associated with the current candidate. Use
coverage when every candidate must have an observation for every member of a
separate required-dimension set.

Coverage asks whether, for one candidate, every qualifying required member has
at least one qualifying observation. required_member_condition is the Boolean
expression for a required-member row, or null when the whole required-member
set applies. observation_condition is the Boolean expression for an
observation row. The candidate_set is the set evaluated by
qualification. The declared association graph supplies one unambiguous path
from the candidate and one from the required member to the observation set.

aggregate summarizes all rows retained by qualification. filtered_aggregate
adds one condition that applies only to that aggregate's argument rows.
distinct_argument states whether repeated argument values count once; neither
aggregate-local choice changes the outer candidate population.

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
candidate_instance_identity for the qualifying instance's exact identity,
related_instance_identity for the exact identity of another instance related
to it, and value for any other row-level grouping value. A value grouping writes
its expression. Grouped outputs and ordering use group_ref for grouping values;
aggregate expressions compute values over each group. A supplied collection may
restrict qualifying rows; the per-row value remains the grouping value.

Each output names the fact, set, group, input, or expression that the user asks
to receive. Ordering names the value that determines order. Selection states
which ordered rows survive.

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


SEMANTIC_QUESTION_FRAME_INSTRUCTIONS = """\
Authoring order

Write decision_basis first. For a complete question, write every answer_request,
then supplied_values.

decision_basis inventories the independent requested results and the concrete
supplied operands that constrain or compute them. Candidate-population phrases
belong to requested meaning rather than the supplied-operand inventory.

Answer requests

Write one answer_request for each independent factual result.

Each answer request writes result_kind, qualifying_row_kind,
grouping_meanings, and returned_candidate_identity for qualifying instances.
answer_values declares each unknown fact or value computed to return or rank
answer rows, once. supplied_values declares concrete values already given by
the question.
These two lists have disjoint ownership.
return_request_basis and returned_result state what the answer returns.
ordering_value_refs references values in ordering priority. Then write
selection and universal_shape.
qualifying_row_kind is the kind of row evaluated by qualification and
aggregate arguments before grouping. It is the returned instance kind only
for qualifying_instances. Identity, property, and time values that restrict
those rows belong to supplied_values.

result_kind describes the result rows before ordering and selection; selecting
one winning row does not make it scalar. Use scalar when the requested result
is one value over the whole qualifying set, qualifying_instances when the
requested rows are candidate instances, and grouped_results when one result
row is produced per grouping value, including when ordering compares an
aggregate computed for each grouping value.

When entities are ranked by an aggregate over related occurrences, the
occurrences are the candidate set, the entity is a grouping meaning, and the
result kind is grouped_results.

For qualifying_instances, returned_candidate_identity states the candidate
identity returned by each row. grouped_results uses grouping_meanings as the
values that identify each returned row.

Each answer value beyond the automatically returned qualifying-instance
identity or grouping identity has one declaration in answer_values. Write value_ref,
meaning, then origin.

return_request_basis states only what the answer wording asks to return. For
row or group results, returned_result is identities when the answer asks
“Which A ... B?” and therefore returns A while B determines ordering. It is
identities_and_values when an added answer clause such as “and what is B?”
asks to state declared non-key values; returned_value_refs lists those values.
A ranking clause places its value in ordering_value_refs.

ordering_value_refs lists the declared values that determine order, in priority
order. Each ordering_value_ref copies value_ref from one preceding
answer_values item. Ranking words such as first, last, highest, lowest, and
top N select result rows.
“Which A has the greatest B?” returns A; B orders A.
“Which A has the greatest B, and what is B?” returns A and B.

For scalar, answer_values contains exactly the one unknown value requested by
the question, returned_result is values, and ordering_value_refs is empty.
Relationships used to qualify candidates retain qualifying_instances.

grouping_meanings is empty for scalar and qualifying_instances. For
grouped_results, each item names one value that varies across the requested
result rows and defines one grouping dimension. A shared qualification or time
scope is not a grouping dimension.

selection is all_results when every qualifying result is requested,
first_rank_with_ties for a singular first, last, highest, or lowest request,
and take_with_boundary_ties for an explicit positive number of ordered results.
The take limit owns its positive-integer supplied value.

universal_shape is none when there is no universal requirement,
every_related_row when every row of one set related to the candidate must
satisfy a condition, and every_required_member_has_observation when every
member of one required set must have a matching observation from a different
set.

Each requested scalar value is one answer request. Sharing a candidate set or
time scope does not merge scalar values. A requested row or group result is one
answer request and may contain several columns at that result grain. A repeated
measure over a specified key set is one grouped requested fact, not one fact per
key.

Supplied values

supplied_values contains concrete values already supplied for qualification,
grouping, computation, ordering, or selection.

A business subject or requested unknown belongs to the answer request.
A supplied name, code, identifier, time expression, number, Boolean, explicit
text value, collection, arithmetic operand, or result limit is a supplied
value.

A business modifier may define qualifying_row_kind when it names the
business population, or a supplied scalar when it is independently compared.
That meaning has one owner.

A supplied value is a value or expression already provided by the question for
use in finding the answer. It may be compared, used in arithmetic, used as a
time scope, included in a supplied collection, or used as an explicit result
limit. Candidate-set nouns and the unknown answer are represented by the
answer requests.

One supplied value item owns one independent operand role. Alternatives filling
the same role share one item. A value shared by several answer requests appears
once.

For each supplied value, write meaning, denotation, then value.

denotation.basis explains what the supplied value itself denotes.
identity_reference means the value is a name, code, or identifier selecting
exactly one instance of a person, organization, place, product, or other named
entity kind. It remains an identity reference when many candidate rows relate
to that instance. A noun phrase naming the candidate kind or population
belongs to qualifying_row_kind. scalar means the value is a property, category,
status, time, quantity, or other value that is not the name, code, or
identifier of one entity instance.

value.operands copies the exact visible wording or exact shown resolved value.
meaning describes the value's role. value_type describes its intrinsic type.
identity_name_or_code is a name, code, or key for the instance denoted by an
identity_reference. property_value is a non-identity textual property,
category, status, or search value. temporal_scope is a date, time, interval, or
relative period that bounds observations.
origin identifies the question or resolved conversation input that supplied it.
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
                    "Return kind=missing_requested_fact only when no complete factual result is identifiable.",
                    "Return kind=unresolved_prior_turn_references only when the factual result is identifiable but required prior-turn references remain unresolved.",
                    "Return exactly one provider-native tool call.",
                ),
            ),
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
                    "Return kind=missing_requested_fact only when no complete factual result is identifiable.",
                    "Return kind=unresolved_prior_turn_references only when the factual result is identifiable but required prior-turn references remain unresolved.",
                    "Return exactly one provider-native tool call.",
                ),
            ),
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
                "result_kind": item.result_kind,
                "qualifying_row_kind": _origin_payload(item.candidate_set_origin),
                "grouping_meanings": [
                    _origin_payload(origin) for origin in item.grouping_origins
                ],
                "row_identity_meaning": (
                    _origin_payload(item.row_identity_origin)
                    if item.row_identity_origin is not None
                    else None
                ),
                "ordering_meanings": [
                    _origin_payload(origin) for origin in item.ordering_origins
                ],
                "output_meanings": [
                    _origin_payload(origin) for origin in item.output_origins
                ],
                "selection": {
                    "kind": item.selection_kind,
                    "limit_input_ref": item.selection_limit_input_ref,
                },
                "universal_shape": item.universal_shape,
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
