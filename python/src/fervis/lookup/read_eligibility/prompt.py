"""Prompt for API read retention."""

from __future__ import annotations

from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections import api_read_cards_xml
from fervis.lookup.read_eligibility.surface import (
    ReadEligibilityCandidateSurface,
    read_eligibility_candidate_surface,
)
from fervis.lookup.read_eligibility.model import ReadEligibilityRequest
from fervis.lookup.read_eligibility.input_bindings import interpretation_question
from fervis.lookup.read_eligibility.schema import build_read_eligibility_schema
from fervis.model_io.structured_output.specs import required_tool_spec


READ_ELIGIBILITY_TOOL_NAME = "submit_read_eligibility"


class ReadEligibilityTurnPrompt(TurnPromptBase):
    turn_name = "read eligibility"
    turn_task = "retain API reads that expose useful source evidence"

    def __init__(self, request: ReadEligibilityRequest) -> None:
        self.request = request
        self._candidate_surface: ReadEligibilityCandidateSurface | None = None

    def candidate_surface(self) -> ReadEligibilityCandidateSurface:
        if self._candidate_surface is None:
            self._candidate_surface = read_eligibility_candidate_surface(self.request)
        return self._candidate_surface

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.text_section(
                "Read eligibility context:",
                api_read_cards_xml(self.candidate_surface().card_payload),
            ),
        )

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Task Boundary",
                (
                    "First, interpret each named input in the complete requested-fact context and select one shown canonical identity.",
                    "Decide which shown API reads expose useful source evidence for the requested fact, treating every selected canonical identity as fixed.",
                    "Retain a read when its rows or fields can contribute to the requested fact.",
                    "Drop a read when it is clearly unrelated to the requested fact.",
                    "Drop a read when it is only related context, audit/detail data, telemetry, receipt data, verification evidence, or an indirect helper.",
                    "Use the fact's known inputs when deciding whether a read could contribute to that fact.",
                    "Do not require the read to contain the final answer, a precomputed aggregate, a precomputed ranking, or a final display row.",
                ),
            ),
            builder.instruction_block(
                "Canonical Interpretation",
                (
                    "Interpret each named input before assessing any candidate read. Use its supplied text, field-label approximation, value-meaning hint, and complete requested fact together.",
                    'For each named input, copy interpretation_question exactly, then write the `because` field as: "{input} denotes one {entity kind} because {requested-fact and resolver-card evidence}." Replace each template term with concrete text. Do not discuss whether a candidate read is attractive or should be retained.',
                    "Then select exactly one shown canonical_option_id whose result has the entity kind stated in because. That selection fixes what the input means for this requested fact. Copy the ID exactly. Do not invent, combine, reconsider, or replace resolver bindings, fields, keys, values, or joins while assessing reads.",
                ),
            ),
            builder.instruction_block(
                "Read Assessment Under Fixed Identities",
                (
                    "Assess every candidate read under the selected canonical interpretations. A candidate read cannot reinterpret a named input or select another resolver.",
                    "The selected canonical option fixes the named input's meaning while every candidate read is assessed. Text match fields are evidence for that identity; they are not computation values. This turn does not choose an application target.",
                ),
            ),
            builder.instruction_block(
                "Conversation Resolution",
                (
                    "When conversation-resolution annotations apply to a requested fact, use them as part of the requested fact meaning.",
                ),
            ),
            builder.instruction_block(
                "Retention Rules",
                (
                    "Retain row-level reads that expose rows the requested fact may count, list, filter, group, rank, or aggregate.",
                    "Retain summary/report reads when returned values may answer or measure the requested fact.",
                    "Do not reject a read only because its endpoint resource name is broader than the requested subject.",
                    "For example, location rows with a type field may remain useful for a store question.",
                    "Do not retain reads whose only basis is that they might help validate, explain, audit, enrich, or provide context for another answer-bearing read.",
                    "For RETAIN, retention_basis describes the rows and fields that make the read relevant to the requested fact. It must not rate, rank, or compare retained reads.",
                    "Do not claim a retained read directly answers the requested fact unless the shown response fields alone contain the requested answer value or measure.",
                    "For RETAIN, relevant_field_tokens is the answer-changing computation support set for the requested fact, not all useful fields on a retained read.",
                    "Include a field only if its value can change the computed answer by establishing the requested row grain or identity, applying a requested filter/time/status/discriminator, grouping, ranking, joining, deduplicating, or supplying a requested measure.",
                    "Do not include fields whose values only describe, decorate, audit, explain, or provide context for rows that are already in or out of the answer population.",
                    "For DROP, use empty relevant_row_path_tokens and empty relevant_field_tokens.",
                ),
            ),
            builder.instruction_block(
                "Output Shape",
                (
                    "In requested_fact_assessments, use every shown requested_fact id as a key.",
                    "In canonical_inputs, use every shown known_input id as a key. Copy interpretation_question, write because, and select one shown canonical_option_id.",
                    "In read_candidate_reviews, use every shown source_candidate id for that requested fact as a key.",
                    "For RETAIN, first cite relevant_row_path_tokens and relevant_field_tokens, then write retention_basis.",
                    "For DROP, do not write relevant_row_path_tokens or relevant_field_tokens.",
                    "relevant_row_path_tokens cites zero or more response_rows evidence_token values from the same read_candidate.",
                    "relevant_field_tokens cites zero or more response_rows.fields evidence_token values from the same read_candidate.",
                    "retention_basis explains which rows and fields justify retaining the read, or why the read provides no useful source evidence.",
                    "retention_decision is the final field in each read_candidate_reviews item and must be RETAIN or DROP.",
                    "Do not create support sets, support roles, metric reviews, param decisions, or source-binding decisions.",
                ),
            ),
            builder.instruction_block(
                "Validity",
                (
                    "Copy every requested_fact, known_input, source_candidate, canonical_option_id, and evidence_token exactly from the prompt.",
                    "Every evidence_token must come from the read_candidate keyed by its read_candidate_reviews entry.",
                    "Do not invent endpoints, params, output fields, catalog facts, or IDs.",
                    "Do not use host-domain assumptions that are not in requested facts, conversation context, or API read cards.",
                ),
            ),
            builder.instruction_block(
                "Output",
                ("Return the submit_read_eligibility tool call only.",),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(provider_schema=self._schema())

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=READ_ELIGIBILITY_TOOL_NAME,
                    tool_description=(
                        "Submit canonical-input selections and API read retention assessments."
                    ),
                    input_schema=self._schema(),
                ),
            )
        )

    def _schema(self) -> dict[str, object]:
        return build_read_eligibility_schema(
            canonical_options_by_requested_fact_id=(
                self._canonical_options_by_requested_fact_id()
            ),
            candidate_reviews_by_requested_fact_id=(
                self._candidate_reviews_by_requested_fact_id()
            ),
        )

    def _candidate_reviews_by_requested_fact_id(
        self,
    ) -> dict[str, tuple[dict[str, object], ...]]:
        reviews_by_fact_id: dict[str, list[dict[str, object]]] = {
            fact.id: [] for fact in self.request.requested_facts
        }
        for scope in self.candidate_surface().candidate_scopes:
            reviews_by_fact_id.setdefault(scope.requested_fact_id, []).append(
                {
                    "source_candidate_id": scope.source_candidate_id,
                    "read_id": scope.read_id,
                    "row_path_tokens": tuple(scope.row_path_ids_by_evidence_token),
                    "field_tokens": tuple(scope.field_refs_by_evidence_token),
                }
            )
        return {
            requested_fact_id: tuple(candidate_reviews)
            for requested_fact_id, candidate_reviews in reviews_by_fact_id.items()
        }

    def _canonical_options_by_requested_fact_id(
        self,
    ) -> dict[str, tuple[dict[str, object], ...]]:
        specs_by_fact: dict[str, list[dict[str, object]]] = {
            fact.id: [] for fact in self.request.requested_facts
        }
        options = self.candidate_surface().canonical_options
        for fact in self.request.requested_facts:
            known_inputs_by_id = {item.id: item for item in fact.known_inputs}
            known_input_ids = tuple(
                dict.fromkeys(
                    option.known_input_id
                    for option in options
                    if option.requested_fact_id == fact.id
                )
            )
            for known_input_id in known_input_ids:
                known_input = known_inputs_by_id[known_input_id]
                matching = tuple(
                    option
                    for option in options
                    if option.requested_fact_id == fact.id
                    and option.known_input_id == known_input_id
                )
                specs_by_fact[fact.id].append(
                    {
                        "known_input_id": matching[0].known_input_token,
                        "interpretation_question": interpretation_question(
                            known_input_text=known_input.text,
                            answer_fact=fact.description,
                        ),
                        "canonical_options": tuple(
                            {
                                "canonical_option_id": option.id,
                            }
                            for option in matching
                        ),
                    }
                )
        return {key: tuple(value) for key, value in specs_by_fact.items()}
