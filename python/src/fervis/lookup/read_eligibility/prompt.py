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
                    "First, assess which shown API reads expose useful source evidence for the requested fact.",
                    "When a named input has several canonical options, assess reads before selecting one option. Retain a read when it could contribute under at least one shown option.",
                    "Retain a read when its rows or fields can contribute to the requested fact.",
                    "Drop a read when it is clearly unrelated to the requested fact.",
                    "Drop a read when it is only related context, audit/detail data, telemetry, receipt data, verification evidence, or an indirect helper.",
                    "Use the fact's known inputs when deciding whether a read could contribute to that fact.",
                    "Do not require the read to contain the final answer, a precomputed aggregate, a precomputed ranking, or a final display row.",
                ),
            ),
            builder.instruction_block(
                "Read Assessment Before Canonical Interpretation",
                (
                    "Assess every candidate read before selecting a canonical option. Use the supplied input text and every shown canonical option as fact context.",
                    "When a read could contribute under at least one shown canonical option, assess its rows and fields normally. Do not choose an option inside a read review.",
                ),
            ),
            builder.instruction_block(
                "Canonical Interpretation",
                (
                    "After assessing every candidate read, interpret each named input using its supplied text, field-label approximation, value-meaning hint, complete requested fact, resolver cards, and the completed read reviews.",
                    'In canonical_option_assessments, assess every shown canonical meaning exactly once. Use the canonical_option_id as the key and write: "{canonical result}: {which reviewed reads expose this identity and what those reads contribute to the requested fact}." Resolver routes nested under one canonical option all produce that same meaning; do not assess them as different meanings.',
                    'After all option assessments, write because as: "Use {selected canonical result} because {its reviewed reads contribute the requested fact using that identity}. Do not use {each alternative canonical result} because {its reviewed reads contribute different evidence or require an undeclared identity conversion}. Therefore, {input} denotes {selected canonical result}." Then select exactly one shown canonical_option_id and copy it exactly.',
                    "Before resolver_option_id, assess every shown resolver route under the selected canonical option in resolver_option_assessments. For each route, state whether its lookup_request_parameters represent the supplied input under identifier_kind.",
                    "A returned_identity_verification_field verifies only the resource retrieved by that request. It cannot make an unsuitable lookup request parameter suitable.",
                    "After all resolver route assessments, select exactly one nested resolver option and copy its option_id as resolver_option_id. This chooses how to obtain or validate the fixed meaning; it does not create another meaning decision.",
                    "The selected canonical option fixes the named input's meaning for subsequent steps. Returned identity verification fields are identity evidence, not computation values. This turn does not choose an application target.",
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
                    "In read_candidate_reviews, use every shown source_candidate id for that requested fact as a key.",
                    "After read_candidate_reviews, write canonical_inputs. Use every shown known_input id as a key. Copy interpretation_question, assess every shown canonical meaning in canonical_option_assessments, then write because and canonical_option_id. Write resolver_option_assessments next, then resolver_option_id when the selected option shows resolver routes.",
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
                    "Copy every resolver option ID in resolver_option_assessments exactly from the selected canonical option. Copy resolver_option_id exactly from the selected nested resolver's option_id.",
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
                        "Submit API read retention assessments followed by canonical-input selections."
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
                                "resolver_option_ids": tuple(
                                    binding.option_id
                                    for binding in option.resolver_bindings
                                ),
                            }
                            for option in matching
                        ),
                    }
                )
        return {key: tuple(value) for key, value in specs_by_fact.items()}
