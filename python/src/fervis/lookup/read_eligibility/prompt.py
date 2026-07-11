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
            builder.json_section("Requested facts:", self.requested_facts_payload()),
            builder.text_section(
                "Candidate API reads:",
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
                    "Decide only whether each shown API read should remain available for later turns.",
                    "Retain a read when it exposes row or field evidence that could be directly consumed by later turns for the requested fact.",
                    "Drop a read only when it is clearly unrelated to the requested fact.",
                    "Drop a read when it is only related context, audit/detail data, telemetry, receipt data, verification evidence, or an indirect helper.",
                    "When a direct answer input read is incomplete, retain it. Later turns decide source binding, filters, metrics, and execution.",
                    "When a candidate shows applicable_known_inputs, treat them as backend-resolved references that can narrow that read for the requested fact.",
                    "Do not require the read to contain the final answer, a precomputed aggregate, a precomputed ranking, or a final display row.",
                    "Do not decide which read is best, bind params, choose enum values, write formulas, select metrics, or validate operation legality.",
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
                    "Retain row-level reads that expose rows that later turns could count, list, filter, group, rank, or aggregate for the requested fact.",
                    "Retain summary/report reads when returned values may answer or measure the requested fact.",
                    "Retain broader row populations when input params or returned discriminator fields can narrow those rows later.",
                    "Retain broader row populations when applicable_known_inputs show that a resolved question input applies through a response field.",
                    "Do not reject a read only because its endpoint resource name is broader than the requested subject.",
                    "For example, location rows with a type field may remain useful for a store question.",
                    "Do not retain reads whose only basis is that they might help validate, explain, audit, enrich, or provide context for another answer-bearing read.",
                    "For RETAIN, retention_basis describes the rows and fields this read exposes for later turns. It must not rate, rank, or compare retained reads.",
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
                    "For each requested_fact, copy requested_fact_id in the same order shown.",
                    "Write one read_candidate_reviews item for every shown read_candidate for that requested_fact, in the same order shown.",
                    "Inside each read_candidate_reviews item, copy source_candidate_id and read_id from that read_candidate.",
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
                    "Copy requested_fact_id, source_candidate_id, read_id, and evidence_token values verbatim from the prompt.",
                    "Every evidence_token must be copied from the same read_candidate identified by read_candidate_reviews.source_candidate_id.",
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
                    tool_description="Submit API read retention assessments.",
                    input_schema=self._schema(),
                ),
            )
        )

    def requested_facts_payload(self) -> dict[str, object]:
        return {
            "requested_facts": [
                {
                    "requested_fact_id": fact.id,
                    "description": fact.description,
                    "answer_request": fact.answer_request_model_dict(),
                    "answer_outputs": [
                        {
                            "answer_output_id": output.id,
                            "description": output.description,
                        }
                        for output in fact.answer_outputs
                    ],
                }
                for fact in self.request.requested_facts
            ]
        }

    def _schema(self) -> dict[str, object]:
        return build_read_eligibility_schema(
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
