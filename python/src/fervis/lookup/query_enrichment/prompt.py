"""Prompt for catalog query enrichment."""

from __future__ import annotations

from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.question_contract import RequestedFactLiteralInput
from fervis.lookup.turn_prompts.projections import answer_output_prompt_payload
from fervis.lookup.query_enrichment.model import (
    QueryEnrichmentRequest,
    query_enrichment_endpoint_names,
    query_enrichment_resource_names,
)
from fervis.lookup.query_enrichment.schema import build_query_enrichment_schema
from fervis.model_io.structured_output.specs import required_tool_spec


QUERY_ENRICHMENT_TOOL_NAME = "submit_query_enrichment"


class QueryEnrichmentTurnPrompt(TurnPromptBase):
    turn_name = "query enrichment"
    turn_task = "create source-text to API resource-name matches for recall"

    def __init__(self, request: QueryEnrichmentRequest) -> None:
        self.request = request

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Requested facts:",
                self.requested_facts_payload(),
                indent=2,
            ),
            builder.json_section(
                "Entity targets:",
                self.entity_targets_payload(),
                indent=2,
            ),
            builder.json_section(
                "API catalog vocabulary:",
                self.api_catalog_vocabulary_payload(),
                indent=2,
            ),
        )

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Vocabulary Scope",
                (
                    "Task: create an answer-output resource-name lineage ledger for each requested fact. This is recall only.",
                    "Objective: maximize recall of API resource_names that the next step should inspect.",
                    "resource_names are the only strings allowed in matching_resource_names.",
                    "endpoint_names are shown only so you know which strings not to copy as matching_resource_names.",
                    "support_roles are the only strings allowed in support_role.",
                ),
            ),
            builder.instruction_block(
                "Conversation Resolution Annotations",
                (),
            ),
            builder.instruction_block(
                "Answer Output Resource Lineage",
                (
                    "For each requested fact, write exactly one requested_fact_resource_name_matches item.",
                    "Within it, write answer_output_resource_lineage rows.",
                    "Each row says: for this answer_output and support_role, this source_text could be backed by these API resource_names.",
                    "Copy answer_output_id verbatim from requested_facts.answer_outputs.",
                    "Copy support_role from the allowed support_roles.",
                    "Each source_text is an exact phrase from the current question, requested fact, answer output, or resolved question inputs.",
                    "Create source_text entries for phrases that could name a domain concept, event, record, amount source, actor, object, or grouping entity in the API.",
                    "Do not create entries for words that only express operation, ranking, output shape, or time scope unless they also name an API resource.",
                    "For each answer_output, include every resource_name that could support that output's row population, answer value, measured value, group key, or population scope.",
                    "When more than one resource lineage is plausible for the same answer_output and support_role, preserve all plausible resource_names.",
                    "If a field or metric phrase points to a resource_name indirectly, include that resource_name in the matching lineage row.",
                    "For each lineage row, copy every API catalog resource_name that could match source_text for that answer_output and support_role.",
                    "Do not choose final query terms here. Do not rank. Do not decide which match is correct.",
                ),
            ),
            builder.instruction_block(
                "Resource Lineage Matches",
                (
                    "matching_resource_names must be exact strings copied from API catalog vocabulary.resource_names.",
                    "Do not output endpoint names, fields, params, IDs, docstrings, invented labels, or paraphrases.",
                    "Do not change case, spacing, spelling, singular/plural form, or punctuation.",
                    "If a useful word is not in resource_names, do not output it.",
                    "Do not repeat matching_resource_names inside one answer_output_resource_lineage item.",
                    "Do not write an answer_output_resource_lineage item with empty matching_resource_names.",
                    "If no resource_name matches any source phrase for a requested fact, return answer_output_resource_lineage: [] for that requested fact.",
                ),
            ),
            builder.instruction_block(
                "Reference Value Resolver Search Terms",
                (
                    "For each entity target, choose catalog_search_terms that help find the canonical identity record for the reference value.",
                    "For each term, write basis before term using this template:",
                    '"{term} can identify {resolved_value_text} because value_meaning_hint is {value_meaning_hint}."',
                    "Only include a term when the basis sentence is true.",
                    "Use value_meaning_hint as evidence for identity lookup terms.",
                    "Order catalog_search_terms from most direct identity class to less direct aliases; resolver recall treats earlier terms as higher priority.",
                    "Prefer terms that name the entity identity class or a close API alias.",
                    "Do not include terms that describe facts about the entity rather than the entity identity.",
                    "Do not repeat catalog_search_terms.",
                    "Do not copy reference_text or resolved_value_text into catalog_search_terms.",
                    "Do not output endpoint names, fields, params, IDs, docstrings, invented labels, or paraphrases.",
                    "Return [] only when no catalog wording helps find the entity identity.",
                ),
            ),
            builder.instruction_block(
                "Boundaries",
                (
                    "Do not choose API reads, endpoints, fields, params, filters, IDs, or source invocations.",
                    "Do not decide whether a resource_name can fully answer the fact.",
                    "Do not decide which resource-name match is correct.",
                    "Do not explain outside the required output fields.",
                ),
            ),
            builder.instruction_block(
                "Copying And Validity",
                (
                    "Copy requested_fact_id verbatim.",
                    "Copy answer_output_id verbatim.",
                    "Copy target_id verbatim.",
                    "Write exactly one requested_fact_resource_name_matches item for every requested fact.",
                    "Write exactly one entity_target_catalog_search_terms item for every entity target.",
                    "Do not invent requested_fact_id values.",
                    "Do not invent target_id values.",
                ),
            ),
            builder.instruction_block(
                "Output",
                ("Return the submit_query_enrichment tool call only.",),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema=build_query_enrichment_schema(
                resource_names=query_enrichment_resource_names(self.request)
            )
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=QUERY_ENRICHMENT_TOOL_NAME,
                    tool_description=(
                        "Submit source-text to API resource-name matches "
                        "for requested facts."
                    ),
                    input_schema=build_query_enrichment_schema(
                        resource_names=query_enrichment_resource_names(self.request)
                    ),
                ),
            )
        )

    def requested_facts_payload(self) -> dict[str, object]:
        return {
            "requested_facts": [
                {
                    "requested_fact_id": fact.id,
                    "requested_fact_description": fact.description,
                    "answer_outputs": [
                        answer_output_prompt_payload(output)
                        for output in fact.answer_outputs
                    ],
                    "question_text_input_text": [
                        known.text for known in fact.known_inputs
                    ],
                }
                for fact in self.request.requested_facts
            ]
        }

    def entity_targets_payload(self) -> dict[str, object]:
        targets: dict[str, dict[str, object]] = {}
        for fact in self.request.requested_facts:
            for known in fact.known_inputs:
                match known:
                    case RequestedFactLiteralInput() if known.is_reference_value:
                        targets.setdefault(
                            known.id,
                            {
                                "target_id": known.id,
                                "reference_text": known.text,
                                "resolved_value_text": known.resolved_value_text,
                                "value_meaning_hint": known.value_meaning_hint,
                            },
                        )
                    case _:
                        continue
        return {"entity_targets": list(targets.values())}

    def api_catalog_vocabulary_payload(self) -> dict[str, object]:
        return {
            "api_catalog_vocabulary": {
                "resource_names": list(query_enrichment_resource_names(self.request)),
                "endpoint_names": list(query_enrichment_endpoint_names(self.request)),
                "support_roles": list(ANSWER_OUTPUT_SUPPORT_ROLE_VALUES),
            }
        }
