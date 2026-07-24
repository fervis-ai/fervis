"""Model-facing semantic Query Enrichment contract."""

from __future__ import annotations

from fervis.lookup.query_enrichment.semantic import SemanticQueryEnrichmentRequest
from fervis.lookup.query_enrichment.semantic_schema import (
    build_semantic_query_enrichment_schema,
)
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.model_io.structured_output.specs import required_tool_spec


SEMANTIC_QUERY_ENRICHMENT_TOOL_NAME = "submit_query_enrichment"


class SemanticQueryEnrichmentTurnPrompt(TurnPromptBase):
    turn_name = "query enrichment"
    turn_task = "match semantic requirements to catalog resource names for recall"

    def __init__(self, request: SemanticQueryEnrichmentRequest) -> None:
        self.request = request

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Recall buckets:",
                {
                    "recall_buckets": [
                        {
                            "bucket_ref": item.bucket_ref,
                            "kind": item.kind.value,
                            "meaning": item.origin.meaning,
                            "requirement_refs": [
                                ref.token for ref in item.requirement_refs
                            ],
                        }
                        for item in self.request.recall_buckets
                    ]
                },
                indent=2,
            ),
            builder.json_section(
                "Reference input uses:",
                {
                    "reference_input_uses": [
                        {
                            "input_use_ref": item.input_use_ref,
                            "input_text": item.input_origin.meaning,
                            "operand_meaning": item.operand_meaning,
                            "reference_fact_ref": item.reference_fact_ref.token,
                            "expected_set_ref": (
                                item.expected_set_ref.token
                                if item.expected_set_ref is not None
                                else None
                            ),
                        }
                        for item in self.request.reference_tasks
                    ]
                },
                indent=2,
            ),
            builder.json_section(
                "Catalog vocabulary:",
                {
                    "resource_name_groups": [
                        {
                            "group_ref": f"group_{index}",
                            "resource_names": list(resource_names),
                        }
                        for index, resource_names in enumerate(
                            self.request.resource_name_groups, start=1
                        )
                    ],
                },
                indent=2,
            ),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Vocabulary scope",
                (
                    "Task: create a resource-name lineage ledger for each recall bucket. This is recall only.",
                    "Objective: maximize recall of catalog resource_names that the next step should inspect.",
                    "resource_names are the only strings allowed in exhaustive_resource_names, matching_resource_names, and catalog_search_terms.",
                ),
            ),
            builder.instruction_block(
                "Recall bucket resource lineage",
                (
                    "Write exactly one recall_bucket_matches item for every shown bucket_ref.",
                    "For each bucket, compare its meaning with every shown resource_name.",
                    "First write exhaustive_resource_names. Include every resource name with semantic proximity to the bucket: the same concept in different words, a broader or narrower form of that concept, or a closely related process. Uncertainty means inclusion.",
                    "Then write matching_resource_names as the names from that inventory that actually fit the bucket.",
                    "Every matching resource name appears in exhaustive_resource_names.",
                    "Several plausible resource names remain together for later catalog-aware assessment.",
                ),
            ),
            builder.instruction_block(
                "Reference value resolver search terms",
                (
                    "Write exactly one input_resource_search_terms item for every shown reference input use.",
                    "catalog_search_terms contains shown resource names whose instances could be denoted by the supplied operand under operand_meaning.",
                    "Order catalog_search_terms from the most direct identity class to less direct aliases.",
                    "An empty array means no shown resource type is denoted.",
                ),
            ),
            builder.instruction_block(
                "Boundary",
                (
                    "This turn selects resource-name recall vocabulary.",
                    "Later catalog-aware turns own API reads, endpoints, fields, parameters, source sufficiency, and final read selection.",
                    "Copy every bucket_ref and input_use_ref verbatim.",
                ),
            ),
            builder.instruction_block(
                "Output",
                ("Return exactly one submit_query_enrichment tool call.",),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                SEMANTIC_QUERY_ENRICHMENT_TOOL_NAME: (
                    build_semantic_query_enrichment_schema(self.request)
                )
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SEMANTIC_QUERY_ENRICHMENT_TOOL_NAME,
                    tool_description="Submit semantic requirement recall matches.",
                    input_schema=build_semantic_query_enrichment_schema(self.request),
                ),
            )
        )


__all__ = [
    "SEMANTIC_QUERY_ENRICHMENT_TOOL_NAME",
    "SemanticQueryEnrichmentTurnPrompt",
]
