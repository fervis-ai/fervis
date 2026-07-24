"""Model-facing semantic Plan Selection contract."""

from __future__ import annotations

from fervis.lookup.relation_catalog.row_sources import RowSourceKind
from fervis.lookup.relation_catalog.model import requires_caller_supplied_input
from fervis.lookup.plan_selection.semantic import SemanticPlanSelectionRequest
from fervis.lookup.plan_selection.semantic_schema import (
    build_semantic_plan_selection_schema,
)
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections.semantic_requirements import (
    plan_selection_fact_prompt_payload,
)
from fervis.lookup.turn_prompts.projections.canonical_values import (
    canonical_values_prompt_items,
)
from fervis.lookup.turn_prompts.projections.source_identities import (
    source_identity_evidence_prompt_items,
)
from fervis.model_io.structured_output.specs import required_tool_spec


SEMANTIC_PLAN_SELECTION_TOOL_NAME = "submit_plan_selection"


class SemanticPlanSelectionTurnPrompt(TurnPromptBase):
    turn_name = "plan selection"
    turn_task = "assess source alignment for every requested fact"

    def __init__(self, request: SemanticPlanSelectionRequest) -> None:
        self.request = request

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Requested facts:",
                {
                    "requested_facts": [
                        plan_selection_fact_prompt_payload(index)
                        for index in self.request.indexes
                    ]
                },
                indent=2,
            ),
            builder.json_section(
                "Certified input values:",
                {"values": list(canonical_values_prompt_items(self.request.canonical_values))},
                indent=2,
            ),
            builder.json_section(
                "Available sources and declared relations:",
                self._source_payload(),
                indent=2,
            ),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Source alignment",
                (
                    "Compare each source against the fact text, required facts, and requested outputs in the same requested fact.",
                    "Assess each source without reinterpreting any shown certified input. Applying those inputs belongs to Source Binding.",
                    "Use the source response rows, field names, row cardinality, and input params to assess business meaning alignment.",
                    "DIRECT means this source contains the complete raw ingredient set needed to answer the requested fact by itself, including every requested output at its declared value kind, even if later steps must choose params, filters, metrics, groups, aggregation, ordering, arithmetic, or rendering. Producing a requested canonical identity requires matching declared identity evidence.",
                    "PARTIAL means this source contains a necessary raw ingredient, but the fact cannot be answered without combining it with another source.",
                    "NOT_ALIGNED means the source is only related, adjacent, or shape-compatible, or lacks the raw ingredients needed for the requested fact.",
                    "A source restricted to a specialized population not requested by the question is NOT_ALIGNED, even when its rows expose every required field.",
                    "A certified identity keeps its declared entity_kind and key_id. Compare that identity contract with each source's declared identity_evidence before assessing alignment.",
                    "A source related to the certified identity is not DIRECT unless it contains the complete raw ingredients for the requested fact.",
                    "Write basis before alignment for every source.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Return exactly one source_assessments_by_requested_fact entry for every shown requested fact.",
                    "Within each requested fact, assess every shown source exactly once.",
                    "The backend derives the source strategy from these DIRECT, PARTIAL, and NOT_ALIGNED assessments and shown relation evidence.",
                    "Return exactly one submit_plan_selection tool call.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                SEMANTIC_PLAN_SELECTION_TOOL_NAME: (
                    build_semantic_plan_selection_schema(self.request)
                )
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SEMANTIC_PLAN_SELECTION_TOOL_NAME,
                    tool_description="Submit source alignment assessments.",
                    input_schema=build_semantic_plan_selection_schema(self.request),
                ),
            )
        )

    def _source_payload(self) -> dict[str, object]:
        return {
            "contract_snapshot_ref": self.request.source_catalog.contract_snapshot.ref,
            "sources": [
                {
                    "source_ref": source.id,
                    "kind": source.kind.value,
                    "label": source.label,
                    "description": source.description,
                    "resource_names": list(source.resource_names),
                    "row_cardinality": source.row_cardinality.value,
                    "fields": [
                        {
                            "field_ref": field.field_ref,
                            "label": field.label,
                            "type": field.type.value,
                            "description": field.description,
                            "choices": list(field.choices),
                        }
                        for field in source.fields
                    ],
                    "params": [
                        {
                            "param_ref": param.param_ref,
                            "name": param.name,
                            "source": str(param.source),
                            "type": param.type.value,
                            "required": param.required,
                            "default": param.default,
                            "choices": list(param.choices),
                            "description": param.description,
                            "entity_target": (
                                None
                                if param.entity_target is None
                                else {
                                    "entity_kind": param.entity_target.entity_kind,
                                    "key_id": param.entity_target.key_id,
                                    "component_id": param.entity_target.component_id,
                                }
                            ),
                        }
                        for param in source.params
                    ],
                    **(
                        {
                            "read_id": source.read_id,
                            "required_parameter_refs": [
                                item.param_ref
                                for item in source.params
                                if requires_caller_supplied_input(item)
                            ],
                        }
                        if source.kind is RowSourceKind.API_READ
                        else {}
                    ),
                }
                for source in self.request.source_catalog.sources
            ],
            "relation_evidence": [
                {
                    "evidence_ref": item.evidence_ref,
                    "left_source_ref": item.left_source_ref,
                    "right_source_ref": item.right_source_ref,
                    "left_field_refs": list(item.left_field_refs),
                    "right_field_refs": list(item.right_field_refs),
                }
                for item in self.request.source_catalog.relation_evidence
            ],
            "identity_evidence": list(
                source_identity_evidence_prompt_items(self.request.source_catalog)
            ),
            "allowed_alignments_by_requested_fact": {
                index.requested_fact_id: {
                    source.id: [
                        item.value
                        for item in self.request.allowed_alignments(
                            requested_fact_id=index.requested_fact_id,
                            source_ref=source.id,
                        )
                    ]
                    for source in self.request.source_catalog.sources
                }
                for index in self.request.indexes
            },
        }


__all__ = [
    "SEMANTIC_PLAN_SELECTION_TOOL_NAME",
    "SemanticPlanSelectionTurnPrompt",
]
