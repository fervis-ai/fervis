"""Prompt for fact-local plan selection."""

from __future__ import annotations

from typing import Any

from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections import source_alignment_reviews_xml
from fervis.lookup.plan_selection.source_strategies import (
    source_alignment_candidate_payload,
    source_alignment_candidates_by_fact,
    source_candidate_ids_by_requested_fact_id,
    source_strategy_payload,
    source_strategies_by_fact,
)
from fervis.lookup.operation_families.plan_selection_registry import (
    plan_selection_shape_specs_for_family,
)
from fervis.lookup.plan_selection.model import PlanSelectionRequest
from fervis.lookup.plan_selection.schema import build_plan_selection_schema
from fervis.model_io.structured_output.specs import required_tool_spec


PLAN_SELECTION_TOOL_NAME = "submit_source_alignment_reviews"


class PlanSelectionTurnPrompt(TurnPromptBase):
    turn_name = "source alignment review"
    turn_task = "review source alignment"

    def __init__(self, request: PlanSelectionRequest) -> None:
        self.request = request

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.text_section(
                "Source alignment reviews:",
                source_alignment_reviews_xml(
                    self.source_alignment_candidates_payload()
                ),
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
                    "For each requested_fact in the XML, review every source_candidate inside it.",
                    "Write exactly one alignment review for every source_candidate shown under that requested_fact.",
                    "Assess source alignment only; executable choices are made in later turns.",
                    "The backend will forward aligned sources to source binding.",
                    "If no shown candidate is aligned, review every source as NOT_ALIGNED; the backend derives the terminal outcome.",
                ),
            ),
            builder.instruction_block(
                "Source Alignment",
                (
                    "Compare each source_candidate against the fact_text and answer_outputs in the same requested_fact block.",
                    "Use the source response rows, field names, row cardinality, input params, and read description to assess business meaning alignment.",
                    "Use source_alignment=DIRECT when the source can plausibly answer the requested fact by itself.",
                    "Use source_alignment=PARTIAL when the source contains concrete evidence needed for the requested fact, but needs another source or later computation to complete the answer.",
                    "Use source_alignment=NOT_ALIGNED when the source is only related or shape-compatible; do not forward it.",
                    "For each review, write basis before source_alignment.",
                    "A source can be DIRECT even if source binding still needs to choose params, filters, metrics, or groups later.",
                ),
            ),
            builder.instruction_block(
                "Validity",
                (
                    "For each requested_fact, review every source_candidate shown inside that requested_fact.",
                    "Use the requested_fact id as the key in reviews_by_requested_fact.",
                    "Use each source_candidate id as the key for its review.",
                    "Each review's source_candidate_id must match that source_candidate key.",
                    "Do not invent sources, fields, params, facts, outputs, metrics, calculations, or labels.",
                    "Do not use read-eligibility retention text as final source truth; evaluate only the source candidate shown in this turn.",
                ),
            ),
            builder.instruction_block(
                "Output",
                ("Return the submit_source_alignment_reviews tool call only.",),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(provider_schema=self._schema())

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=PLAN_SELECTION_TOOL_NAME,
                    tool_description="Submit source alignment reviews.",
                    input_schema=self._schema(),
                ),
            )
        )

    def plan_selection_candidates_payload(self) -> dict[str, object]:
        strategies_by_fact = source_strategies_by_fact(
            self.request.source_candidate_payload,
            requested_facts=self.request.requested_facts,
            relation_catalog=self.request.relation_catalog,
            shape_specs_for_family=plan_selection_shape_specs_for_family,
        )
        return {
            "requested_fact_source_strategies": [
                {
                    "requested_fact_id": fact.id,
                    "answer_outputs": [
                        {
                            "answer_output_id": output.id,
                            "description": output.description,
                        }
                        for output in fact.support_answer_outputs
                    ],
                    "source_strategies": [
                        source_strategy_payload(source_strategy)
                        for source_strategy in strategies_by_fact.get(fact.id, ())
                    ],
                }
                for fact in self.request.requested_facts
            ]
        }

    def source_alignment_candidates_payload(self) -> dict[str, object]:
        strategies_by_fact = source_strategies_by_fact(
            self.request.source_candidate_payload,
            requested_facts=self.request.requested_facts,
            relation_catalog=self.request.relation_catalog,
            shape_specs_for_family=plan_selection_shape_specs_for_family,
        )
        candidates_by_fact = source_alignment_candidates_by_fact(strategies_by_fact)
        return {
            "requested_fact_source_candidates": [
                {
                    "requested_fact_id": fact.id,
                    "fact_text": fact.description,
                    "answer_outputs": [
                        {
                            "answer_output_id": output.id,
                            "description": output.description,
                        }
                        for output in fact.support_answer_outputs
                    ],
                    "source_candidates": [
                        source_alignment_candidate_payload(candidate)
                        for candidate in candidates_by_fact.get(fact.id, ())
                    ],
                }
                for fact in self.request.requested_facts
            ]
        }

    def _schema(self) -> dict[str, Any]:
        strategies_by_fact = source_strategies_by_fact(
            self.request.source_candidate_payload,
            requested_facts=self.request.requested_facts,
            relation_catalog=self.request.relation_catalog,
            shape_specs_for_family=plan_selection_shape_specs_for_family,
        )
        candidates_by_fact = source_alignment_candidates_by_fact(strategies_by_fact)
        return build_plan_selection_schema(
            requested_fact_ids=tuple(fact.id for fact in self.request.requested_facts),
            source_candidate_ids_by_requested_fact_id=(
                source_candidate_ids_by_requested_fact_id(candidates_by_fact)
            ),
        )
