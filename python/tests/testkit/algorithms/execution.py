from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.plan_execution.compiled_execution import compile_fact_execution
from fervis.lookup.fact_plan.fact_plan import AnswerPlan, FactFulfillment
from fervis.lookup.fact_plan.operations import Operation, ProjectField, ProjectSpec
from fervis.lookup.fact_plan.relations import (
    PopulationChoiceControllerKind,
    Relation,
    RelationField,
    RelationSource,
    RelationSourcePopulationChoice,
    RelationSourceReviewScopeDecision,
    ReviewScopeDecisionKind,
    SourceKind,
)
from fervis.lookup.fact_plan.render_spec import RenderRelationOutput, RenderSpec
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
)

from tests.testkit.assertions import subset_mismatches


def run_execution_proof_graph_case(payload: dict[str, Any]) -> list[str]:
    compiled = compile_fact_execution(
        answer=_answer_plan(payload["input"]),
        catalog=RelationCatalog(),
        row_sources=_row_source_catalog(payload["input"]),
    )
    return subset_mismatches(
        actual={
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind.value,
                    "label": node.label,
                    "value": node.value,
                }
                for node in compiled.proof_graph.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "role": edge.role.value,
                }
                for edge in compiled.proof_graph.edges
            ],
            "contributions": [
                {
                    "origin": item.origin.value,
                    "label": item.label,
                    "node_refs": list(item.node_refs),
                }
                for item in compiled.proof_graph.contributions
            ],
            "node_kinds": sorted({node.kind.value for node in compiled.proof_graph.nodes}),
            "edge_roles": sorted({edge.role.value for edge in compiled.proof_graph.edges}),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _answer_plan(input_payload: dict[str, Any]) -> AnswerPlan:
    return AnswerPlan(
        fulfillment=(
            FactFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                render_output_id="answer_1",
            ),
        ),
        relations=(_relation(input_payload),),
        operations=(
            Operation(
                id="project",
                spec=ProjectSpec(
                    input_relation="source_1",
                    fields=(ProjectField(source="id", output="id"),),
                ),
                output_relation="result",
            ),
        ),
        render_spec=RenderSpec(
            relation_outputs=(
                RenderRelationOutput(
                    id="answer_1",
                    relation_id="result",
                    field_id="id",
                ),
            ),
        ),
    )


def _relation(input_payload: dict[str, Any]) -> Relation:
    return Relation(
        id="source_1",
        source=RelationSource(
            kind=SourceKind.API_READ,
            read_id="read_1",
            row_source_id="rs_read_1_root",
            population_choices=tuple(
                _population_choice(item)
                for item in input_payload.get("population_choices") or ()
            ),
        ),
        fields=(RelationField(field_id="id", roles=()),),
    )


def _population_choice(item: dict[str, Any]) -> RelationSourcePopulationChoice:
    return RelationSourcePopulationChoice(
        controller_kind=PopulationChoiceControllerKind(str(item["controller_kind"])),
        controller_id=str(item["controller_id"]),
        field_id=str(item["field_id"]),
        included_values=tuple(str(value) for value in item["included_values"]),
        excluded_values=tuple(str(value) for value in item.get("excluded_values") or ()),
        proof_refs=tuple(str(ref) for ref in item.get("proof_refs") or ()),
        review_scope_decisions=tuple(
            _review_scope_decision(decision)
            for decision in item.get("review_scope_decisions") or ()
        ),
    )


def _review_scope_decision(item: dict[str, Any]) -> RelationSourceReviewScopeDecision:
    return RelationSourceReviewScopeDecision(
        membership_test_id=str(item["membership_test_id"]),
        decision=ReviewScopeDecisionKind(str(item["decision"])),
        axis_kind=str(item["axis_kind"]),
        axis_id=str(item["axis_id"]),
        owner_surface_ids=tuple(
            str(owner) for owner in item.get("owner_surface_ids") or ()
        ),
        proof_refs=tuple(str(ref) for ref in item.get("proof_refs") or ()),
    )


def _row_source_catalog(input_payload: dict[str, Any]) -> RowSourceCatalog:
    return RowSourceCatalog(
        sources=(
            RowSource(
                id="rs_read_1_root",
                kind=RowSourceKind.API_READ,
                label="read_1 root",
                read_id="read_1",
                row_path_id="root",
                row_path="root",
            ),
        )
    )
