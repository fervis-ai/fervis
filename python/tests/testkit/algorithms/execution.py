from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.answer_program.instantiation import _materialize_execution
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.answer_program.operations import Operation, ProjectField, ProjectSpec
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    PopulationChoiceControllerKind,
    Relation,
    RelationField,
    RelationSource,
    RelationSourcePopulationChoice,
    RelationSourceReviewScopeDecision,
    ReviewScopeDecisionKind,
    SourceKind,
)
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ResultProjection,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
)
from fervis.lookup.answer_program.values import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    FactValue,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRef,
    ParameterRole,
    ParameterValueType,
)
from fervis.lookup.question_contract import RequestedFact, RequestedFactAnswerOutput

from tests.testkit.assertions import subset_mismatches


def run_execution_proof_graph_case(payload: dict[str, Any]) -> list[str]:
    compiled = _materialize_execution(
        answer=_answer_plan(payload["input"]),
        bindings=_population_bindings(payload["input"]),
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
                for node in sorted(
                    compiled.proof_graph.nodes,
                    key=lambda item: item.kind.value != "population_choice",
                )
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "role": edge.role.value,
                }
                for edge in sorted(
                    compiled.proof_graph.edges,
                    key=lambda item: not item.source.startswith("population_choice:"),
                )
            ],
            "contributions": [
                {
                    "origin": item.origin.value,
                    "label": item.label,
                    "node_refs": list(item.node_refs),
                }
                for item in sorted(
                    compiled.proof_graph.contributions,
                    key=lambda contribution: (
                        not contribution.node_refs[0].startswith("population_choice:")
                    ),
                )
            ],
            "node_kinds": sorted(
                {node.kind.value for node in compiled.proof_graph.nodes}
            ),
            "edge_roles": sorted(
                {edge.role.value for edge in compiled.proof_graph.edges}
            ),
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def _answer_plan(input_payload: dict[str, Any]) -> AnswerProgram:
    return AnswerProgram(
        fact_template=(
            RequestedFact(
                id="fact_1",
                description="requested fact",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),
                ),
            ),
        ),
        fulfillment=(
            FactFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                result_output_id="answer_1",
            ),
        ),
        parameters=tuple(
            _population_parameter(item)
            for item in input_payload.get("population_choices") or ()
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
        result_projection=ResultProjection(
            relation_outputs=(
                RelationResultOutput(
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
        fields=(
            RelationField(field_id="id", roles=()),
            *tuple(
                RelationField(
                    field_id=field_id,
                    roles=(FieldBindingRole.PREDICATE,),
                )
                for field_id in dict.fromkeys(
                    str(item["field_id"])
                    for item in input_payload.get("population_choices") or ()
                )
                if field_id != "id"
            ),
        ),
    )


def _population_choice(item: dict[str, Any]) -> RelationSourcePopulationChoice:
    included_values = tuple(str(value) for value in item["included_values"])
    excluded_values = tuple(str(value) for value in item.get("excluded_values") or ())
    proof_refs = tuple(str(ref) for ref in item.get("proof_refs") or ())
    return RelationSourcePopulationChoice(
        controller_kind=PopulationChoiceControllerKind(str(item["controller_kind"])),
        controller_id=str(item["controller_id"]),
        field_id=str(item["field_id"]),
        requested_fact_ids=tuple(
            str(fact_id) for fact_id in item.get("requested_fact_ids") or ("fact_1",)
        ),
        selection_expr=ParameterRef(
            parameter_id=_population_parameter_id(item),
        ),
        allowed_values=(*included_values, *excluded_values),
        proof_refs=proof_refs,
        review_scope_decisions=tuple(
            _review_scope_decision(decision)
            for decision in item.get("review_scope_decisions") or ()
        ),
    )


def _population_parameter(item: dict[str, Any]) -> ParameterDeclaration:
    included_values = tuple(str(value) for value in item["included_values"])
    excluded_values = tuple(str(value) for value in item.get("excluded_values") or ())
    return ParameterDeclaration(
        id=_population_parameter_id(item),
        role=ParameterRole.SEMANTIC_CONTROL,
        value_type=ParameterValueType.STRING_SET,
        allowed_values=(*included_values, *excluded_values),
        semantic_control_ref=str(item["controller_id"]),
    )


def _population_bindings(input_payload: dict[str, Any]) -> BindingSet:
    return BindingSet.from_bindings(
        tuple(
            ParameterBinding(
                parameter_id=_population_parameter_id(item),
                value=FactValue.string_set(
                    id=f"fixture.population.{item['controller_id']}",
                    values=tuple(str(value) for value in item["included_values"]),
                    proof_refs=tuple(str(ref) for ref in item.get("proof_refs") or ()),
                ),
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.SEMANTIC_CHOICE,
                ),
            )
            for item in input_payload.get("population_choices") or ()
        )
    )


def _population_parameter_id(item: dict[str, Any]) -> str:
    return f"semantic.{item['controller_id']}"


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
