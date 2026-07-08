from __future__ import annotations

import pytest

from fervis.lineage.enums import ContributionOrigin, ProofEdgeRole, ProofNodeKind
from fervis.lookup.relation_catalog import (
    CatalogField,
    EndpointRead,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.fact_plan.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    RelationSourceAppliedFilter,
    RelationSourceRowFilter,
    SourceKind,
)
from fervis.lookup.plan_execution.compiled_execution import (
    ExecutionProofContribution,
    ExecutionProofEdge,
    ExecutionProofGraph,
    ExecutionProofNode,
    compile_fact_execution,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.memory.available_values import active_memory_operation_values
from fervis.lookup.memory.projection import LookupMemory, MemoryValue
from fervis.lookup.fact_plan.fact_plan import AnswerPlan
from fervis.lookup.fact_plan.operations import ComputeSpec, Operation
from fervis.lookup.fact_plan.values import (
    FactValue,
    ScalarInputUse,
    ValueFilterOperator,
    ValueKind,
    ValueUse,
)


def test_active_memory_scalar_input_keeps_memory_and_prior_proof_refs() -> None:
    memory_value_id = "memory_artifact_1.value.total"
    memory = LookupMemory(
        values=(
            MemoryValue(
                id=memory_value_id,
                value="14",
                value_type="number",
                proof_refs=("prior_step:answer_output",),
            ),
        ),
    )

    compiled = compile_fact_execution(
        answer=AnswerPlan(
            value_uses=(
                ValueUse(
                    id="use_total",
                    value_id=memory_value_id,
                    target=ScalarInputUse(
                        operation_id="compare",
                        input_id="total",
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="compare",
                    spec=ComputeSpec(
                        expression="total",
                        scalar_inputs=("total",),
                    ),
                ),
            ),
        ),
        catalog=None,
        row_sources=(),
        available_values=active_memory_operation_values(
            memory=memory,
            active_memory_ids=frozenset({memory_value_id}),
        ),
    )

    contribution = next(
        item
        for item in compiled.proof_graph.contributions
        if item.origin is ContributionOrigin.CONTEXTUAL
    )
    assert contribution.proof_refs == (
        "memory:memory_artifact_1.value.total",
        "prior_step:answer_output",
    )


def test_executed_memory_relation_adds_current_run_contribution() -> None:
    graph = ExecutionProofGraph(
        nodes=(
            ExecutionProofNode(
                id="relation:memory_artifact_1.relation.rows",
                kind=ProofNodeKind.RELATION,
            ),
            ExecutionProofNode(
                id="answer_output:fact_1:answer_1",
                kind=ProofNodeKind.ANSWER_OUTPUT,
            ),
        ),
        edges=(
            ExecutionProofEdge(
                source="relation:memory_artifact_1.relation.rows",
                target="answer_output:fact_1:answer_1",
                role=ProofEdgeRole.PRODUCES,
            ),
        ),
    )

    graph = graph.with_executed_relations(
        (
            RelationRows(
                id="memory_artifact_1.relation.rows",
                completeness=CompletenessProof(
                    status=CompletenessStatus.COMPLETE,
                    source_kind=CompletenessSourceKind.MEMORY_READ,
                    proof_refs=(
                        "memory:memory_artifact_1.relation.rows",
                        "prior_step:relation",
                    ),
                ),
            ),
        )
    )

    assert graph.contributions == (
        ExecutionProofContribution(
            origin=ContributionOrigin.CONTEXTUAL,
            label="memory_artifact_1.relation.rows",
            node_refs=("relation:memory_artifact_1.relation.rows",),
            proof_refs=("memory:memory_artifact_1.relation.rows", "prior_step:relation"),
        ),
    )


def test_compile_merges_equivalent_applied_and_concrete_row_filters() -> None:
    catalog = _sales_catalog()
    compiled = compile_fact_execution(
        answer=AnswerPlan(
            relations=(
                Relation(
                    id="sales",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_sales",
                        applied_filters=(
                            RelationSourceAppliedFilter(
                                predicate_field_ids=("staff_id",),
                                known_input_id="qi_staff",
                                value_kind=ValueKind.IDENTITY.value,
                                identity_type="staff",
                            ),
                        ),
                        row_filters=(
                            RelationSourceRowFilter(
                                field_id="staff_id",
                                operator=ValueFilterOperator.EQUALS.value,
                                values=("staff_1",),
                                proof_refs=("backend:closed_key",),
                            ),
                        ),
                    ),
                    fields=(
                        RelationField(
                            field_id="staff_id",
                            roles=(FieldBindingRole.PREDICATE,),
                        ),
                    ),
                ),
            ),
        ),
        catalog=catalog,
        row_sources=(),
        available_values=(
            FactValue.identity(
                id="value_staff",
                identity_type="staff",
                identity_field="staff_id",
                value="staff_1",
                proof_refs=("known_input:qi_staff",),
            ),
        ),
    )

    row_filter_nodes = tuple(
        node
        for node in compiled.proof_graph.nodes
        if node.kind is ProofNodeKind.ROW_FILTER
    )

    assert row_filter_nodes == (
        ExecutionProofNode(
            id="row_filter:sales:staff_id",
            kind=ProofNodeKind.ROW_FILTER,
            proof_refs=("backend:closed_key", "known_input:qi_staff"),
            label="staff_id=staff_1",
            value="staff_1",
            operator=ValueFilterOperator.EQUALS.value,
        ),
    )


def test_compile_rejects_conflicting_same_field_row_filters() -> None:
    catalog = _sales_catalog()
    with pytest.raises(VerificationError, match="conflicting row filters"):
        compile_fact_execution(
            answer=AnswerPlan(
                relations=(
                    Relation(
                        id="sales",
                        source=RelationSource(
                            kind=SourceKind.API_READ,
                            read_id="list_sales",
                            row_filters=(
                                RelationSourceRowFilter(
                                    field_id="staff_id",
                                    operator=ValueFilterOperator.EQUALS.value,
                                    values=("staff_1",),
                                    proof_refs=("known_input:qi_staff_1",),
                                ),
                                RelationSourceRowFilter(
                                    field_id="staff_id",
                                    operator=ValueFilterOperator.EQUALS.value,
                                    values=("staff_2",),
                                    proof_refs=("known_input:qi_staff_2",),
                                ),
                            ),
                        ),
                        fields=(
                            RelationField(
                                field_id="staff_id",
                                roles=(FieldBindingRole.PREDICATE,),
                            ),
                        ),
                    ),
                ),
            ),
            catalog=catalog,
            row_sources=(),
        )


def _sales_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_sales",
                endpoint_name="list_sales",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                ),
            ),
        ),
    )
