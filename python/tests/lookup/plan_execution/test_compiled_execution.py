from __future__ import annotations

from fervis.lineage.enums import ContributionOrigin, ProofEdgeRole, ProofNodeKind
from fervis.lookup.plan_execution.compiled_execution import (
    ExecutionProofContribution,
    ExecutionProofEdge,
    ExecutionProofGraph,
    ExecutionProofNode,
    compile_fact_execution,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.memory.available_values import active_memory_operation_values
from fervis.lookup.memory.projection import LookupMemory, MemoryValue
from fervis.lookup.fact_plan.fact_plan import AnswerPlan
from fervis.lookup.fact_plan.values import ScalarInputUse, ValueUse


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
