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
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    RelationSourceAppliedFilter,
    RelationSourceRowFilter,
    SourceKind,
)
from fervis.lookup.answer_program.instantiation import (
    ExecutionProofContribution,
    ExecutionProofEdge,
    ExecutionProofGraph,
    ExecutionProofNode,
    _materialize_execution,
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
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.operations import (
    ComputeBinary,
    ComputeBinaryOperator,
    ComputeSpec,
    Operation,
)
from fervis.lookup.answer_program.values import (
    BindingSet,
    ConstantRef,
    FactValue,
    LiteralType,
    ValueFilterOperator,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)


def _constant_ref(value: FactValue, *, constant_id: str) -> ConstantRef:
    return ConstantRef(
        constant_id=constant_id,
        version_ref="test@1",
        value=value,
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

    memory_value = active_memory_operation_values(
        memory=memory,
        active_memory_ids=frozenset({memory_value_id}),
    )[0]
    compiled = _materialize_execution(
        answer=AnswerProgram(
            operations=(
                Operation(
                    id="compare",
                    spec=ComputeSpec(
                        expression=_constant_ref(
                            memory_value,
                            constant_id=f"context.{memory_value_id}",
                        ),
                    ),
                ),
            ),
        ),
        bindings=BindingSet(),
        catalog=None,
        row_sources=(),
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


def test_repeated_compute_origin_has_one_operation_input_node() -> None:
    value = FactValue.literal(
        id="amount",
        literal_type=LiteralType.NUMBER,
        value="10",
        proof_refs=("known_input:amount",),
    )
    expression = _constant_ref(value, constant_id="question.amount")

    compiled = _materialize_execution(
        answer=AnswerProgram(
            operations=(
                Operation(
                    id="double",
                    spec=ComputeSpec(
                        expression=ComputeBinary(
                            operator=ComputeBinaryOperator.ADD,
                            left=expression,
                            right=expression,
                        ),
                        output_scalar="total",
                    ),
                ),
            ),
        ),
        bindings=BindingSet(),
        catalog=None,
        row_sources=(),
    )

    assert tuple(
        node.id
        for node in compiled.proof_graph.nodes
        if node.kind is ProofNodeKind.OPERATION_INPUT
    ) == ("operation_input:double:constant:question.amount@test@1",)


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
            proof_refs=(
                "memory:memory_artifact_1.relation.rows",
                "prior_step:relation",
            ),
        ),
    )


def test_compile_merges_equivalent_applied_and_concrete_row_filters() -> None:
    catalog = _sales_catalog()
    staff_value = FactValue.identity(
        id="value_staff",
        entity_kind="staff",
        key_id="primary_key",
        key_component_id="staff_id",
        value="staff_1",
        display_value="Nadia Wanjiku",
        proof_refs=("known_input:qi_staff",),
    )
    compiled = _materialize_execution(
        answer=AnswerProgram(
            fact_template=(
                RequestedFact(
                    id="fact_1",
                    description="sales by Nadia",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                        ),
                    ),
                    known_inputs=(
                        RequestedFactLiteralInput(
                            id="qi_staff",
                            source=KnownInputSource.QUESTION_CONTEXT,
                            role=LiteralInputRole.REFERENCE_VALUE,
                            text="Nadia",
                            resolved_value_text="Nadia",
                        ),
                    ),
                    input_refs=("qi_staff",),
                ),
            ),
            relations=(
                Relation(
                    id="sales",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_sales",
                        applied_filters=(
                            RelationSourceAppliedFilter(
                                predicate_field_ids=("staff_id",),
                                value_expr=_constant_ref(
                                    staff_value,
                                    constant_id="question.qi_staff",
                                ),
                            ),
                        ),
                        row_filters=(
                            RelationSourceRowFilter(
                                field_id="staff_id",
                                operator=ValueFilterOperator.EQUALS.value,
                                value_expr=_constant_ref(
                                    FactValue.string_set(
                                        id="closed_key_staff",
                                        values=("staff_1",),
                                        proof_refs=("backend:closed_key",),
                                    ),
                                    constant_id="closed-key.staff",
                                ),
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
        bindings=BindingSet(),
        catalog=catalog,
        row_sources=(),
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
    explicit_labels = tuple(
        contribution.label
        for contribution in compiled.proof_graph.contributions
        if contribution.origin is ContributionOrigin.EXPLICIT
    )
    assert explicit_labels == ("Nadia",)


def test_compile_rejects_conflicting_same_field_row_filters() -> None:
    catalog = _sales_catalog()
    with pytest.raises(VerificationError, match="conflicting row filters"):
        _materialize_execution(
            answer=AnswerProgram(
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
                                    value_expr=_constant_ref(
                                        FactValue.string_set(
                                            id="staff_1",
                                            values=("staff_1",),
                                        ),
                                        constant_id="staff.1",
                                    ),
                                    proof_refs=("known_input:qi_staff_1",),
                                ),
                                RelationSourceRowFilter(
                                    field_id="staff_id",
                                    operator=ValueFilterOperator.EQUALS.value,
                                    value_expr=_constant_ref(
                                        FactValue.string_set(
                                            id="staff_2",
                                            values=("staff_2",),
                                        ),
                                        constant_id="staff.2",
                                    ),
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
            bindings=BindingSet(),
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
