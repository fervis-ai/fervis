from __future__ import annotations

from decimal import Decimal

from fervis.lookup.answer_rendering import render_fact_result, rendered_fact_payload
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ResultProjection,
)
from fervis.lookup.outcomes.model import AnswerResult, FactResult
from fervis.lookup.plan_execution.relations import RelationRows

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_renderer_exposes_answer_and_ranking_outputs_without_support_fields():
    result = FactResult(
        outcome=AnswerResult(
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="staff",
                        relation_id="ranked_staff",
                        field_id="staff_id",
                        role="answer_value",
                    ),
                    RelationResultOutput(
                        id="compensation",
                        relation_id="ranked_staff",
                        field_id="calculated_pay",
                        role="ranking_metric",
                    ),
                    RelationResultOutput(
                        id="payment_status",
                        relation_id="ranked_staff",
                        field_id="payment_status",
                        role="support",
                    ),
                )
            ),
            relations=(
                RelationRows(
                    id="ranked_staff",
                    rows=(
                        {
                            "staff_id": "staff-3",
                            "calculated_pay": Decimal("1200.00"),
                            "payment_status": "PARTIALLY_PAID",
                        },
                    ),
                ),
            ),
        )
    )

    rendered = render_fact_result(result)

    assert rendered.rows == (
        {"staff": "staff-3", "compensation": Decimal("1200.00")},
    )
    assert rendered_fact_payload(rendered)["renderOutputs"] == [
        {"key": "staff", "role": "answer_value"},
        {"key": "compensation", "role": "ranking_metric"},
    ]


def test_lookup_cutover_persists_fact_addresses_with_typed_entity_results():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="staff_sales_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="staff_sales_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="staff_id",
                            roles=(
                                FieldBindingRole.IDENTITY,
                                FieldBindingRole.OUTPUT,
                            ),
                        ),
                        RelationField(
                            field_id="sales_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="staff_sales_rows",
                        fields=(
                            ProjectField(source="staff_id"),
                            ProjectField(source="sales_total"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="staff",
                        relation_id="answer_rows",
                        entity_key=EntityKeyProjection(
                            entity_kind="staff",
                            key_id="primary_key",
                            components=(
                                EntityKeyProjectionComponent(
                                    component_id="staff_id",
                                    field_id="staff_id",
                                ),
                            ),
                        ),
                    ),
                    RelationResultOutput(
                        id="sales_total",
                        relation_id="answer_rows",
                        field_id="sales_total",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_catalog(
            EndpointRead(
                id="staff_sales_read",
                endpoint_name="staff_sales_read",
                resource_names=("staff sales read",),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.sales_total",
                        path="data.sales_total",
                        row_path_id="data",
                        type="number",
                    ),
                ),
                entity_references=(
                    EntityReference(
                        id="staff_reference",
                        target_entity_kind="staff",
                        target_key_id="primary_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="staff_id",
                                local_field_ref="field.staff_id",
                            ),
                        ),
                        context_field_refs=("field.staff_name",),
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
            EndpointRead(
                id="staff_read",
                endpoint_name="list_staff_list",
                resource_names=("staff",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="staff.field.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="primary_key",
                        entity_kind="staff",
                        components=(
                            CandidateKeyComponent(
                                id="staff_id",
                                field_ref="staff.field.staff_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
            ),
        ),
        responses={
            "staff_sales_read": {
                "data": [
                    {
                        "staff_id": "staff-1",
                        "staff_name": "Alice",
                        "sales_total": "12000.00",
                    }
                ]
            }
        },
    )

    result = run_lookup_question(
        LookupRequest(
            question="How much did the staff member sell?",
            run_id="run_staff_identity",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    addresses = {item["address"]: item for item in result.fact_addresses}
    assert result.rendered_fact.rows == (  # type: ignore[union-attr]
        {
            "answer_1": {
                "entityKind": "staff",
                "keyId": "primary_key",
                "components": {"staff_id": "staff-1"},
            },
            "answer_2": Decimal("12000.00"),
        },
    )
    assert addresses["row.answer_1_rows.1"]["identity"] == {"staff_id": "staff-1"}
    assert "staff_id" not in addresses["row.answer_1_rows.1"]["values"]


def test_lookup_cutover_fact_addresses_expose_only_rendered_answer_fields():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="customer_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="customer_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="customer_name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="private_email",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="customer_rows",
                        fields=(ProjectField(source="customer_name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="customer_name",
                        relation_id="answer_rows",
                        field_id="customer_name",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_catalog(
            EndpointRead(
                id="customer_read",
                endpoint_name="customer_read",
                resource_names=("customer read",),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.customer_name",
                        path="data.customer_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.private_email",
                        path="data.private_email",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            )
        ),
        responses={
            "customer_read": {
                "data": [
                    {
                        "customer_name": "Customer Alpha",
                        "private_email": "alpha@example.test",
                    }
                ]
            }
        },
    )

    result = run_lookup_question(
        LookupRequest(
            question="Which customer matched the query?",
            run_id="run_public_fact_addresses",
        ),
        ports,
    )

    serialized_addresses = json.dumps(result.fact_addresses, sort_keys=True)
    assert result.status == "COMPLETED", result
    assert result.answer == "Customer Alpha"
    assert "Customer Alpha" in serialized_addresses
    assert "private_email" not in serialized_addresses
    assert "alpha@example.test" not in serialized_addresses
    assert "customer_rows" not in serialized_addresses


def test_lookup_cutover_result_data_exposes_only_rendered_answer_fields():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="customer_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="customer_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="customer_name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="customer_rows",
                        fields=(ProjectField(source="customer_name"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="customer_name",
                        relation_id="answer_rows",
                        field_id="customer_name",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_catalog(
            EndpointRead(
                id="customer_read",
                endpoint_name="customer_read",
                resource_names=("customer read",),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.customer_name",
                        path="data.customer_name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            )
        ),
        responses={"customer_read": {"data": [{"customer_name": "Customer Alpha"}]}},
    )

    result = run_lookup_question(
        LookupRequest(
            question="Which customer matched the query?",
            run_id="run_public_result_data",
        ),
        ports,
    )
    payload = rendered_fact_payload(result.rendered_fact)  # type: ignore[arg-type]

    assert result.status == "COMPLETED", result
    assert payload == {
        "kind": "answer",
        "rows": [{"answer_1": "Customer Alpha"}],
        "scalars": {},
        "message": "",
        "details": {},
        "proofRefs": ["read:customer_read", "answer_1_rows_project"],
        "renderOutputs": [{"key": "answer_1", "role": "answer_value"}],
    }


def test_lookup_cutover_aggregate_result_payload_is_json_safe():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="sales_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="amount",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="sum_sales",
                    spec=AggregateSpec(
                        input_relation="sales_rows",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.SUM,
                                input_field="amount",
                                output_field="total_sales",
                            ),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="total_sales",
                        relation_id="answer_rows",
                        field_id="total_sales",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_catalog(
            EndpointRead(
                id="sales_read",
                endpoint_name="sales_read",
                resource_names=("sales read",),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="number",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            )
        ),
        responses={
            "sales_read": {
                "data": [
                    {"amount": "10.25"},
                    {"amount": "20.75"},
                ]
            }
        },
    )

    result = run_lookup_question(
        LookupRequest(question="How much sales?", run_id="run_json_safe_aggregate"),
        ports,
    )

    assert result.status == "COMPLETED"
    payload = rendered_fact_payload(result.rendered_fact)  # type: ignore[arg-type]
    assert payload["rows"] == [{"amount": "31.00"}]
    json.dumps(payload)
