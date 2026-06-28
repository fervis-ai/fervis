from __future__ import annotations

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_lookup_cutover_persists_fact_addresses_with_non_rendered_identity_fields():
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
                            roles=(FieldBindingRole.IDENTITY,),
                        ),
                        RelationField(
                            field_id="staff_name",
                            roles=(FieldBindingRole.OUTPUT,),
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
                            ProjectField(source="staff_name"),
                            ProjectField(source="sales_total"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="staff_name",
                        relation_id="answer_rows",
                        field_id="staff_name",
                    ),
                    RenderRelationOutput(
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
                        identity=IdentityMetadata(
                            entity_ref="staff",
                            primary_key=True,
                            display_fields=("field.staff_name",),
                        ),
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
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            )
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
    assert result.answer == "Alice: 12000.00"
    assert result.rendered_fact.rows == (  # type: ignore[union-attr]
        {"answer_1": "Alice", "answer_2": "12000.00"},
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
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
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
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
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


def test_lookup_cutover_renders_selected_compute_scalar():
    question_contract = _question_contract_for(
        "rf_answer",
        description="remaining amount metric total",
        subject_text="we need",
        binding_target_ids=("remaining",),
        known_inputs=(
            RequestedFactKnownInput(
                id="current_value",
                kind=KnownInputKind.NUMBER,
                source=KnownInputSource.QUESTION_CONTEXT,
                text="35",
                numeric_value=35,
            ),
            RequestedFactKnownInput(
                id="target_value",
                kind=KnownInputKind.NUMBER,
                source=KnownInputSource.QUESTION_CONTEXT,
                text="100",
                numeric_value=100,
            ),
        ),
    )
    result = run_lookup_question(
        LookupRequest(question="How much more do we need from 35 to reach 100?"),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort(
                {
                    "metric_read": {
                        "data": [
                            {
                                "location_name": "Location Alpha",
                                "metric_total": "35.00",
                            }
                        ]
                    }
                }
            ),
            planner_model_port=_RawPlannerPort(
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "rf_answer",
                                "answer_output_ids": ["remaining"],
                                "pattern": "computed_scalar",
                                "source": {"kind": "values"},
                                "scalar_inputs": [
                                    {
                                        "input_id": "target",
                                        "value_id": "target_value",
                                    },
                                    {
                                        "input_id": "current",
                                        "value_id": "current_value",
                                    },
                                ],
                                "expression": "target - current",
                                "output": {
                                    "scalar_id": "remaining_total",
                                    "label": "remaining",
                                },
                            }
                        ],
                    }
                },
                question_contract=question_contract,
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact is not None
    assert rendered_fact_payload(result.rendered_fact)["scalars"] == {"remaining": "65"}
    assert "remaining: 65" in result.answer


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
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
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
