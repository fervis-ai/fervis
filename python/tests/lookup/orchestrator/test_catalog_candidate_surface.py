from __future__ import annotations

from fervis.lookup.fact_plan.fact_plan import BlockedFactField

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_lookup_cutover_projects_selected_endpoint_candidates_to_fact_planning():
    sales_read_id = "sales"
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_response(
                subject="sales amount",
                parts=("sales amount",),
            ),
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1"],
                            "pattern": "aggregate_scalar",
                            "source": {
                                "kind": "read",
                                "read_id": sales_read_id,
                            },
                            "metric": {
                                "kind": "aggregate_field",
                                "function": "sum",
                                "field_id": "amount",
                                "label": "total",
                            },
                        },
                    ],
                }
            },
        }
    )
    catalog = _catalog(
        EndpointRead(
            id="sales",
            endpoint_name="list_sale_list",
            resource_names=("sales",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.data.amount",
                    path="data.amount",
                    row_path_id="data",
                    type="decimal",
                    metadata={
                        "name": "amount",
                        "description": "Sale amount for one completed sale.",
                    },
                ),
            ),
            source_metadata={
                "description": "Returns sale rows. Each row is one completed sale."
            },
        ),
        EndpointRead(
            id="payments",
            endpoint_name="list_payment_list",
            resource_names=("payments",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.data.amount",
                    path="data.amount",
                    row_path_id="data",
                    type="decimal",
                    metadata={
                        "name": "amount",
                        "description": "Payment amount received from a buyer.",
                    },
                ),
            ),
            source_metadata={
                "description": "Returns payment rows. Each row is one buyer payment."
            },
        ),
    )
    data_access = _DataAccessPort(
        {
            "list_sale_list": {"data": [{"amount": "4321.00"}]},
            "list_payment_list": {"data": [{"amount": "9000.00"}]},
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is the total sales amount?",
            run_id="run_fact_plan_candidate_surface",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert planner.tool_names == [
        "submit_answer_request_contract",
        "submit_query_enrichment",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
        "submit_pattern_fact_plan",
    ]
    source_binding_prompt_text = _source_binding_prompt(planner)
    assert "Each row is one completed sale" in source_binding_prompt_text
    assert '<field name="amount"' in source_binding_prompt_text
    assert 'read="sales"' in source_binding_prompt_text
    assert result.status == "COMPLETED", result
    assert result.answer == "4321.00"
    assert data_access.requests == [{"endpointName": "list_sale_list", "args": {}}]


def test_lookup_cutover_selected_endpoint_surface_is_not_limited_to_matched_fields():
    planner = _PromptSurfacePlannerPort()
    catalog = _catalog(
        EndpointRead(
            id="sales",
            endpoint_name="list_sale_list",
            resource_names=("sales",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.sale_id",
                    path="data.sale_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.staff_name",
                    path="data.staff_name",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="field.amount",
                    path="data.amount",
                    row_path_id="data",
                    type="decimal",
                ),
            ),
            source_metadata={
                "description": (
                    "Returns sale rows for salespeople, sales, revenue, and "
                    "person-level sales analysis."
                )
            },
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        )
    )

    result = run_lookup_question(
        LookupRequest(
            question="List salespeople with sales.",
            run_id="run_selected_endpoint_complete_surface",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_sale_list": {
                        "data": [
                            {
                                "sale_id": "sale_1",
                                "staff_name": "Alice Smith",
                                "amount": "4321.00",
                            }
                        ]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "Alice Smith"
    assert planner.fact_plan_field_id == "staff_name"
    assert '"field_id": "staff_name"' in _fact_plan_prompt(planner)


def test_lookup_cutover_selects_catalog_reads_per_requested_fact():
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_sales",
                description="sales amount",
                answer_outputs=(RequestedFactAnswerOutput(id="sales_amount"),),
            ),
            RequestedFact(
                id="rf_staff",
                description="staff last name",
                answer_outputs=(RequestedFactAnswerOutput(id="staff_last_name"),),
            ),
        )
    )
    planner = _RawPlannerPort(
        {
            "outcome": {
                "kind": "fact_plan",
                "answers": [
                    {
                        "requested_fact_id": "rf_sales",
                        "answer_output_ids": ["sales_amount"],
                        "pattern": "list_rows",
                        "source": {"kind": "read", "read_id": "sales_read"},
                        "output_fields": [{"field_id": "amount"}],
                    },
                    {
                        "requested_fact_id": "rf_staff",
                        "answer_output_ids": ["staff_last_name"],
                        "pattern": "list_rows",
                        "source": {"kind": "read", "read_id": "staff_read"},
                        "output_fields": [{"field_id": "last_name"}],
                    },
                ],
            }
        },
        question_contract=question_contract,
    )
    catalog = _catalog(
        EndpointRead(
            id="sales_read",
            endpoint_name="list_sales",
            resource_names=("sales read",),
            row_paths=(
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
            facts=(CatalogFact(ref="sales.amount", field_ref="field.amount"),),
        ),
        EndpointRead(
            id="staff_read",
            endpoint_name="get_staff",
            resource_names=("staff read",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.last_name",
                    path="data.last_name",
                    row_path_id="data",
                    type="string",
                ),
            ),
            facts=(CatalogFact(ref="staff.last_name", field_ref="field.last_name"),),
        ),
    )
    data_access = _DataAccessPort(
        {
            "list_sales": {"data": [{"amount": "125.00"}]},
            "get_staff": {"data": [{"last_name": "Smith"}]},
        },
    )

    result = run_lookup_question(
        LookupRequest(
            question="What are the sales amount and staff last name?",
            run_id="run_per_requested_fact_selection",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    query_enrichment_prompt = _query_enrichment_prompt(planner)
    assert '"requested_fact_id": "fact_1"' in query_enrichment_prompt
    assert '"requested_fact_id": "fact_2"' in query_enrichment_prompt
    source_binding_prompt = _source_binding_prompt(planner)
    assert 'read="sales_read"' in source_binding_prompt
    assert 'read="staff_read"' in source_binding_prompt
    assert data_access.requests == [
        {"endpointName": "list_sales", "args": {}},
        {"endpointName": "get_staff", "args": {}},
    ]


def test_lookup_cutover_classifies_blocked_requested_fact_as_impossible_without_execution():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_restricted_secret",
                    basis=BlockedFactBasis.POLICY_ACCESS,
                    evidence_refs=(
                        "policy:secret_full_token",
                        "secrets",
                        "secrets.masked_token",
                    ),
                    reviewed_read_ids=("secrets",),
                    nearest_fields=(
                        BlockedFactField(
                            read_id="secrets",
                            field_id="masked_token",
                        ),
                    ),
                    explanation="Secret tokens are not readable.",
                ),
            )
        )
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_restricted_secret",
                description="restricted secret tokens",
                required_for="secret token",
                answer_outputs=(RequestedFactAnswerOutput(id="secret_token"),),
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="secrets",
                    endpoint_name="list_secrets",
                    resource_names=("secrets",),
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.masked_token",
                            path="data.masked_token",
                            row_path_id="data",
                            type="string",
                        ),
                    ),
                    facts=(
                        CatalogFact(
                            ref="secret.full_token",
                            availability=CatalogFactAvailability.POLICY_BLOCKED,
                            field_ref="field.masked_token",
                            read_id="secrets",
                            proof_refs=("policy:secret_full_token",),
                        ),
                    ),
                    source_metadata={"description": "Secret tokens are not readable."},
                ),
            )
        ),
        responses={},
        question_contract=question_contract,
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="secrets",
            ),
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What are the restricted secret tokens?",
            run_id="run_impossible",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED"
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    details = result.rendered_fact.details  # type: ignore[union-attr]
    blocked = details["blockedRequirements"][0]  # type: ignore[index]
    assert blocked["kind"] == "policy"
    assert blocked["requiredFor"] == "restricted secret tokens"
    assert blocked["reviewedReadIds"] == ["secrets"]
    assert blocked["nearestFields"] == [
        {"readId": "secrets", "fieldId": "masked_token"}
    ]
    assert blocked["proofRefs"] == [
        "policy:secret_full_token",
        "secrets",
        "secrets.masked_token",
    ]
    assert ports.data_access_port.requests == []
