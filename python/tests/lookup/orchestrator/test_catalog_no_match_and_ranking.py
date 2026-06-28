from __future__ import annotations

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_lookup_accepts_zero_catalog_match_as_impossible_proof():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="card_number",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=("catalog_selection:card_number",),
                ),
            )
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_catalog(
            EndpointRead(
                id="inventory_read",
                endpoint_name="list_inventory",
                resource_names=("inventory read",),
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.quantity",
                        path="data.quantity",
                        row_path_id="data",
                        type="number",
                    ),
                ),
            )
        ),
        responses={},
        question_contract=_question_contract_for(
            "card_number",
            description="full payment card number",
            binding_target_ids=("card_number",),
        ),
        query_enrichment=_query_enrichment_payload(),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is the full payment card number?",
            run_id="run_zero_catalog_selection",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert ports.planner_model_port.calls == 4
    assert result.fact_result is not None
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    blocked = result.fact_result.outcome.blocked_requirements[0]
    assert blocked.kind == BlockedRequirementKind.OPERATION_NOT_SUPPORTED_BY_CATALOG
    assert blocked.requested_fact_id == "fact_1"
    assert blocked.proof_refs == ("catalog_selection:fact_1",)


def test_lookup_cutover_selects_ranked_catalog_before_fact_planning():
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="rf_total",
            answer_output_ids=("total",),
            read_id="metric_read",
            output_fields=({"field_id": "metric_total", "label": "total"},),
        ),
        question_contract=_question_contract_for(
            "rf_total",
            description="metric total",
            binding_target_ids=("total",),
        ),
        query_enrichment=_query_enrichment_payload(
            ("metric",),
        ),
    )
    catalog = _catalog(
        _metric_read("metric_read"),
        _inventory_read("inventory_read"),
    )
    data_access = _DataAccessPort(
        {"metric_read": {"data": [{"metric_total": "125.00"}]}}
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is the metric total?",
            run_id="run_catalog_selection",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "125.00"
    fact_plan_prompt = _source_binding_prompt(planner)
    assert '"read_id": "metric_read"' in fact_plan_prompt
    assert "inventory_read" not in fact_plan_prompt
    assert data_access.requests == [{"endpointName": "metric_read", "args": {}}]


def test_lookup_default_catalog_selection_caps_positive_reads_at_five():
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="rf_total",
            answer_output_ids=("total",),
            read_id="metric_read_1",
            output_fields=({"field_id": "metric_total", "label": "total"},),
        ),
        question_contract=_question_contract_for(
            "rf_total",
            description="metric total",
            binding_target_ids=("total",),
        ),
        query_enrichment=_query_enrichment_payload(
            ("metric",),
        ),
        read_eligibility_retention_specs=tuple(
            ReadEligibilityRetentionSpec(
                requested_fact_id="rf_total",
                read_id=f"metric_read_{index}",
                answer_value_fields=("metric_total",),
            )
            for index in range(1, 7)
        ),
    )
    catalog = _catalog(*(_metric_read(f"metric_read_{index}") for index in range(1, 7)))
    data_access = _DataAccessPort(
        {"metric_read_1": {"data": [{"metric_total": "125.00"}]}}
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is the metric total?",
            run_id="run_default_catalog_selection_limit",
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    read_eligibility_prompt = next(
        prompt for prompt in planner.prompts if "Candidate API reads:" in prompt
    )
    assert 'read="metric_read_1"' in read_eligibility_prompt
    assert 'read="metric_read_5"' in read_eligibility_prompt
    assert 'read="metric_read_6"' in read_eligibility_prompt


def test_lookup_cutover_uses_query_enrichment_resource_name_for_selection():
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="rf_revenue_total",
            answer_output_ids=("revenue_total",),
            read_id="orders_read",
            output_fields=({"field_id": "total", "label": "revenue_total"},),
        ),
        question_contract=_question_contract_for(
            "rf_revenue_total",
            description="order total",
            subject_text="revenue",
            binding_target_ids=("revenue_total",),
        ),
        query_enrichment=_query_enrichment_payload(
            ("order",),
        ),
    )
    catalog = _catalog(
        EndpointRead(
            id="orders_read",
            endpoint_name="list_orders",
            resource_names=("order",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.order_total",
                    path="data.total",
                    row_path_id="data",
                    type="number",
                    metadata={"label": "revenue amount"},
                ),
            ),
            source_metadata={"label": "customer revenue records"},
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
        _inventory_read("inventory_read"),
    )
    data_access = _DataAccessPort({"list_orders": {"data": [{"total": "6400.00"}]}})

    result = run_lookup_question(
        LookupRequest(
            question="How much revenue did we make?",
            run_id="run_query_expansion",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "6400.00"
    fact_plan_prompt = _source_binding_prompt(planner)
    assert '"read_id": "orders_read"' in fact_plan_prompt
    assert "inventory_read" not in fact_plan_prompt


def test_lookup_cutover_uses_query_enrichment_terms_for_catalog_selection():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_response(
                subject="sales amount",
                answer_subject="booked money",
                parts=("sales amount",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sale",)),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="sales_read",
                pattern="aggregate_scalar",
                metric={
                    "kind": "aggregate_field",
                    "function": "sum",
                    "field_id": "amount",
                    "label": "answer_1",
                },
            ),
        }
    )
    catalog = _catalog(
        EndpointRead(
            id="sales_read",
            endpoint_name="list_sale_list",
            resource_names=("sale",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.amount",
                    path="data.amount",
                    row_path_id="data",
                    type="decimal",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
        EndpointRead(
            id="payment_read",
            endpoint_name="list_payment_list",
            resource_names=("payment",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.payment_total",
                    path="data.total",
                    row_path_id="data",
                    type="decimal",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="How much booked money do we have?",
            run_id="run_query_enrichment",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {"list_sale_list": {"data": [{"amount": "7200.00"}]}}
            ),
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
    assert result.status == "COMPLETED", result
    assert result.answer == "7200.00"
    fact_plan_prompt = _source_binding_prompt(planner)
    assert "sales_read" in fact_plan_prompt or "list_sale_list" in fact_plan_prompt
    if "payment_read" in fact_plan_prompt and "sales_read" in fact_plan_prompt:
        assert fact_plan_prompt.index('"read_id": "sales_read"') < (
            fact_plan_prompt.index('"read_id": "payment_read"')
        )


def test_lookup_cutover_uses_query_enrichment_resource_for_catalog_selection():
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="rf_value",
            answer_output_ids=("full_number",),
            read_id="payment_read",
            output_fields=({"field_id": "full_number"},),
        ),
        question_contract=_question_contract_for(
            "rf_value",
            description="payment card full number",
            subject_text="payment card number",
            binding_target_ids=("full_number",),
        ),
        query_enrichment=_query_enrichment_payload(
            ("payment",),
        ),
    )
    catalog = _catalog(
        EndpointRead(
            id="aaa_generic_read",
            endpoint_name="list_generic",
            resource_names=("aaa generic read",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.generic_value",
                    path="data.value",
                    row_path_id="data",
                    type="string",
                ),
            ),
        ),
        EndpointRead(
            id="payment_read",
            endpoint_name="list_payments",
            resource_names=("payment",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.full_number",
                    path="data.full_number",
                    row_path_id="data",
                    type="string",
                ),
            ),
            facts=(
                CatalogFact(
                    ref="payment.card.full_number",
                    field_ref="field.full_number",
                ),
            ),
        ),
    )
    data_access = _DataAccessPort(
        {"list_payments": {"data": [{"full_number": "4111111111111111"}]}}
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is the full payment card number?",
            run_id="run_requested_fact_text_selection",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "4111111111111111"
    fact_plan_prompt = _source_binding_prompt(planner)
    assert 'read="payment_read"' in fact_plan_prompt
    assert 'read="aaa_generic_read"' not in fact_plan_prompt
