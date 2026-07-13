from __future__ import annotations

from decimal import Decimal

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def _primary_key(
    entity_kind: str,
    component_id: str,
    field_ref: str,
) -> tuple[CandidateKey, ...]:
    return (
        CandidateKey(
            id="primary_key",
            entity_kind=entity_kind,
            components=(CandidateKeyComponent(id=component_id, field_ref=field_ref),),
            primary=True,
        ),
    )


def test_lookup_cutover_catalog_selection_weights_catalog_search_terms_over_incidental_fields():
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="rf_active_price_lists",
            answer_output_ids=("active_price_list_count",),
            read_id="price_list_read",
            pattern="aggregate_scalar",
            metric={
                "kind": "count_records",
                "count_basis": {
                    "kind": "row_population",
                    "row_path_id": "data",
                    "row_cardinality": "many",
                },
                "label": "active_price_list_count",
            },
        ),
        question_contract=_question_contract_for(
            "rf_active_price_lists",
            description="count active price lists",
            subject_text="price lists",
            binding_target_ids=("active_price_list_count",),
            answer_output_role="ROW_COUNT",
        ),
        query_enrichment=_query_enrichment_payload(
            ("price list",),
        ),
    )
    catalog = _catalog(
        EndpointRead(
            id="merch_read",
            endpoint_name="list_merch",
            resource_names=("merchandise",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.merch_is_active",
                    path="data.is_active",
                    row_path_id="data",
                    type="boolean",
                ),
                CatalogField(
                    ref="field.merch_starting_price",
                    path="data.starting_price",
                    row_path_id="data",
                    type="decimal",
                ),
                CatalogField(
                    ref="field.merch_review_count",
                    path="data.review_count",
                    row_path_id="data",
                    type="integer",
                ),
            ),
        ),
        EndpointRead(
            id="price_list_read",
            endpoint_name="list_price_lists",
            resource_names=("price list",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.price_list_id",
                    path="data.price_list_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.price_list_is_active",
                    path="data.is_active",
                    row_path_id="data",
                    type="boolean",
                ),
            ),
            candidate_keys=_primary_key(
                "price_list", "price_list_id", "field.price_list_id"
            ),
        ),
    )
    result = run_lookup_question(
        LookupRequest(
            question="How many active price lists are there?",
            run_id="run_resource_weighted_catalog_selection",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_price_lists": {
                        "data": [{"price_list_id": "price-list-1", "is_active": True}]
                    }
                },
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == ({"count": 1},)
    source_binding_prompt = _source_binding_prompt(planner)
    assert 'read="price_list_read"' in source_binding_prompt
    assert 'read="merch_read"' not in source_binding_prompt


def test_lookup_cutover_keeps_collection_read_when_singleton_has_incidental_field_hits():
    planner = _ToolNamePlannerPort(
        {
            "submit_question_contract_outcome": _question_contract_response(
                subject="count active price lists",
                answer_subject="price lists",
                parts=("count active price lists",),
                answer_output_role="ROW_COUNT",
            ),
            "submit_query_enrichment": _query_enrichment_payload(("price list",)),
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
                                "read_id": "list_price_list_list_create",
                            },
                            "metric": {
                                "kind": "count_records",
                                "count_basis": {
                                    "kind": "row_population",
                                    "row_path_id": "data",
                                    "row_cardinality": "many",
                                },
                                "label": "active_price_lists",
                            },
                        }
                    ],
                }
            },
        }
    )
    catalog = _catalog(
        EndpointRead(
            id="list_price_list_active",
            endpoint_name="list_price_list_active",
            resource_names=("price list",),
            row_paths=(RowPath(id="root", path="", cardinality=RowCardinality.ONE),),
            fields=(
                CatalogField(
                    ref="field.active.price_list_id",
                    path="price_list_id",
                    row_path_id="root",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.active.is_active",
                    path="is_active",
                    row_path_id="root",
                    type="boolean",
                ),
                CatalogField(
                    ref="field.active.items",
                    path="items",
                    row_path_id="root",
                    type="array",
                ),
                CatalogField(
                    ref="field.active.items.price_list_item_id",
                    path="items.price_list_item_id",
                    row_path_id="root",
                    type="uuid",
                ),
            ),
            candidate_keys=_primary_key(
                "price_list", "price_list_id", "field.active.price_list_id"
            ),
        ),
        EndpointRead(
            id="list_price_list_list_create",
            endpoint_name="list_price_list_list_create",
            resource_names=("price list",),
            params=(
                CatalogParam(
                    ref="list_price_list_list_create.query.is_active",
                    name="is_active",
                    source=ParamSource.QUERY,
                    type="boolean",
                    required=False,
                ),
            ),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.collection.price_list_id",
                    path="data.price_list_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.collection.is_active",
                    path="data.is_active",
                    row_path_id="data",
                    type="boolean",
                ),
            ),
            candidate_keys=_primary_key(
                "price_list", "price_list_id", "field.collection.price_list_id"
            ),
        ),
        *tuple(
            EndpointRead(
                id=f"incidental_price_read_{index}",
                endpoint_name=f"list_incidental_price_records_{index}",
                resource_names=("incidental record",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref=f"field.incidental_{index}.price",
                        path="data.price",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            )
            for index in range(9)
        ),
    )
    result = run_lookup_question(
        LookupRequest(
            question="How many active price lists are there?",
            run_id="run_collection_read_over_singleton_field_noise",
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_price_list_list_create": {
                        "data": [{"price_list_id": "price-list-1", "is_active": True}]
                    }
                },
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == ({"count": 1},)
    source_binding_prompt = _source_binding_prompt(planner)
    assert 'read="list_price_list_list_create"' in source_binding_prompt
    assert 'cardinality="many"' in source_binding_prompt


def test_lookup_cutover_source_candidates_do_not_mix_root_and_child_result_grains():
    planner = _ToolNamePlannerPort(
        {
            "submit_question_contract_outcome": _question_contract_response(
                subject="active price lists",
                parts=("how many active price lists there are",),
                answer_output_role="ROW_COUNT",
            ),
            "submit_query_enrichment": _query_enrichment_payload(("price list",)),
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
                                "read_id": "list_price_list_active",
                                "param_bindings": [
                                    {"param_id": "is_active", "value": "true"}
                                ],
                            },
                            "metric": {
                                "kind": "count_records",
                                "count_basis": {
                                    "kind": "row_population",
                                    "row_path_id": "data",
                                    "row_cardinality": "many",
                                },
                                "label": "active_price_lists",
                            },
                        }
                    ],
                }
            },
        },
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="list_price_list_active",
                row_path_ids=("root",),
                answer_value_fields=("price_list_id",),
            ),
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="list_price_list_list_create",
                row_path_ids=("data",),
                answer_value_fields=("price_list_id",),
            ),
        ),
    )
    catalog = _catalog(
        EndpointRead(
            id="list_price_list_active",
            endpoint_name="list_price_list_active",
            resource_names=("price list",),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.MANY),
                RowPath(
                    id="items",
                    path="items",
                    parent_path="",
                    cardinality=RowCardinality.MANY,
                ),
            ),
            fields=(
                CatalogField(
                    ref="field.active.price_list_id",
                    path="price_list_id",
                    row_path_id="root",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.active.is_active",
                    path="is_active",
                    row_path_id="root",
                    type="boolean",
                ),
                CatalogField(
                    ref="field.active.items.price_list_item_id",
                    path="items.price_list_item_id",
                    row_path_id="items",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.active.items.merch_name",
                    path="items.merch_name",
                    row_path_id="items",
                    type="string",
                ),
            ),
            candidate_keys=_primary_key(
                "price_list", "price_list_id", "field.active.price_list_id"
            ),
        ),
        EndpointRead(
            id="list_price_list_list_create",
            endpoint_name="list_price_list_list_create",
            resource_names=("price list",),
            params=(
                CatalogParam(
                    ref="list_price_list_list_create.query.is_active",
                    name="is_active",
                    source=ParamSource.QUERY,
                    type="boolean",
                    required=False,
                    choices=("true", "false"),
                ),
            ),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.collection.price_list_id",
                    path="data.price_list_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.collection.is_active",
                    path="data.is_active",
                    row_path_id="data",
                    type="boolean",
                ),
            ),
            candidate_keys=_primary_key(
                "price_list", "price_list_id", "field.collection.price_list_id"
            ),
        ),
    )
    result = run_lookup_question(
        LookupRequest(
            question="How many active price lists are there?",
            run_id="run_active_price_list_grain_candidate_surface",
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_price_list_active": [
                        {
                            "price_list_id": "price-list-1",
                            "is_active": True,
                            "items": [],
                        }
                    ],
                    "list_price_list_list_create": {
                        "data": [{"price_list_id": "price-list-1", "is_active": True}]
                    },
                },
            ),
            planner_model_port=planner,
        ),
    )

    payload = _planner_prompt_json_section(
        _source_binding_prompt(planner),
        label="Candidate evidence sources",
    )
    active_candidates = [
        candidate
        for context in payload["requested_fact_sources"][0]["source_contexts"]
        for candidate in context["source_options"]
        if candidate.get("read_id") == "list_price_list_active"
    ]
    assert len(active_candidates) == 1
    response_rows = active_candidates[0]["response_rows"]
    assert response_rows[0]["path"] == "root"
    assert response_rows[1]["path"] == "items"
    assert response_rows[1]["cardinality"] == "many"


def test_lookup_cutover_source_candidate_can_fulfill_parent_and_child_evidence_from_one_read():
    planner = _ToolNamePlannerPort(
        {
            "submit_question_contract_outcome": _question_contract_response(
                subject="salespeople, products, and total sales",
                parts=("salesperson name", "product name", "total sales"),
                answer_output_roles=(
                    "ANSWER_VALUE",
                    "ANSWER_VALUE",
                    "MEASURED_VALUE",
                ),
                answer_expression_family="list_rows",
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sale",)),
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": [
                                "answer_1",
                                "answer_2",
                                "answer_3",
                            ],
                            "pattern": "list_rows",
                            "source": {"kind": "read", "read_id": "list_sale_list"},
                            "output_fields": [
                                {"field_id": "staff_name"},
                                {"field_id": "snapshot_merch_name"},
                                {"field_id": "amount"},
                            ],
                        }
                    ],
                }
            },
        }
    )
    catalog = _catalog(
        EndpointRead(
            id="list_sale_list",
            endpoint_name="list_sale_list",
            resource_names=("sale",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                RowPath(
                    id="data_items",
                    path="data.items",
                    parent_path="data",
                    cardinality=RowCardinality.MANY,
                ),
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
                CatalogField(
                    ref="field.sale_item_id",
                    path="data.items.sale_item_id",
                    row_path_id="data_items",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.snapshot_merch_name",
                    path="data.items.snapshot_merch_name",
                    row_path_id="data_items",
                    type="string",
                ),
            ),
            candidate_keys=(
                *_primary_key("sale", "sale_id", "field.sale_id"),
                *_primary_key(
                    "sale_item",
                    "sale_item_id",
                    "field.sale_item_id",
                ),
            ),
        )
    )
    result = run_lookup_question(
        LookupRequest(
            question="List the salespeople, products, and total sales.",
            run_id="run_parent_child_evidence_source",
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_sale_list": {
                        "data": [
                            {
                                "sale_id": "sale-1",
                                "staff_name": "Amina",
                                "amount": "10.00",
                                "items": [
                                    {
                                        "sale_item_id": "item-1",
                                        "snapshot_merch_name": "Lipstick",
                                    }
                                ],
                            }
                        ]
                    }
                },
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == (
        {
            "answer_1": "Amina",
            "answer_2": "Lipstick",
                "answer_3": Decimal("10.00"),
        },
    )
    source_binding_prompt = _source_binding_prompt(planner)
    payload = _planner_prompt_json_section(
        source_binding_prompt,
        label="Candidate evidence sources",
    )
    candidates = [
        candidate
        for context in payload["requested_fact_sources"][0]["source_contexts"]
        for candidate in context["source_options"]
        if candidate.get("read_id") == "list_sale_list"
    ]
    assert len(candidates) == 1
    response_rows = candidates[0]["response_rows"]
    assert response_rows[0]["path"] == "data"
    assert response_rows[1]["path"] == "data.items"
    metric_evidence = {
        item["evidence_id"]
        for support_set in candidates[0]["fulfillment_choices"]
        for slot in support_set["fulfillment_slots"]
        for item in slot.get("metric_measure_evidence") or ()
    }
    assert "source_1.data.amount" in metric_evidence
    available_fields = {
        field["field_id"]
        for row in candidates[0]["response_rows"]
        for field in row.get("fields") or ()
    }
    assert {"staff_name", "amount", "snapshot_merch_name"} <= available_fields


def test_source_binding_exposes_conditional_child_fields_without_query_term_pruning():
    planner = _ToolNamePlannerPort(
        {
            "submit_question_contract_outcome": _question_contract_response(
                subject="products by transaction",
                answer_subject="products",
                parts=("product names", "transaction grouping"),
                answer_expression_family="list_rows",
            ),
            "submit_query_enrichment": _query_enrichment_payload(("transaction",)),
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1", "answer_2"],
                            "pattern": "grouped_rows",
                            "source": {
                                "kind": "read",
                                "read_id": "list_transactions",
                            },
                            "group_fields": [{"field_id": "transaction_id"}],
                            "output_fields": [{"field_id": "merch_label"}],
                        }
                    ],
                }
            },
        }
    )
    catalog = _catalog(
        EndpointRead(
            id="list_transactions",
            endpoint_name="list_transactions",
            resource_names=("transaction",),
            params=(
                CatalogParam(
                    ref="list_transactions.query.include_lines",
                    name="include_lines",
                    source=ParamSource.QUERY,
                    type="boolean",
                    required=False,
                    default=False,
                ),
            ),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                RowPath(
                    id="data_lines",
                    path="data.lines",
                    parent_path="data",
                    cardinality=RowCardinality.MANY,
                ),
            ),
            fields=(
                CatalogField(
                    ref="field.transaction_id",
                    path="data.transaction_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.item_count",
                    path="data.item_count",
                    row_path_id="data",
                    type="integer",
                ),
                CatalogField(
                    ref="field.lines.line_id",
                    path="data.lines.line_id",
                    row_path_id="data_lines",
                    type="uuid",
                    requirements=(
                        FieldRequirement(
                            param_ref="list_transactions.query.include_lines",
                            value=True,
                        ),
                    ),
                ),
                CatalogField(
                    ref="field.lines.merch_label",
                    path="data.lines.merch_label",
                    row_path_id="data_lines",
                    type="string",
                    requirements=(
                        FieldRequirement(
                            param_ref="list_transactions.query.include_lines",
                            value=True,
                        ),
                    ),
                ),
            ),
        )
    )
    result = run_lookup_question(
        LookupRequest(
            question="Which products were in the transactions? Group them by transaction.",
            run_id="run_conditional_child_fields_are_model_selectable",
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_transactions": {
                        "data": [
                            {
                                "transaction_id": "txn-1",
                                "item_count": 1,
                                "lines": [
                                    {
                                        "line_id": "line-1",
                                        "merch_label": "Lipstick",
                                    }
                                ],
                            }
                        ]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert _source_binding_prompt(planner)
    payload = _planner_prompt_json_section(
        _source_binding_prompt(planner),
        label="Candidate evidence sources",
    )
    candidates = [
        candidate
        for context in payload["requested_fact_sources"][0]["source_contexts"]
        for candidate in context["source_options"]
        if candidate.get("read_id") == "list_transactions"
    ]
    assert len(candidates) == 1
    response_rows = candidates[0]["response_rows"]
    assert response_rows[0]["path"] == "data"
    assert response_rows[1]["path"] == "data.lines"


def test_lookup_cutover_catalog_selection_keeps_non_catalog_search_terms_unweighted():
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="rf_active_record_count",
            answer_output_ids=("active_record_count",),
            read_id="activity_read",
            pattern="aggregate_scalar",
            metric={
                "kind": "count_records",
                "count_basis": {
                    "kind": "row_population",
                    "row_path_id": "data",
                    "row_cardinality": "many",
                },
                "label": "active_record_count",
            },
        ),
        question_contract=_question_contract_for(
            "rf_active_record_count",
            description="count active activity records",
            subject_text="activity records",
            binding_target_ids=("active_record_count",),
            answer_output_role="ROW_COUNT",
        ),
        query_enrichment=_query_enrichment_payload(
            ("activity record",),
        ),
    )
    catalog = _catalog(
        EndpointRead(
            id="activity_read",
            endpoint_name="list_activity_records",
            resource_names=("activity record",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.record_id",
                    path="data.record_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.is_active",
                    path="data.is_active",
                    row_path_id="data",
                    type="boolean",
                ),
            ),
            candidate_keys=_primary_key(
                "activity_record", "record_id", "field.record_id"
            ),
        ),
        EndpointRead(
            id="price_list_read",
            endpoint_name="list_price_lists",
            resource_names=("price list",),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.price_list_id",
                    path="data.price_list_id",
                    row_path_id="data",
                    type="uuid",
                ),
                CatalogField(
                    ref="field.is_active",
                    path="data.is_active",
                    row_path_id="data",
                    type="boolean",
                ),
            ),
            candidate_keys=_primary_key(
                "price_list", "price_list_id", "field.price_list_id"
            ),
        ),
    )
    result = run_lookup_question(
        LookupRequest(
            question="How many active activity records are there?",
            run_id="run_non_catalog_search_terms_catalog_selection",
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort(
                {
                    "list_activity_records": {
                        "data": [{"record_id": "activity-1", "is_active": True}]
                    }
                },
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == ({"count": 1},)
    fact_plan_prompt = _source_binding_prompt(planner)
    assert 'read="activity_read"' in fact_plan_prompt
    assert 'read="price_list_read"' not in fact_plan_prompt
