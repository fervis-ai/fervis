from ._helpers import *  # noqa: F403


def test_fact_plan_prompt_marks_grounded_required_params_as_satisfied():
    row_source_id = api_row_source_id("sales", "root")
    request = FactPlanRequest(
        question="How much revenue on April 8?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    params=(
                        CatalogParam(
                            ref="sales.query.start_date",
                            name="start_date",
                            source=ParamSource.QUERY,
                            type="date",
                            required=True,
                        ),
                        CatalogParam(
                            ref="sales.query.end_date",
                            name="end_date",
                            source=ParamSource.QUERY,
                            type="date",
                            required=True,
                        ),
                    ),
                    fields=(CatalogField(ref="field.total", type="decimal"),),
                ),
            )
        ),
        available_values=(
            FactValue.time(
                id="april_8",
                expression="April 8",
                resolved_start="2026-04-08",
                resolved_end="2026-04-08",
                granularity="day",
            ),
        ),
        available_value_uses=(
            GroundedInputUse(
                id="grounded_start",
                value_id="april_8",
                row_source_id=row_source_id,
                param_id="start_date",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_end",
                value_id="april_8",
                row_source_id=row_source_id,
                param_id="end_date",
                value_component=TimeComponent.END,
            ),
        ),
    )

    prompt = _fact_plan_prompt(request)
    assert prompt.index("Operation input values:") < prompt.index("Bound sources:")
    relation_catalog = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Catalog selection",
    )

    sales_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "sales"
    )
    assert "params" not in sales_source
    assert sales_source["applied_filters"] == [
        {
            "display_value": "April 8",
            "kind": "time",
            "resolved_end": "2026-04-08",
            "resolved_start": "2026-04-08",
            "value_id": "april_8",
        }
    ]


def test_fact_plan_prompt_marks_grounded_optional_params_as_satisfied():
    row_source_id = api_row_source_id("sales", "root")
    request = FactPlanRequest(
        question="How much did Azraah make in sales?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    params=(
                        CatalogParam(
                            ref="sales.query.staff_id",
                            name="staff_id",
                            source=ParamSource.QUERY,
                            type="uuid",
                        ),
                    ),
                    fields=(CatalogField(ref="field.amount", type="decimal"),),
                ),
            )
        ),
        available_values=(
            FactValue.identity(
                id="azraah",
                entity_kind="staff",
                key_id="primary_key",
                key_component_id="staff_id",
                value="staff_1",
                display_value="Azraah Fatuma",
            ),
        ),
        available_value_uses=(
            GroundedInputUse(
                id="grounded_staff",
                value_id="azraah",
                row_source_id=row_source_id,
                param_id="staff_id",
            ),
        ),
    )

    prompt = _fact_plan_prompt(request)
    assert prompt.index("Operation input values:") < prompt.index("Bound sources:")
    relation_catalog = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Catalog selection",
    )

    sales_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "sales"
    )
    assert "params" not in sales_source
    assert sales_source["applied_filters"] == [
        {
            "display_value": "Azraah Fatuma",
            "kind": "identity",
            "value_id": "azraah",
        }
    ]

    operation_values = _json_prompt_section(
        prompt,
        label="Operation input values",
        next_label="Bound sources",
    )
    assert operation_values == {"values": []}


def test_fact_plan_prompt_projects_grounded_inputs_as_scoped_row_set():
    row_source_id = api_row_source_id("sales", "root")
    request = FactPlanRequest(
        question="How much did Azraah make in sales on February 14, 2026?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    params=(
                        CatalogParam(
                            ref="sales.query.staff_id",
                            name="staff_id",
                            source=ParamSource.QUERY,
                            type="uuid",
                        ),
                        CatalogParam(
                            ref="sales.query.start_date",
                            name="start_date",
                            source=ParamSource.QUERY,
                            type="date",
                        ),
                        CatalogParam(
                            ref="sales.query.end_date",
                            name="end_date",
                            source=ParamSource.QUERY,
                            type="date",
                        ),
                    ),
                    fields=(CatalogField(ref="field.amount", type="decimal"),),
                ),
            )
        ),
        available_values=(
            FactValue.identity(
                id="azraah",
                known_input_id="fact_1_input_1",
                entity_kind="staff",
                key_id="primary_key",
                key_component_id="staff_id",
                value="staff_1",
                display_value="Azraah Fatuma",
                proof_refs=("known_input:fact_1_input_1",),
            ),
            FactValue.time(
                id="feb_14",
                expression="February 14, 2026",
                resolved_start="2026-02-14",
                resolved_end="2026-02-14",
                granularity="day",
            ),
        ),
        available_value_uses=(
            GroundedInputUse(
                id="grounded_staff",
                value_id="azraah",
                row_source_id=row_source_id,
                param_id="staff_id",
            ),
            GroundedInputUse(
                id="grounded_start",
                value_id="feb_14",
                row_source_id=row_source_id,
                param_id="start_date",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_end",
                value_id="feb_14",
                row_source_id=row_source_id,
                param_id="end_date",
                value_component=TimeComponent.END,
            ),
        ),
    )

    prompt = _fact_plan_prompt(request)
    relation_catalog = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Catalog selection",
    )

    sales_relation = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "sales"
    )
    assert "params" not in sales_relation
    assert sales_relation["applied_filters"] == [
        {
            "display_value": "Azraah Fatuma",
            "kind": "identity",
            "known_input_id": "fact_1_input_1",
            "value_id": "azraah",
        },
        {
            "display_value": "February 14, 2026",
            "kind": "time",
            "resolved_end": "2026-02-14",
            "resolved_start": "2026-02-14",
            "value_id": "feb_14",
        },
    ]
    assert "staff_1" not in json.dumps(relation_catalog)

    operation_values = _json_prompt_section(
        prompt,
        label="Operation input values",
        next_label="Bound sources",
    )
    assert operation_values == {"values": []}


def test_fact_plan_prompt_projects_identity_value_to_matching_source_field_filter():
    request = FactPlanRequest(
        question="How many stores are in London?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="locations",
                    endpoint_name="list_location_list",
                    resource_names=("location",),
                    params=(
                        CatalogParam(
                            ref="locations.query.type",
                            name="type",
                            source=ParamSource.QUERY,
                            type="choice",
                            choices=("STORE", "WAREHOUSE"),
                        ),
                    ),
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.location_id",
                            path="data.location_id",
                            row_path_id="data",
                            type="uuid",
                        ),
                        CatalogField(
                            ref="field.type",
                            path="data.type",
                            row_path_id="data",
                            type="choice",
                            choices=("STORE", "WAREHOUSE"),
                        ),
                        CatalogField(
                            ref="field.area_id",
                            path="data.area.area_id",
                            row_path_id="data",
                            type="uuid",
                        ),
                    ),
                    candidate_keys=(
                        CandidateKey(
                            id="primary_key",
                            entity_kind="location",
                            components=(
                                CandidateKeyComponent(
                                    id="location_id",
                                    field_ref="field.location_id",
                                ),
                            ),
                            primary=True,
                        ),
                    ),
                    entity_references=(
                        EntityReference(
                            id="area_reference",
                            target_entity_kind="area",
                            target_key_id="primary_key",
                            components=(
                                EntityReferenceComponent(
                                    target_component_id="area_id",
                                    local_field_ref="field.area_id",
                                ),
                            ),
                        ),
                    ),
                ),
                EndpointRead(
                    id="areas",
                    endpoint_name="list_area_list",
                    resource_names=("area",),
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.area_authority_id",
                            path="data.area_id",
                            row_path_id="data",
                            type="uuid",
                        ),
                    ),
                    candidate_keys=(
                        CandidateKey(
                            id="primary_key",
                            entity_kind="area",
                            components=(
                                CandidateKeyComponent(
                                    id="area_id",
                                    field_ref="field.area_authority_id",
                                ),
                            ),
                            primary=True,
                        ),
                    ),
                ),
            )
        ),
        available_values=(
            FactValue.identity(
                id="nairobi_area",
                known_input_id="input_1",
                entity_kind="area",
                key_id="primary_key",
                key_component_id="area_id",
                value="area_nairobi",
                display_value="London",
                proof_refs=("known_input:input_1",),
            ),
        ),
        available_value_uses=(),
    )

    prompt = _fact_plan_prompt(request)
    relation_catalog = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Catalog selection",
    )

    locations = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "locations"
    )
    assert locations["applied_filters"] == [
        {
            "display_value": "London",
            "field_ids": ["area_area_id"],
            "kind": "identity",
            "known_input_id": "input_1",
            "value_id": "nairobi_area",
        }
    ]


def test_fact_plan_prompt_treats_duplicate_grounded_dates_as_satisfied_once():
    row_source_id = api_row_source_id("sales", "root")
    request = FactPlanRequest(
        question="How much revenue and average ticket on January 1?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    params=(
                        CatalogParam(
                            ref="sales.query.start_date",
                            name="start_date",
                            source=ParamSource.QUERY,
                            type="date",
                            required=True,
                        ),
                        CatalogParam(
                            ref="sales.query.end_date",
                            name="end_date",
                            source=ParamSource.QUERY,
                            type="date",
                            required=True,
                        ),
                    ),
                    fields=(CatalogField(ref="field.total", type="decimal"),),
                ),
            )
        ),
        available_values=(
            FactValue.time(
                id="jan_1_revenue",
                expression="January 1",
                resolved_start="2030-01-01",
                resolved_end="2030-01-01",
                granularity="day",
            ),
            FactValue.time(
                id="jan_1_average_ticket",
                expression="January 1",
                resolved_start="2030-01-01",
                resolved_end="2030-01-01",
                granularity="day",
            ),
        ),
        available_value_uses=(
            GroundedInputUse(
                id="grounded_start_revenue",
                value_id="jan_1_revenue",
                row_source_id=row_source_id,
                param_id="start_date",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_start_average_ticket",
                value_id="jan_1_average_ticket",
                row_source_id=row_source_id,
                param_id="start_date",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_end_revenue",
                value_id="jan_1_revenue",
                row_source_id=row_source_id,
                param_id="end_date",
                value_component=TimeComponent.END,
            ),
            GroundedInputUse(
                id="grounded_end_average_ticket",
                value_id="jan_1_average_ticket",
                row_source_id=row_source_id,
                param_id="end_date",
                value_component=TimeComponent.END,
            ),
        ),
    )

    relation_catalog = _json_prompt_section(
        _fact_plan_prompt(request),
        label="Bound sources",
        next_label="Catalog selection",
    )

    sales_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "sales"
    )
    assert "params" not in sales_source
    assert sales_source["applied_filters"] == [
        {
            "display_value": "January 1",
            "kind": "time",
            "resolved_end": "2030-01-01",
            "resolved_start": "2030-01-01",
            "value_id": "jan_1_revenue",
        }
    ]


def test_fact_plan_prompt_uses_explicit_source_binding_for_required_dates():
    request = FactPlanRequest(
        question="How much revenue on January 1 and January 2?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    params=(
                        CatalogParam(
                            ref="sales.query.start_date",
                            name="start_date",
                            source=ParamSource.QUERY,
                            type="date",
                            required=True,
                        ),
                    ),
                    fields=(CatalogField(ref="field.total", type="decimal"),),
                ),
            )
        ),
        available_values=(
            FactValue.time(
                id="jan_1",
                expression="January 1",
                resolved_start="2030-01-01",
                resolved_end="2030-01-01",
                granularity="day",
            ),
            FactValue.time(
                id="jan_2",
                expression="January 2",
                resolved_start="2030-01-02",
                resolved_end="2030-01-02",
                granularity="day",
            ),
        ),
        bound_sources=(
            BoundSource(
                id="sb_sales_jan_1",
                requested_fact_id="rf_answer",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="sales",
                    param_bindings=(
                        DraftEndpointParamBinding(
                            param_id="start_date",
                            value="2030-01-01",
                        ),
                    ),
                ),
                available_field_ids=("total",),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="rf_answer",
                        answer_output_id="answer",
                        value_evidence_ids=("total",),
                        match_basis_explanation=(
                            "answer is fulfilled by total because total provides "
                            "the requested revenue value."
                        ),
                    ),
                ),
            ),
        ),
    )

    relation_catalog = _json_prompt_section(
        _fact_plan_prompt(request),
        label="Bound sources",
        next_label="Catalog selection",
    )

    sales_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "sales"
    )
    assert sales_source["bound_params"] == [
        {
            "param_id": "start_date",
            "value": "2030-01-01",
        }
    ]
    assert "missing_required_inputs" not in relation_catalog


def test_fact_plan_prompt_hides_missing_inputs_when_fact_has_executable_relation():
    request = _request_with_executable_relation_and_required_detail()

    relation_catalog = _json_prompt_section(
        _fact_plan_prompt(request),
        label="Bound sources",
        next_label="Catalog selection",
    )

    available_ids = {item.get("read_id") for item in _bound_sources(relation_catalog)}
    assert "sales" in available_ids
    assert "sale_detail" not in available_ids
    assert "missing_required_inputs" not in relation_catalog


def test_fact_plan_schema_hides_clarification_when_fact_has_executable_relation():
    class ProviderAssertsNoClarificationBranch:
        def generate(self, **kwargs):
            schema_text = json.dumps(kwargs["tool_specs"][0].input_schema)
            assert "needs_clarification" not in schema_text
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_pattern_fact_plan",
                        "arguments": {
                            "outcome": {
                                "kind": "impossible",
                                "blocked_facts": [
                                    {
                                        "requested_fact_id": "rf_answer",
                                        "basis": "catalog_access",
                                        "evidence_refs": [
                                            f"row_source:{api_row_source_id('sales', 'root')}"
                                        ],
                                    }
                                ],
                            }
                        },
                    }
                ),
                "usage": {},
            }

    request = _request_with_executable_relation_and_required_detail()
    generate_pattern_fact_plan(
        request=request,
        plan_selection=_plan_selection_for_request(request),
        model_port=ProviderAssertsNoClarificationBranch(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )
