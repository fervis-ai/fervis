from ._helpers import *  # noqa: F403

def test_pattern_fact_plan_prompt_uses_selected_shape_boundary():
    prompt = _pattern_fact_plan_prompt(
        FactPlanRequest(
            question="Which products did Alice sell today?",
            question_contract=_question_contract(),
            relation_catalog=RelationCatalog(),
        )
    )

    assert "Selected plan shapes:" in prompt
    assert "List And Field Patterns" in prompt

def test_pattern_fact_plan_uses_selected_candidate_member_fields():
    request = FactPlanRequest(
        question="Where did Alice work?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="Alice work location",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer_1"),
                        RequestedFactAnswerOutput(id="answer_2"),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=RelationSource(
                    kind=SourceKind.API_READ,
                    read_id="get_staff_sales",
                ),
                cardinality="many",
                available_field_ids=("amount", "location_name"),
                available_fields=(
                    SourceField(field_id="amount", type="decimal"),
                    SourceField(field_id="location_name", type="string"),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.data.amount",
                        field_id="amount",
                        type="decimal",
                        row_cardinality="many",
                    ),
                    SourceEvidenceItem(
                        evidence_id="source_1.data.location_name",
                        field_id="location_name",
                        type="string",
                        row_cardinality="many",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="location_name fulfills work location.",
                        group_key_evidence_ids=("source_1.data.location_name",),
                    ),
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_2",
                        match_basis_explanation="amount fulfills sales amount.",
                        metric_measure_evidence_ids=("source_1.data.amount",),
                    ),
                ),
            ),
        ),
    )
    plan_selection = BoundPlanSelectionSet(
        plan_selections=(
            BoundSelectedSourceStrategy(
                requested_fact_id="fact_1",
                plan_selection_id="fact_1.list_rows.sb_1",
                source_strategy_id="source_strategy.fact_1.list_rows.1",
                plan_shape="list_rows",
                required_answer_output_ids=("answer_1", "answer_2"),
                source_members=(
                    BoundSourceStrategyMember(
                        source_candidate_id="source_1",
                        source_binding_ids=("sb_1",),
                        field_ids=("location_name", "amount"),
                    ),
                ),
            ),
        )
    )
    prompt = _pattern_fact_plan_prompt(request, plan_selection=plan_selection)
    bound_sources_payload = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Required fulfillment evidence",
    )
    field_ids = {
        field["field_id"]
        for source in bound_sources_payload["bound_sources"]
        for field in source["fields"]
    }

    assert field_ids == {"location_name", "amount"}

    schema = (
        PatternFactPlanTurnPrompt(request, plan_selection=plan_selection)
        .response_contract()
        .provider_schema
    )
    valid_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                    {
                        "requested_fact_id": "fact_1",
                        "source_binding_id": "sb_1",
                        "answer_output_ids": ["answer_1", "answer_2"],
                        "output_fields": [
                        {"field_id": "location_name"},
                        {"field_id": "amount"},
                    ],
                }
            ],
        }
    }

    validate(instance=valid_payload, schema=schema)

    class ProviderReturnsFixedListRowsPlan:
        def generate(self, **kwargs):
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_pattern_fact_plan",
                        "arguments": valid_payload,
                    }
                ),
                "usage": {},
            }

    result = generate_pattern_fact_plan(
        request=request,
        plan_selection=plan_selection,
        model_port=ProviderReturnsFixedListRowsPlan(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert tuple(
        field.field_id for field in result.plan.outcome.relations[0].fields
    ) == ("location_name", "amount")

def test_pattern_fact_prompt_excludes_raw_memory_inputs():
    request = FactPlanRequest(
        question="Which store had the highest sales today?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(),
        bound_sources=(_api_bound_source_for_memory_boundary_test(),),
        memory_inputs={
            "memoryRelations": [{"id": "prior.rows", "fields": [{"id": "secret"}]}],
            "memoryValues": [{"id": "prior.value", "value": "secret"}],
            "memoryOutcomes": [{"id": "prior.outcome"}],
        },
    )
    prompt = _pattern_fact_plan_prompt(request)

    assert "Memory inputs:" not in prompt
    assert "prior.value" not in prompt
    assert "prior.outcome" not in prompt

def test_fact_plan_prompt_uses_bound_source_lineage_not_rehydrated_candidate_fields():
    request = FactPlanRequest(
        question="What was the sales amount?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sales",
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.amount",
                            path="data.amount",
                            row_path_id="data",
                            type="decimal",
                        ),
                        CatalogField(
                            ref="field.internal_notes",
                            path="data.internal_notes",
                            row_path_id="data",
                            type="string",
                        ),
                    ),
                ),
            )
        ),
        bound_sources=(
            _bound_source_fixture(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="rf_answer",
                    answer_population=_answer_population(),
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="sales",
                        row_source_id=api_row_source_id("sales", "data"),
                    ),
                    source_candidate_id="source_1",
                    cardinality="many",
                    available_field_ids=("amount",),
                    available_fields=(SourceField(field_id="amount", type="decimal"),),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="bound.amount",
                            field_id="amount",
                            type="decimal",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="rf_answer",
                            answer_output_id="answer",
                            match_basis_explanation="amount answers the request",
                            metric_measure_evidence_ids=("bound.amount",),
                        ),
                    ),
                )
            ),
        ),
    )

    payload = _json_prompt_section(
        _fact_plan_prompt(request),
        label="Bound sources",
        next_label="Catalog selection",
    )

    source = _bound_sources(payload)[0]
    assert [field["field_id"] for field in source["fields"]] == ["amount"]
    assert "internal_notes" not in json.dumps(source)

def test_fact_plan_prompt_projects_read_scoped_fields_as_read_handles():
    prompt = _fact_plan_prompt(
        FactPlanRequest(
            question="Which profile field is available?",
            question_contract=_question_contract(),
            relation_catalog=RelationCatalog(
                reads=(
                    EndpointRead(
                        id="customer_read",
                        endpoint_name="get_customer",
                        fields=(
                            CatalogField(ref="field.customer.name", type="string"),
                        ),
                        facts=(
                            CatalogFact(
                                ref="profile.name",
                                field_ref="field.customer.name",
                            ),
                        ),
                    ),
                    EndpointRead(
                        id="staff_read",
                        endpoint_name="get_staff",
                        fields=(CatalogField(ref="field.staff.name", type="string"),),
                        facts=(
                            CatalogFact(
                                ref="profile.name",
                                field_ref="field.staff.name",
                            ),
                        ),
                    ),
                ),
            ),
        )
    )

    relation_catalog = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Catalog selection",
    )

    customer_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "customer_read"
    )
    staff_source = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "staff_read"
    )

    assert customer_source["fields"] == [
        {
            "field_id": "field_customer_name",
            "label": "field customer name",
            "roles": ["output", "predicate"],
            "row_cardinality": "one",
            "type": "string",
        }
    ]
    assert staff_source["fields"] == [
        {
            "field_id": "field_staff_name",
            "label": "field staff name",
            "roles": ["output", "predicate"],
            "row_cardinality": "one",
            "type": "string",
        }
    ]

def test_fact_plan_prompt_does_not_expose_raw_unselected_read_ids():
    request = FactPlanRequest(
        question="What was the answer?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="selected_read",
                    endpoint_name="get_selected",
                    fields=(CatalogField(ref="field.answer", type="string"),),
                ),
                EndpointRead(
                    id="sensitive_unselected_read",
                    endpoint_name="get_internal",
                    fields=(CatalogField(ref="field.answer", type="string"),),
                ),
            )
        ),
        catalog_selection=CatalogSelectionResult(
            relation_catalog=RelationCatalog(
                reads=(
                    EndpointRead(
                        id="selected_read",
                        endpoint_name="get_selected",
                        fields=(CatalogField(ref="field.answer", type="string"),),
                    ),
                )
            ),
            selected_read_ids=("selected_read",),
            requested_fact_selections=(
                RequestedFactCatalogSelection(
                    requested_fact_id="rf_answer",
                    query_terms=("answer",),
                    rankings=(
                        CatalogSelectionRanking(
                            read_id="selected_read",
                            score=1,
                            matched_terms=("answer",),
                        ),
                    ),
                    selected_read_ids=("selected_read",),
                    unselected_positive_read_ids=("sensitive_unselected_read",),
                ),
            ),
        ),
    )

    prompt = _fact_plan_prompt(request)

    assert "selected_read" in prompt
    assert api_row_source_id("selected_read", "root") not in prompt
    assert "sensitive_unselected_read" not in prompt

def test_shape_compatible_payload_keeps_fields_that_source_binding_proved():
    source = {
        "fields": [
            {
                "evidence_id": "source_1_evidence_1",
                "field_id": "price_list_id",
                "row_cardinality": "one",
                "type": "uuid",
            },
            {
                "evidence_id": "source_1_evidence_2",
                "field_id": "price_list_item_id",
                "row_cardinality": "many",
                "type": "uuid",
            },
        ],
        "fulfills": [
            {
                "answer_output_id": "answer_1",
                "group_key_evidence_ids": ["source_1_evidence_1"],
                "scope_evidence_ids": [],
            }
        ],
    }

    result = _shape_compatible_bound_source(source, plan_shape="aggregate_scalar")

    assert [field["field_id"] for field in result["fields"]] == [
        "price_list_id",
        "price_list_item_id",
    ]

def test_shape_compatible_payload_keeps_group_key_fields():
    source = {
        "fields": [
            {
                "evidence_id": "source_1.location_name",
                "field_id": "location_name",
                "row_cardinality": "one",
                "type": "string",
            },
            {
                "evidence_id": "source_1.calculated_pay",
                "field_id": "calculated_pay",
                "row_cardinality": "many",
                "type": "number",
            },
        ],
        "fulfills": [
            {
                "answer_output_id": "answer_1",
                "group_key_evidence_ids": ["source_1.calculated_pay"],
                "metric_measure_evidence_ids": ["source_1.calculated_pay"],
                "scope_evidence_ids": [],
                "group_key_evidence_ids": ["source_1.location_name"],
            }
        ],
    }

    result = _shape_compatible_bound_source(source, plan_shape="ranked_aggregate")

    assert [field["field_id"] for field in result["fields"]] == [
        "location_name",
        "calculated_pay",
    ]

def test_pattern_schema_uses_shape_compatible_bound_source_fields():
    request = FactPlanRequest(
        question="Which location has the highest total amount?",
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="nested_amounts",
                    endpoint_name="list_nested_amounts",
                    resource_names=("amount",),
                    row_paths=(
                        RowPath(
                            id="root",
                            path="root",
                            cardinality=RowCardinality.ONE,
                        ),
                        RowPath(
                            id="items",
                            path="data.items",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.root_label",
                            path="root.label",
                            row_path_id="root",
                            type="string",
                        ),
                        CatalogField(
                            ref="field.child_location_id",
                            path="data.items.location_id",
                            row_path_id="items",
                            type="uuid",
                        ),
                        CatalogField(
                            ref="field.child_amount",
                            path="data.items.amount",
                            row_path_id="items",
                            type="number",
                        ),
                    ),
                ),
            )
        ),
        bound_sources=(
            _bound_source_fixture(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="rf_answer",
                    answer_population=_answer_population(),
                    source_candidate_id="source_1",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="nested_amounts",
                    ),
                    cardinality="many",
                    available_field_ids=("label", "location_id", "amount"),
                    available_fields=(
                        SourceField(field_id="label", type="string"),
                        SourceField(field_id="location_id", type="uuid"),
                        SourceField(field_id="amount", type="number"),
                    ),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="source_1.root.label",
                            field_id="label",
                            row_cardinality="one",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.items.location_id",
                            field_id="location_id",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.items.amount",
                            field_id="amount",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="rf_answer",
                            answer_output_id="answer",
                            match_basis_explanation="child rows provide the ranked amount.",
                            group_key_evidence_ids=("source_1.items.location_id",),
                            metric_measure_evidence_ids=("source_1.items.amount",),
                        ),
                    ),
                )
            ),
        ),
    )
    plan_selection = _plan_selection_for_request(request, plan_shape="ranked_aggregate")
    prompt = PatternFactPlanTurnPrompt(request, plan_selection=plan_selection)
    prompt_text = prompt.to_model_payload(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
            memory_payload={},
        )
    ).prompt_text
    schema_text = json.dumps(prompt.response_contract().provider_schema)

    assert '"field_id": "label"' not in prompt_text
    assert '"label"' not in schema_text
    assert '"location_id"' in schema_text
    assert '"amount"' in schema_text
    assert '<group id="group_1" field="location_id"' in prompt_text
    assert '<metric id="metric_1" kind="aggregate_field" field="amount"' in prompt_text
