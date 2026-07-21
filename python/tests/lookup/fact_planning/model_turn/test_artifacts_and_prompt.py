from ._helpers import *  # noqa: F403


def test_model_turn_prompt_budget_ceiling_is_explicit():
    assert MODEL_TURN_PROMPT_BUDGET_CHARS == 400_000


def test_model_turn_artifact_captures_prompt_schema_and_payload():
    artifact = ModelTurnArtifact(
        system_prompt="system",
        prompt_text="prompt",
        provider_schema={"type": "object"},
        tool_specs=(
            ToolSpec(
                name="submit_pattern_fact_plan",
                description="Submit a fact plan.",
                input_schema={"type": "object"},
            ),
        ),
        submitted_payload={"answer": {}},
        selected_tool_name="submit_pattern_fact_plan",
        verifier_diagnostics=("ok",),
    )

    assert artifact.system_prompt == "system"
    assert artifact.prompt_text == "prompt"
    assert artifact.provider_schema["type"] == "object"
    assert artifact.tool_specs[0].name == "submit_pattern_fact_plan"
    assert artifact.selected_tool_name == "submit_pattern_fact_plan"
    assert artifact.submitted_payload["answer"] == {}
    assert artifact.verifier_diagnostics == ("ok",)


def test_fact_plan_generation_enforces_prompt_budget_before_provider_call():
    class ProviderMustNotBeCalled:
        def generate(self, **kwargs):
            raise AssertionError("provider call must be blocked by prompt budget")

    request = FactPlanRequest(
        question="x" * (MODEL_TURN_PROMPT_BUDGET_CHARS + 1),
        question_contract=_question_contract(),
        relation_catalog=RelationCatalog(),
    )

    with pytest.raises(FactPlanGenerationError, match="prompt budget"):
        generate_pattern_fact_plan(
            request=request,
            plan_selection=_plan_selection_for_request(request),
            model_port=ProviderMustNotBeCalled(),
            provider="anthropic",
            model_key="HAIKU",
            max_thinking_tokens=64,
        )


def test_fact_plan_prompt_renders_endpoint_fields_once():
    prompt = _fact_plan_prompt(
        FactPlanRequest(
            question="What secret can I read?",
            question_contract=_question_contract(),
            relation_catalog=RelationCatalog(
                facts=(
                    CatalogFact(
                        ref="secret.full_value",
                        availability=CatalogFactAvailability.POLICY_BLOCKED,
                        proof_refs=("policy:secret",),
                    ),
                ),
                reads=(
                    EndpointRead(
                        id="read_items",
                        endpoint_name="list_items",
                        fields=(
                            CatalogField(ref="field.item_id", type="string"),
                            CatalogField(ref="field.item_name", type="string"),
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

    read = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "read_items"
    )
    assert "row_source_id" not in read
    assert read["fields"] == [
        {
            "field_id": "field_item_id",
            "roles": ["output", "predicate"],
            "row_cardinality": "one",
            "type": "string",
        },
        {
            "field_id": "field_item_name",
            "roles": ["output", "predicate"],
            "row_cardinality": "one",
            "type": "string",
        },
    ]
    assert "params" not in read


def test_fact_plan_prompt_exposes_bound_params_without_available_param_choices():
    prompt = _fact_plan_prompt(
        FactPlanRequest(
            question="How many in-person records are there?",
            question_contract=_question_contract(),
            relation_catalog=RelationCatalog(
                reads=(
                    EndpointRead(
                        id="records",
                        endpoint_name="list_records",
                        params=(
                            CatalogParam(
                                ref="records.query.channel",
                                name="channel",
                                source=ParamSource.QUERY,
                                type="choice",
                                choices=("STORE", "ONLINE"),
                                choice_labels={
                                    "STORE": "In-person/store checkout records",
                                    "ONLINE": "Ecommerce/online records",
                                },
                            ),
                            CatalogParam(
                                ref="records.query.status",
                                name="status",
                                source=ParamSource.QUERY,
                                type="choice",
                                choices=("OPEN", "COMPLETED"),
                                default="COMPLETED",
                            ),
                        ),
                        fields=(CatalogField(ref="field.record_id", type="uuid"),),
                    ),
                ),
            ),
            bound_sources=(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="fact_1",
                    answer_population=_answer_population(),
                    source=DraftRelationSource(
                        kind=SourceKind.API_READ,
                        read_id="records",
                        param_bindings=(
                            DraftEndpointParamBinding("channel", "STORE"),
                            DraftEndpointParamBinding("status", "COMPLETED"),
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_1",
                            value_evidence_ids=("record_id",),
                            match_basis_explanation=(
                                "answer_value is fulfilled by record_id because "
                                "record_id provides the answer evidence."
                            ),
                        ),
                    ),
                    available_field_ids=("record_id",),
                ),
            ),
        )
    )

    relation_catalog = _json_prompt_section(
        prompt,
        label="Bound sources",
        next_label="Catalog selection",
    )

    read = next(
        item
        for item in _bound_sources(relation_catalog)
        if item.get("read_id") == "records"
    )
    assert "params" not in read
    assert read["bound_params"] == [
        {"param_id": "channel", "value": "STORE"},
        {"param_id": "status", "value": "COMPLETED"},
    ]


def test_fact_plan_prompt_groups_pattern_details_before_copy_rules():
    prompt = _fact_plan_prompt(
        FactPlanRequest(
            question="Which products did Alice sell today? Group them by sale.",
            question_contract=_question_contract(),
            relation_catalog=RelationCatalog(),
        )
    )

    assert "List And Field Patterns" in prompt
    assert "metric.kind" not in prompt
    assert "Metric Patterns" not in prompt
    assert "Grouped Metric Patterns" not in prompt
    assert "Computed Scalar" not in prompt
    assert "Set Difference" not in prompt
    assert "Joined Rows" not in prompt
    assert "grounded_input_ids" not in prompt
    assert "computed_scalar" not in prompt
    assert "rank.limit_value_id" not in prompt


def test_fulfillment_evidence_field_resolution_requires_explicit_mapping_or_exact_field_id():
    assert (
        field_id_for_fulfillment_evidence(
            "source_1.orders.id",
            field_id_by_evidence_id={},
            available_field_ids={"id"},
        )
        == ""
    )
    assert (
        field_id_for_fulfillment_evidence(
            "source_1.orders.id",
            field_id_by_evidence_id={"source_1.orders.id": "id"},
            available_field_ids={"id"},
        )
        == "id"
    )
    assert (
        field_id_for_fulfillment_evidence(
            "id",
            field_id_by_evidence_id={},
            available_field_ids={"id"},
        )
        == "id"
    )
