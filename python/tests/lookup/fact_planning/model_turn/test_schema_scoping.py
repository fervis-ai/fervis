from ._helpers import *  # noqa: F403


def test_pattern_fact_planning_schema_scopes_answer_body_to_selected_shape():
    request = FactPlanRequest(
        question="How much did Alice make yesterday, and where did Alice work?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="Alice sales amount",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="sales amount",
                        ),
                    ),
                ),
                RequestedFact(
                    id="fact_2",
                    description="Alice work location",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="work location",
                        ),
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
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="get_staff_sales",
                ),
                cardinality="many",
                available_field_ids=("sale_id", "amount", "location_name"),
                available_fields=(
                    SourceField(field_id="sale_id", type="uuid", roles=("identity",)),
                    SourceField(field_id="amount", type="decimal"),
                    SourceField(field_id="location_name", type="string"),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.data.amount",
                        field_id="amount",
                        row_cardinality="many",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="amount fulfills the sales amount.",
                        metric_measure_evidence_ids=("source_1.data.amount",),
                    ),
                ),
            ),
            BoundSource(
                id="sb_2",
                requested_fact_id="fact_2",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="get_staff_sales",
                ),
                cardinality="many",
                available_field_ids=("location_name",),
                available_fields=(
                    SourceField(field_id="location_name", type="string"),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_2.data.location_name",
                        field_id="location_name",
                        row_cardinality="many",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_2",
                        answer_output_id="answer_1",
                        match_basis_explanation="location_name fulfills work location.",
                        value_evidence_ids=("source_2.data.location_name",),
                    ),
                ),
            ),
        ),
    )
    schema = (
        PatternFactPlanTurnPrompt(
            request,
            plan_selection=BoundPlanSelectionSet(
                plan_selections=(
                    BoundSelectedSourceStrategy(
                        requested_fact_id="fact_1",
                        plan_selection_id="fact_1.aggregate_scalar.sb_1",
                        source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                        plan_shape="aggregate_scalar",
                        required_answer_output_ids=("answer_1",),
                        source_members=(
                            _bound_plan_member(request, source_binding_ids=("sb_1",)),
                        ),
                    ),
                    BoundSelectedSourceStrategy(
                        requested_fact_id="fact_2",
                        plan_selection_id="fact_2.list_rows.sb_2",
                        source_strategy_id="source_strategy.fact_2.list_rows.1",
                        plan_shape="list_rows",
                        required_answer_output_ids=("answer_1",),
                        source_members=(
                            _bound_plan_member(request, source_binding_ids=("sb_2",)),
                        ),
                    ),
                )
            ),
        )
        .response_contract()
        .provider_schema
    )

    valid_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "aggregate_scalar",
                    "source_binding_id": "sb_1",
                    "metric": {
                        "selection_basis": "The requested answer is sales amount.",
                        "id": "metric_1",
                        "kind": "aggregate_field",
                        "field_id": "amount",
                    },
                    "function": {
                        "selection_basis": "The requested scalar amount is a total.",
                        "id": "function_sum",
                        "value": "sum",
                    },
                },
                {
                    "requested_fact_id": "fact_2",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "list_rows",
                    "source_binding_id": "sb_2",
                    "output_fields": [{"field_id": "location_name"}],
                },
            ],
        }
    }
    invalid_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "aggregate_scalar",
                    "output_fields": [{"field_id": "amount"}],
                },
                {
                    "requested_fact_id": "fact_2",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "list_rows",
                    "output_fields": [{"field_id": "location_name"}],
                },
            ],
        }
    }

    validate(instance=valid_payload, schema=schema)
    with pytest.raises(ValidationError):
        validate(instance=invalid_payload, schema=schema)


def test_set_difference_schema_respects_selected_member_roles():
    request = FactPlanRequest(
        question="Which products were not sold?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="products not sold",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_products",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ, read_id="products"
                ),
                cardinality="many",
                available_field_ids=("product_id", "name"),
                available_fields=(
                    SourceField(
                        field_id="product_id",
                        type="uuid",
                        roles=("identity", "output"),
                    ),
                    SourceField(field_id="name", type="string"),
                ),
            ),
            BoundSource(
                id="sb_sales",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="sales"),
                cardinality="many",
                available_field_ids=("product_id",),
                available_fields=(
                    SourceField(
                        field_id="product_id",
                        type="uuid",
                        roles=("identity", "output"),
                    ),
                ),
            ),
        ),
    )
    schema = (
        PatternFactPlanTurnPrompt(
            request,
            plan_selection=BoundPlanSelectionSet(
                plan_selections=(
                    BoundSelectedSourceStrategy(
                        requested_fact_id="fact_1",
                        plan_selection_id="fact_1.set_difference.1",
                        source_strategy_id="source_strategy.fact_1.set_difference.1",
                        plan_shape="set_difference",
                        required_answer_output_ids=("answer_1",),
                        source_members=(
                            BoundSourceStrategyMember(
                                source_candidate_id="source_products",
                                role_targets=(
                                    BoundRoleTarget(
                                        requirement_id="candidate_set",
                                        source_candidate_id="source_products",
                                        source_binding_ids=("sb_products",),
                                    ),
                                ),
                            ),
                            BoundSourceStrategyMember(
                                source_candidate_id="source_sales",
                                role_targets=(
                                    BoundRoleTarget(
                                        requirement_id="observed_set",
                                        source_candidate_id="source_sales",
                                        source_binding_ids=("sb_sales",),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )
        .response_contract()
        .provider_schema
    )
    valid_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "set_difference",
                    "candidate": {
                        "source_binding_id": "sb_products",
                        "identity_fields": ["product_id"],
                    },
                    "observed": {
                        "source_binding_id": "sb_sales",
                        "identity_fields": ["product_id"],
                    },
                }
            ],
        }
    }
    swapped_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "set_difference",
                    "candidate": {
                        "source_binding_id": "sb_sales",
                        "identity_fields": ["product_id"],
                    },
                    "observed": {
                        "source_binding_id": "sb_products",
                        "identity_fields": ["product_id"],
                    },
                }
            ],
        }
    }

    validate(instance=valid_payload, schema=schema)
    with pytest.raises(ValidationError):
        validate(instance=swapped_payload, schema=schema)


def test_set_difference_schema_limits_identity_fields_to_declared_identity_roles():
    request = FactPlanRequest(
        question="Which products were not sold?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="products not sold",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_products",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ, read_id="products"
                ),
                cardinality="many",
                available_field_ids=("product_id", "name", "stock_count"),
                available_fields=(
                    SourceField(
                        field_id="product_id",
                        type="uuid",
                        roles=("identity", "output"),
                    ),
                    SourceField(field_id="name", type="string"),
                    SourceField(field_id="stock_count", type="integer"),
                ),
            ),
            BoundSource(
                id="sb_sales",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="sales"),
                cardinality="many",
                available_field_ids=("product_id", "name"),
                available_fields=(
                    SourceField(
                        field_id="product_id",
                        type="uuid",
                        roles=("identity", "output"),
                    ),
                    SourceField(field_id="name", type="string"),
                ),
            ),
        ),
    )
    schema = (
        PatternFactPlanTurnPrompt(
            request,
            plan_selection=BoundPlanSelectionSet(
                plan_selections=(
                    BoundSelectedSourceStrategy(
                        requested_fact_id="fact_1",
                        plan_selection_id="fact_1.set_difference.1",
                        source_strategy_id="source_strategy.fact_1.set_difference.1",
                        plan_shape="set_difference",
                        required_answer_output_ids=("answer_1",),
                        source_members=(
                            BoundSourceStrategyMember(
                                source_candidate_id="source_products",
                                role_targets=(
                                    BoundRoleTarget(
                                        requirement_id="candidate_set",
                                        source_candidate_id="source_products",
                                        source_binding_ids=("sb_products",),
                                    ),
                                ),
                            ),
                            BoundSourceStrategyMember(
                                source_candidate_id="source_sales",
                                role_targets=(
                                    BoundRoleTarget(
                                        requirement_id="observed_set",
                                        source_candidate_id="source_sales",
                                        source_binding_ids=("sb_sales",),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )
        .response_contract()
        .provider_schema
    )
    valid_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "set_difference",
                    "candidate": {
                        "source_binding_id": "sb_products",
                        "identity_fields": ["product_id"],
                    },
                    "observed": {
                        "source_binding_id": "sb_sales",
                        "identity_fields": ["product_id"],
                    },
                }
            ],
        }
    }
    label_identity_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "set_difference",
                    "candidate": {
                        "source_binding_id": "sb_products",
                        "identity_fields": ["name"],
                    },
                    "observed": {
                        "source_binding_id": "sb_sales",
                        "identity_fields": ["name"],
                    },
                }
            ],
        }
    }

    validate(instance=valid_payload, schema=schema)
    with pytest.raises(ValidationError):
        validate(instance=label_identity_payload, schema=schema)


def test_joined_rows_schema_keeps_join_keys_outside_support_fields():
    request = FactPlanRequest(
        question="List products with sales.",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="products and sales",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_products",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ, read_id="products"
                ),
                cardinality="many",
                available_field_ids=("product_id", "name"),
                available_fields=(
                    SourceField(field_id="product_id", type="uuid"),
                    SourceField(field_id="name", type="string"),
                ),
            ),
            BoundSource(
                id="sb_sales",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(kind=SourceKind.API_READ, read_id="sales"),
                cardinality="many",
                available_field_ids=("product_id", "amount"),
                available_fields=(
                    SourceField(field_id="product_id", type="uuid"),
                    SourceField(field_id="amount", type="decimal"),
                ),
            ),
        ),
    )
    schema = (
        PatternFactPlanTurnPrompt(
            request,
            plan_selection=BoundPlanSelectionSet(
                plan_selections=(
                    BoundSelectedSourceStrategy(
                        requested_fact_id="fact_1",
                        plan_selection_id="fact_1.joined_rows.1",
                        source_strategy_id="source_strategy.fact_1.joined_rows.1",
                        plan_shape="joined_rows",
                        required_answer_output_ids=("answer_1",),
                        source_members=(
                            BoundSourceStrategyMember(
                                source_candidate_id="source_products",
                                role_targets=(
                                    BoundRoleTarget(
                                        requirement_id="left",
                                        source_candidate_id="source_products",
                                        source_binding_ids=("sb_products",),
                                    ),
                                ),
                                field_ids=("name",),
                            ),
                            BoundSourceStrategyMember(
                                source_candidate_id="source_sales",
                                role_targets=(
                                    BoundRoleTarget(
                                        requirement_id="right",
                                        source_candidate_id="source_sales",
                                        source_binding_ids=("sb_sales",),
                                    ),
                                ),
                                field_ids=("amount",),
                            ),
                        ),
                    ),
                )
            ),
        )
        .response_contract()
        .provider_schema
    )

    validate(
        instance={
            "outcome": {
                "kind": "fact_plan",
                "answers": [
                    {
                        "requested_fact_id": "fact_1",
                        "answer_output_ids": ["answer_1"],
                        "pattern": "joined_rows",
                        "left": {
                            "source_binding_id": "sb_products",
                            "fields": [{"field_id": "name"}],
                        },
                        "right": {
                            "source_binding_id": "sb_sales",
                            "fields": [{"field_id": "amount"}],
                        },
                        "join_keys": [
                            {
                                "left_field_id": "product_id",
                                "right_field_id": "product_id",
                            }
                        ],
                        "output_fields": [{"side": "left", "field_id": "name"}],
                    }
                ],
            }
        },
        schema=schema,
    )
