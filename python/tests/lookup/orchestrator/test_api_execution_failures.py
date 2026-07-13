from __future__ import annotations

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_lookup_cutover_fails_closed_when_endpoint_read_returns_non_2xx():
    ports = _ports(
        plan=_metric_answer_plan(),
        catalog=_metric_catalog(),
        responses={},
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=ports.relation_catalog_port,
        data_access_port=_StatusDataAccessPort(
            {
                "metric_read": {
                    "responseStatus": 500,
                    "responseBody": {"error": "database unavailable"},
                    "truncated": False,
                }
            }
        ),
        planner_model_port=ports.planner_model_port,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was the metric total?",
            run_id="run_endpoint_failure",
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "fact_plan_execution_failed"


def test_lookup_cutover_execution_issue_event_carries_failure_context():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="metric_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="metric_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="total_metric",
                    spec=AggregateSpec(
                        input_relation="rows",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.SUM,
                                input_field="metric_total",
                                output_field="total",
                            ),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="total",
                        relation_id="answer_rows",
                        field_id="total",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_metric_catalog(),
        responses={},
        query_enrichment=_query_enrichment_payload(("metric",)),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=ports.relation_catalog_port,
        data_access_port=_StatusDataAccessPort(
            {
                "metric_read": {
                    "responseStatus": 200,
                    "responseBody": {
                        "data": [
                            {
                                "location_id": "location_alpha",
                                "metric_total": "125.00",
                            }
                        ]
                    },
                    "truncated": True,
                }
            }
        ),
        planner_model_port=ports.planner_model_port,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was the total?",
            run_id="run_incomplete_execution_issue",
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "incomplete_evidence"


def test_lookup_cutover_rejects_non_empty_incomplete_final_relation():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="metric_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="metric_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="metric_total"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="metric_total",
                        relation_id="answer_rows",
                        field_id="metric_total",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_metric_catalog(),
        responses={},
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=ports.relation_catalog_port,
        data_access_port=_StatusDataAccessPort(
            {
                "metric_read": {
                    "responseStatus": 200,
                    "responseBody": {
                        "data": [
                            {
                                "location_id": "location_alpha",
                                "metric_total": "125.00",
                            }
                        ]
                    },
                    "truncated": True,
                }
            }
        ),
        planner_model_port=ports.planner_model_port,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was the metric total?",
            run_id="run_incomplete_non_empty",
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "incomplete_evidence"


def test_lookup_cutover_rejects_missing_required_endpoint_param_before_execution():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="metric_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="metric_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="metric_total"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="metric_total",
                        relation_id="answer_rows",
                        field_id="metric_total",
                    ),
                )
            ),
        )
    )
    catalog = _catalog(
        EndpointRead(
            id="metric_read",
            endpoint_name="metric_read",
            resource_names=("metric read",),
            params=(
                CatalogParam(
                    ref="metric_read.query.location_id",
                    name="location_id",
                    source=ParamSource.QUERY,
                    type="uuid",
                    required=True,
                ),
            ),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="field.metric_total",
                    path="data.metric_total",
                    row_path_id="data",
                    type="number",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        )
    )

    result = run_lookup_question(
        LookupRequest(question="What was the metric total?"),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(catalog),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_PlannerPort(plan),
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "plan_validation_failed"


def test_lookup_cutover_rejects_unknown_api_read_before_execution():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="missing_metric_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="location_name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                        RelationField(
                            field_id="metric_total",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(
                            ProjectField(source="location_name", output="location"),
                            ProjectField(source="metric_total"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="location",
                        relation_id="answer_rows",
                        field_id="location",
                    ),
                    RelationResultOutput(
                        id="metric_total",
                        relation_id="answer_rows",
                        field_id="metric_total",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_metric_catalog(),
        responses={
            "metric_read": {
                "data": [{"location_name": "Location Alpha", "metric_total": "125.00"}]
            }
        },
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was the metric total?",
            run_id="run_bad_row_path",
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "provider_runtime_failed"
    assert ports.data_access_port.requests == []


def test_lookup_cutover_stops_before_fact_planning_when_no_source_candidates_exist():
    planner = _RawPlannerPort(
        {
            "outcome": {
                "kind": "fact_plan",
                "answers": [
                    {
                        "requested_fact_id": "rf_answer",
                        "answer_output_ids": ["metric_total"],
                        "pattern": "computed_scalar",
                        "source": {"kind": "values"},
                        "scalar_inputs": [
                            {
                                "input_id": "location_a",
                                "value_id": "missing_location_value",
                            }
                        ],
                        "expression": [{"input_id": "location_a"}],
                        "output": {"scalar_id": "metric_total"},
                    }
                ],
            }
        },
        question_contract=_question_contract_for(
            "rf_answer",
            description="metric total",
            binding_target_ids=("metric_total",),
        ),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                EndpointRead(
                    id="metric_read",
                    endpoint_name="metric_read",
                    resource_names=("metric read",),
                    params=(
                        CatalogParam(
                            ref="metric_read.query.location_id",
                            name="location_id",
                            source=ParamSource.QUERY,
                            type="string",
                        ),
                    ),
                    row_paths=(
                        RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                        RowPath(
                            id="data", path="data", cardinality=RowCardinality.MANY
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.location_name",
                            path="data.location_name",
                            row_path_id="data",
                            type="string",
                        ),
                        CatalogField(
                            ref="field.metric_total",
                            path="data.metric_total",
                            row_path_id="data",
                            type="number",
                        ),
                    ),
                    pagination=PaginationMetadata(
                        mode=PaginationMode.NONE,
                        completeness_policy=CompletenessPolicy.COMPLETE,
                    ),
                )
            )
        ),
        data_access_port=_DataAccessPort(
            {
                "metric_read": {
                    "data": [
                        {"location_name": "Location Beta", "metric_total": "125.00"}
                    ]
                }
            }
        ),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was the metric total for Location Alpha and Location Beta?",
            run_id="run_duplicate_endpoint_param",
        ),
        ports,
    )

    assert result.status == "COMPLETED"
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    assert ports.data_access_port.requests == []


def test_lookup_cutover_fails_when_api_row_path_is_missing_from_response():
    result = run_lookup_question(
        LookupRequest(question="What was the metric total?"),
        _ports(
            plan=_metric_answer_plan(),
            catalog=_metric_catalog(),
            responses={"metric_read": {"detail": "schema changed"}},
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "fact_plan_execution_failed"


def test_lookup_cutover_fails_when_declared_api_field_path_is_missing_from_response():
    result = run_lookup_question(
        LookupRequest(question="What was the metric total?"),
        _ports(
            plan=_metric_answer_plan(),
            catalog=_metric_catalog(),
            responses={"metric_read": {"data": [{"location_name": "Location Alpha"}]}},
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "fact_plan_execution_failed"


def test_lookup_cutover_fails_when_api_row_path_contains_non_object_rows():
    result = run_lookup_question(
        LookupRequest(question="What was the metric total?"),
        _ports(
            plan=_metric_answer_plan(),
            catalog=_metric_catalog(),
            responses={"metric_read": {"data": ["schema changed"]}},
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "fact_plan_execution_failed"


def test_lookup_cutover_fails_when_api_row_path_violates_cardinality():
    result = run_lookup_question(
        LookupRequest(question="What was the metric total?"),
        _ports(
            plan=_metric_answer_plan(),
            catalog=_metric_catalog(),
            responses={"metric_read": {"data": {"metric_total": "125.00"}}},
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "fact_plan_execution_failed"
