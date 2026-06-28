from __future__ import annotations

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_lookup_cutover_fails_closed_on_invalid_source_binding_output():
    valid_fact_plan = _pattern_fact_plan_payload(
        requested_fact_id="fact_1",
        answer_output_ids=("answer_1",),
        read_id="metric_read",
        output_fields=({"field_id": "metric_total", "label": "Metric total"},),
    )
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(
                _question_contract_for(
                    "fact_1",
                    description="metric total",
                    binding_target_ids=("answer_1",),
                )
            ),
            "submit_query_enrichment": _query_enrichment_payload(("metric",)),
            "submit_pattern_fact_plan": valid_fact_plan,
            "submit_source_binding": {"outcome": {"kind": "not_valid"}},
        },
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="metric_read",
                measured_value_fields=("metric_total",),
            ),
        ),
    )
    result = run_lookup_question(
        LookupRequest(
            question="What is the answer?",
            run_id="run_bad_plan",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort(
                {"metric_read": {"data": [{"metric_total": "125.00"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"


def test_lookup_cutover_fails_closed_on_invalid_verified_plan():
    valid_source_binding_plan = _pattern_fact_plan_payload(
        requested_fact_id="rf_answer",
        answer_output_ids=("answer",),
        read_id="metric_read",
        output_fields=({"field_id": "metric_total"},),
    )
    invalid_fact_plan = _pattern_fact_plan_payload(
        requested_fact_id="rf_answer",
        answer_output_ids=("answer",),
        read_id="metric_read",
        output_fields=({"field_id": "missing_field"},),
    )
    result = run_lookup_question(
        LookupRequest(
            question="What is the answer?",
            run_id="run_invalid_verified_plan",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort(
                {"metric_read": {"data": [{"metric_total": "125.00"}]}}
            ),
            planner_model_port=_RawPlannerPort(
                invalid_fact_plan,
                source_binding_arguments=valid_source_binding_plan,
                query_enrichment=_query_enrichment_payload(("metric",)),
                read_eligibility_retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="metric_read",
                        measured_value_fields=("metric_total",),
                    ),
                ),
            ),
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"


def test_lookup_cutover_records_provider_failure():
    result = run_lookup_question(
        LookupRequest(
            question="What is the answer?",
            run_id="run_provider_failure",
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-06",
                timezone="Africa/London",
            ),
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_FailingPlannerPort(),
        ),
    )

    assert result.status == "FAILED"
    assert result.error == ErrorCode.PROVIDER_RUNTIME_FAILED


def test_lookup_cutover_preserves_provider_timeout_error_code():
    result = run_lookup_question(
        LookupRequest(
            question="What is the answer?",
            run_id="run_provider_timeout",
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-06",
                timezone="Africa/London",
            ),
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_TimeoutPlannerPort(),
        ),
    )

    assert result.status == "FAILED"
    assert result.error == ErrorCode.PROVIDER_TIMEOUT


def test_lookup_cutover_returns_validation_failure_for_invalid_fact_plan():
    valid_source_binding_plan = _pattern_fact_plan_payload(
        requested_fact_id="rf_answer",
        answer_output_ids=("answer",),
        read_id="metric_read",
        output_fields=({"field_id": "metric_total"},),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_metric_catalog()),
        data_access_port=_DataAccessPort(
            {"metric_read": {"data": [{"metric_total": "125.00"}]}}
        ),
        planner_model_port=_RawPlannerPort(
            _pattern_fact_plan_payload(
                requested_fact_id="rf_answer",
                answer_output_ids=("other_answer",),
                read_id="metric_read",
                output_fields=({"field_id": "metric_total", "label": "answer"},),
            ),
            source_binding_arguments=valid_source_binding_plan,
            question_contract=_question_contract_for("rf_answer"),
            query_enrichment=_query_enrichment_payload(("metric",)),
            read_eligibility_retention_specs=(
                ReadEligibilityRetentionSpec(
                    requested_fact_id="fact_1",
                    read_id="metric_read",
                    measured_value_fields=("metric_total",),
                ),
            ),
        ),
    )

    result = run_lookup_question(
        LookupRequest(question="What is the answer?", run_id="run_invalid_plan"),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"


def test_lookup_cutover_blocks_invalid_plan_clarification_before_synthesis():
    valid_source_binding_plan = _pattern_fact_plan_payload(
        requested_fact_id="rf_answer",
        answer_output_ids=("answer",),
        read_id="metric_read",
        output_fields=({"field_id": "metric_total"},),
    )
    invalid_clarification_plan = {
        "outcome": {
            "kind": "needs_clarification",
            "missing_catalog_inputs": [
                {
                    "kind": "required_input",
                    "id": "ask_person_again",
                    "requested_fact_id": "rf_answer",
                    "required_catalog_input_id": "person_name",
                }
            ],
        }
    }
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_metric_catalog()),
        data_access_port=_DataAccessPort(
            {"metric_read": {"data": [{"metric_total": "125.00"}]}}
        ),
        planner_model_port=_RawPlannerPort(
            invalid_clarification_plan,
            source_binding_arguments=valid_source_binding_plan,
            question_contract=_question_contract_for("rf_answer"),
            query_enrichment=_query_enrichment_payload(("metric",)),
            read_eligibility_retention_specs=(
                ReadEligibilityRetentionSpec(
                    requested_fact_id="fact_1",
                    read_id="metric_read",
                    measured_value_fields=("metric_total",),
                ),
            ),
        ),
    )

    result = run_lookup_question(
        LookupRequest(question="How much sales did Alice make?"),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"
