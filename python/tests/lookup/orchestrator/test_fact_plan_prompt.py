from __future__ import annotations

from tests.lookup.orchestrator._helpers import *  # noqa: F403


def test_lookup_cutover_renders_runtime_values_in_fact_plan_prompt():
    plan = FactPlan(outcome=_plan_clarification("time"))
    planner = _PlannerPort(
        plan,
        question_contract=_question_contract_for(
            "rf_answer",
            description="sales today",
            known_inputs=(_known_time_input("time_today", "today"),),
        ),
        query_enrichment=_query_enrichment_payload(("metric",)),
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
            question="How much sales today?",
            run_id="run_time_anchor",
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-06",
                timezone="Africa/London",
            ),
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_metric_read("metric_read"))),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"
    question_prompt = planner.prompts[0]
    source_binding_prompt = _source_binding_prompt(planner)
    fact_plan_prompt = _fact_plan_prompt(planner)
    assert "Runtime anchors:" not in question_prompt
    assert "Runtime anchors:" not in fact_plan_prompt
    assert "2026-05-06" in source_binding_prompt


def test_lookup_cutover_renders_scalar_memory_values_in_source_binding_prompt():
    artifact = build_fact_artifact(
        artifact_id="run_prior_total",
        outcome=FactOutcome.ANSWERED,
        source_question="What was the prior sales total?",
        source_answer="The prior sales total was 125.00.",
        addresses=(
            FactAddress.value(
                address="value.sales_total",
                value={"type": "decimal", "value": "125.00"},
                derivation={"source": "prior_result"},
            ),
        ),
    )
    plan = FactPlan(outcome=_plan_clarification("follow_up"))
    planner = _PlannerPort(
        plan,
        conversation_resolution=lambda prompt: (
            _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="What percentage increase is there from the prior sales total?",
                actual_text="that",
            )
        ),
        query_enrichment=_query_enrichment_payload(),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What percentage increase is that?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_clarification_read())),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED"
    assert result.error == "planning_failed"
    source_binding_prompt = _source_binding_prompt(planner)
    assert "125" in source_binding_prompt


def test_lookup_cutover_projects_terminal_outcome_memory_into_resolution_prompt():
    artifact = build_fact_artifact(
        artifact_id="run_needs_location",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_location",
                terminal=FactOutcome.NEEDS_CLARIFICATION.value,
                clarification_questions=("location",),
            ),
        ),
        source_question="Which location should I use?",
    )
    planner = _RawPlannerPort(
        _pattern_fact_plan_payload(
            requested_fact_id="fact_1",
            answer_output_ids=("answer_1",),
            read_id="metric_read",
            output_fields=({"field_id": "metric_total"},),
        ),
        question_contract=_question_contract_for(
            "fact_1",
            description="metric total",
            binding_target_ids=("answer_1",),
        ),
        conversation_resolution=lambda prompt: (
            _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="What is the metric total for ABC?",
                actual_text="ABC",
            )
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="ABC",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort(
                {"metric_read": {"data": [{"metric_total": "125.00"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status in {"COMPLETED", "FAILED"}
    assert "Active clarification context:" in planner.prompts[1]
    assert "Which location should I use?" in planner.prompts[1]


def test_lookup_cutover_renders_memory_relation_fields_in_fact_plan_prompt():
    artifact = build_fact_artifact(
        artifact_id="run_prior_items",
        outcome=FactOutcome.ANSWERED,
        source_question="Which prior items were sold?",
        source_answer="SKU-1 had quantity 7.",
        addresses=(
            FactAddress.relation(
                address="relation.items",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                grain_keys=("sku",),
                field_coverage={
                    "sku": "answer_rows.sku",
                    "quantity": "answer_rows.quantity",
                },
                completeness={
                    "status": "complete",
                    "setKind": "observed",
                    "rowCount": 1,
                    "pagination": "not_paginated",
                    "scopeFingerprint": "prior_items",
                },
                row_addresses=("row.items.1",),
            ),
            FactAddress.row(
                address="row.items.1",
                relation="relation.items",
                grain={"sku": "SKU-1"},
                values={
                    "sku": {"type": "string", "value": "SKU-1"},
                    "quantity": {"type": "number", "value": 7},
                },
            ),
        ),
    )
    plan = FactPlan(outcome=_plan_clarification("follow_up"))
    planner = _PlannerPort(
        plan,
        conversation_resolution=lambda prompt: (
            _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="What quantities were in the prior items?",
                actual_text="those",
            )
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What quantities were those?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_clarification_read())),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status in {"NEEDS_CLARIFICATION", "FAILED"}
    fact_plan_prompt = _source_binding_prompt(planner)
    assert "memoryRelations" in fact_plan_prompt
    assert "quantity" in fact_plan_prompt
    assert '"type": "number"' in fact_plan_prompt
