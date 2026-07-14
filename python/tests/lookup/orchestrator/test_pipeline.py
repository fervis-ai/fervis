from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from uuid import UUID

from fervis import error_codes as common_error_codes

from fervis.lineage.enums import RunStepKey, SourceReadStatus
from fervis.lineage.recorder import (
    AnsweredRunResultWrite,
    CatalogEndpointWrite,
    FactualTerminalRunResultWrite,
    RuntimeErrorResultWrite,
    RunStepWrite,
    SourceReadWrite,
)
from fervis.lineage.step_summary import step_semantic_items_from_json
from fervis.lookup.lineage.steps import LineageRuntimeStepSink
from fervis.lookup.lineage.errors import LineagePersistenceUnavailable
from fervis.lookup.relation_catalog import CatalogEndpointMetadata
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole
from fervis.lookup.turn_prompts import HostPromptContext
from fervis.run_work.events import CollectingQuestionRunEventSink

from tests.lookup.orchestrator._helpers import *  # noqa: F403
from tests.lookup.source_binding_helpers import (
    _param_decisions_for_prompt,
    _source_candidate_fulfillment_support_sets,
    _source_candidate_param_decision_options,
    plan_selection_payload_from_fact_plan,
    source_binding_payload_from_fact_plan_with_invocation_overrides,
    source_binding_target_id_for_candidate,
)


def test_lookup_uses_semantic_read_eligibility_after_recall():
    target_read_id = "z_sales_summary"
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id=target_read_id,
                    ),
                    fields=(
                        RelationField(
                            field_id="total_revenue",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="sales_summary",
                        fields=(
                            ProjectField(source="total_revenue", output="answer_1"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="total_revenue",
                    ),
                )
            ),
        )
    )
    decoys = tuple(_sales_revenue_read(f"a_decoy_{index}") for index in range(6))
    catalog = _catalog(*decoys, _sales_revenue_read(target_read_id))
    planner = _ReadEligibilityPlannerPort(
        plan=plan,
        eligible_read_id=target_read_id,
    )
    data_access = _DataAccessPort(
        {target_read_id: {"data": [{"total_revenue": "14.00"}]}}
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog),
        data_access_port=data_access,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue?",
            run_id="run_read_eligibility",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", (
        result.error,
        result.answer,
    )
    assert data_access.requests == [{"endpointName": target_read_id, "args": {}}]
    assert target_read_id in planner.read_eligibility_prompt
    assert "a_decoy_5" in planner.read_eligibility_prompt
    assert target_read_id in planner.source_binding_selection_prompt
    assert "a_decoy_0" not in planner.source_binding_selection_prompt


def test_lookup_pipeline_emits_runtime_progress_for_major_phases() -> None:
    read_id = "sales_summary"
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id=read_id,
                    ),
                    fields=(
                        RelationField(
                            field_id="total_revenue",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="sales_summary",
                        fields=(
                            ProjectField(source="total_revenue", output="answer_1"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="total_revenue",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=_catalog(_sales_revenue_read(read_id)),
        responses={read_id: {"data": [{"total_revenue": "14.00"}]}},
    )
    events = CollectingQuestionRunEventSink()
    ports = replace(ports, progress_sink=events)

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue?",
            run_id="run_progress",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED"
    assert [
        (event["stage"], event["message"])
        for event in events.events
        if event["event"] == "run.progress"
    ] == [
        ("question_contract", "normalizing requested fact"),
        ("query_enrichment", "matching question terms to API resources"),
        ("grounding", "grounding question inputs"),
        ("read_eligibility", "selecting candidate reads"),
        ("plan_selection", "choosing answer strategy"),
        ("source_binding", "selecting source read"),
        ("fact_planning", "building answer plan"),
        ("execution", "reading source"),
    ]


def test_lookup_runtime_records_source_reads_against_canonical_execute_step():
    read_id = "sales_summary"
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id=read_id,
                    ),
                    fields=(
                        RelationField(
                            field_id="total_revenue",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="sales_summary",
                        fields=(
                            ProjectField(source="total_revenue", output="answer_1"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="total_revenue",
                    ),
                )
            ),
        )
    )
    catalog = _catalog(_sales_revenue_read(read_id))
    planner = _ReadEligibilityPlannerPort(plan=plan, eligible_read_id=read_id)
    data_access = _DataAccessPort({read_id: {"data": [{"total_revenue": "14.00"}]}})
    lineage = _LineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog),
        data_access_port=data_access,
        planner_model_port=planner,
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_source_read_lineage",
            recorder=lineage,
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue?",
            run_id="run_source_read_lineage",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    execute_steps = [
        step for step in lineage.steps if step.step_key is RunStepKey.EXECUTE
    ]
    assert result.status == "COMPLETED", result.error
    assert lineage.model_call_audits
    assert all(
        audit.model_call.run_id == "run_source_read_lineage"
        for audit in lineage.model_call_audits
    )
    assert all(audit.artifacts for audit in lineage.model_call_audits)
    assert len(execute_steps) == 1
    assert len(lineage.answered_results) == 1
    answered = lineage.answered_results[0]
    assert answered.result.run_id == "run_source_read_lineage"
    assert len(answered.requested_facts) == 1
    assert len(answered.fact_results) == 1
    assert len(answered.proof_graphs) == 1
    assert answered.outputs[0].output_key == "answer_1"
    assert answered.outputs[0].value_json == {
        "kind": "number",
        "value": "14.00",
    }
    assert execute_steps[0].output_summary_json["relationCount"] == 2
    assert [
        (
            item.catalog_endpoint_id,
            item.catalog_endpoint_key,
            item.endpoint_name,
            item.source_namespace_path_json,
            item.route_path_template,
            item.handler_ref,
        )
        for item in lineage.catalog_endpoints
    ] == [
        (
            lineage.catalog_endpoints[0].catalog_endpoint_id,
            f"django_sales_{read_id}:test",
            read_id,
            ("sales",),
            "/v1/sales/revenue/",
            "tests.SalesRevenueView",
        )
    ]
    UUID(lineage.catalog_endpoints[0].catalog_endpoint_id)
    assert [
        (item.catalog_endpoint_id, item.step_id, item.row_count)
        for item in lineage.source_reads
    ] == [
        (lineage.catalog_endpoints[0].catalog_endpoint_id, execute_steps[0].step_id, 1)
    ]
    assert (
        f"source_read:{lineage.source_reads[0].source_read_id}"
        in result.rendered_fact.proof_refs
    )  # type: ignore[union-attr]


def test_lookup_retains_ambiguous_read_eligibility_candidates_with_docstring_context():
    read_id = "sales_summary"
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id=read_id,
                    ),
                    fields=(
                        RelationField(
                            field_id="total_revenue",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="sales_summary",
                        fields=(
                            ProjectField(source="total_revenue", output="answer_1"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="total_revenue",
                    ),
                )
            ),
        )
    )
    catalog = _catalog(
        EndpointRead(
            id=read_id,
            endpoint_name="list_sales_summary",
            resource_names=("sales", "summary"),
            source_metadata={
                "description": (
                    "Aggregated sales summary for business questions. "
                    "Supports grouped revenue outputs."
                )
            },
            fields=(
                CatalogField(
                    ref=f"{read_id}.field.total_revenue",
                    path="data.total_revenue",
                    row_path_id="data",
                    type="decimal",
                ),
            ),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
        )
    )
    planner = _DocstringAmbiguousReadEligibilityPlannerPort(
        plan=plan,
        ambiguous_read_id=read_id,
        required_docstring_text="Aggregated sales summary",
    )
    data_access = _DataAccessPort(
        {"list_sales_summary": {"data": [{"total_revenue": "14.00"}]}}
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog),
        data_access_port=data_access,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue?",
            run_id="run_ambiguous_read_eligibility",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result.error
    assert data_access.requests == [{"endpointName": "list_sales_summary", "args": {}}]
    assert read_id in planner.source_binding_selection_prompt


def test_lookup_source_binding_can_bind_required_finite_choice_param():
    read_id = "sales_summary"
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id=read_id,
                        param_bindings=(
                            EndpointParamBinding(
                                param_id="group_by",
                                value_expr=ConstantRef(
                                    constant_id="group_by.location",
                                    version_ref="test-fixture-v1",
                                    value=FactValue.literal(
                                        id="group_by.location",
                                        literal_type=LiteralType.STRING,
                                        value="location",
                                    ),
                                ),
                            ),
                        ),
                    ),
                    fields=(
                        RelationField(
                            field_id="total_revenue",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="sales_summary",
                        fields=(
                            ProjectField(source="total_revenue", output="answer_1"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="total_revenue",
                    ),
                )
            ),
        )
    )
    catalog = _catalog(
        EndpointRead(
            id=read_id,
            endpoint_name=read_id,
            resource_names=("sales", "revenue"),
            params=(
                CatalogParam(
                    ref=f"{read_id}.query.group_by",
                    name="group_by",
                    source=ParamSource.QUERY,
                    type="choice",
                    required=True,
                    choices=("date", "location"),
                ),
            ),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref=f"{read_id}.field.total_revenue",
                    path="data.total_revenue",
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
    planner = _ReadEligibilityPlannerPort(
        plan=plan,
        eligible_read_id=read_id,
    )
    data_access = _DataAccessPort({read_id: {"data": [{"total_revenue": "14.00"}]}})
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog),
        data_access_port=data_access,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue by store?",
            run_id="run_required_choice_binding",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", (
        result.error,
        result.answer,
    )
    assert data_access.requests == [
        {
            "endpointName": read_id,
            "args": {f"{read_id}.query.group_by": "location"},
        }
    ]
    assert read_id in planner.source_binding_selection_prompt


def test_lookup_derives_finite_choice_membership_from_answer_population_tests():
    read_id = "sales_summary"
    question_contract_payload = {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [],
        "answer_requests": [
            {
                "answer_fact": "count of sales",
                "answer_expression": {"family": "list_rows"},
                "answer_subject": _answer_subject_payload("sales"),
                "answer_population": {
                    "population_label": "sales",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "pop_test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Does the row/value represent a sale?",
                            "owned_question_input_refs": [],
                        },
                        {
                            "test_id": "pop_test_3",
                            "kind": "NORMAL_INSTANCE_GUARD",
                            "polarity": "MUST_PASS",
                            "test_question": (
                                "Is this an actual business sale instance rather "
                                "than a draft, canceled, voided, failed, or raw "
                                "storage representation unless the user explicitly "
                                "requested that state?"
                            ),
                            "owned_question_input_refs": [],
                        },
                    ],
                },
                "answer_outputs": [{"description": "count", "role": "ANSWER_VALUE"}],
                "used_question_inputs": [],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }
    fact_plan_payload = {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "list_rows",
                    "source": {"kind": "read", "read_id": read_id},
                    "source_binding_id": "sb_1",
                    "output_fields": [{"field_id": "total_count"}],
                }
            ],
        }
    }

    def source_binding_payload(*, prompt, tool_specs):
        del tool_specs
        return source_binding_payload_from_fact_plan_with_invocation_overrides(
            fact_plan_payload,
            prompt=prompt,
            invocation_overrides=(
                {
                    "requested_fact_id": "fact_1",
                    "finite_choice_param_reviews": {
                        "status": _finite_choice_param_review(
                            _finite_choice_review(
                                choice="DRAFT",
                                normal_test_ids=("normal_instance_guard",),
                                test_effects={
                                    "subject_identity": "DOES_NOT_DECIDE_TEST",
                                    "normal_instance_guard": "CONFLICTS_WITH_TEST",
                                },
                            ),
                            _finite_choice_review(
                                choice="COMPLETED",
                                normal_test_ids=("normal_instance_guard",),
                                test_effects={
                                    "subject_identity": "DOES_NOT_DECIDE_TEST",
                                    "normal_instance_guard": "SATISFIES_TEST",
                                },
                            ),
                            _finite_choice_review(
                                choice="CANCELED",
                                normal_test_ids=("normal_instance_guard",),
                                test_effects={
                                    "subject_identity": "DOES_NOT_DECIDE_TEST",
                                    "normal_instance_guard": "CONFLICTS_WITH_TEST",
                                },
                            ),
                        )
                    },
                },
            ),
        )

    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id=read_id,
                measured_value_fields=("total_count",),
            ),
        ),
        responses={
            "submit_question_contract_outcome": question_contract_payload,
            "submit_query_enrichment": _query_enrichment_payload(("sales",)),
            "submit_source_binding": source_binding_payload,
            "submit_pattern_fact_plan": fact_plan_payload,
        },
    )
    data_access = _DataAccessPort(
        {"list_sales_summary": {"data": [{"total_count": 14}]}}
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                EndpointRead(
                    id=read_id,
                    endpoint_name="list_sales_summary",
                    resource_names=("sales", "summary"),
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    params=(
                        CatalogParam(
                            ref="list_sales_summary.query.status",
                            name="status",
                            source=ParamSource.QUERY,
                            type="string",
                            choices=("DRAFT", "COMPLETED", "CANCELED"),
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.total_count",
                            path="data.total_count",
                            row_path_id="data",
                            type="number",
                        ),
                        CatalogField(
                            ref="field.status",
                            path="data.status",
                            row_path_id="data",
                            type="string",
                        ),
                    ),
                    facts=(CatalogFact(ref="sales.count"),),
                    pagination=PaginationMetadata(
                        mode=PaginationMode.NONE,
                        completeness_policy=CompletenessPolicy.COMPLETE,
                    ),
                )
            )
        ),
        data_access_port=data_access,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many sales are there?",
            run_id="run_answer_population_tests",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result.error
    assert data_access.requests == [
        {
            "endpointName": "list_sales_summary",
            "args": {"list_sales_summary.query.status": "COMPLETED"},
        }
    ]


def _finite_choice_review(
    *,
    choice: str,
    test_effects: dict[str, str],
    normal_test_ids: tuple[str, ...] = ("normal_instance_guard",),
) -> dict[str, object]:
    return {
        "choice_option_id": choice,
        "choice_domain_meaning": f"{choice} sales",
        "choice_inclusion_basis": f"{choice} is reviewed for inclusion.",
        "choice_inclusion": (
            "EXCLUDE"
            if any(v == "CONFLICTS_WITH_TEST" for v in test_effects.values())
            else "INCLUDE"
        ),
        "population_test_results": {
            test_id: {
                **(
                    {
                        "role_match_basis": (
                            f"{choice} is compared to the excluded "
                            "normal-instance roles."
                        ),
                        "explicit_user_override_evidence": [],
                        "explicit_user_override_applies": False,
                        "population_consequence": (
                            f"{choice} effect for {test_id} is {test_effect}."
                        ),
                        "disposition": {
                            "matched_excluded_role": (
                                "NONE"
                                if test_effect != "CONFLICTS_WITH_TEST"
                                else NormalInstanceExcludedStateRole.CANCELED_OR_VOIDED.value
                            ),
                            "test_effect": test_effect,
                        },
                    }
                    if test_id in normal_test_ids
                    else {
                        "test_basis": f"{choice} effect for {test_id} is {test_effect}.",
                        "population_consequence": (
                            f"{choice} effect for {test_id} is {test_effect}."
                        ),
                        "test_effect": test_effect,
                    }
                ),
            }
            for test_id, test_effect in test_effects.items()
        },
    }


def _finite_choice_param_review(
    *choice_reviews: dict[str, object],
) -> dict[str, object]:
    return {
        "controlled_population_role_id": "role_1",
        "role_selection_basis": "The param controls returned rows.",
        "population_test_basis": {
            "subject_identity": {
                "test_question": "Does this choice pass subject identity?",
                "role_scoped_test_question": (
                    "For returned rows, does this choice pass subject identity?"
                ),
            },
            "normal_instance_guard": {
                "test_question": "Does this choice pass the normal instance guard?",
                "role_scoped_test_question": (
                    "For returned rows, does this choice pass the normal instance guard?"
                ),
            },
        },
        "choice_reviews": list(choice_reviews),
    }


def test_lookup_allows_omitted_optional_default_finite_choice_param():
    read_id = "sales_summary"
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_summary",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id=read_id,
                    ),
                    fields=(
                        RelationField(
                            field_id="total_revenue",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="sales_summary",
                        fields=(
                            ProjectField(source="total_revenue", output="answer_1"),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer_1",
                        relation_id="answer_rows",
                        field_id="total_revenue",
                    ),
                )
            ),
        )
    )
    catalog = _catalog(
        EndpointRead(
            id=read_id,
            endpoint_name=read_id,
            resource_names=("sales", "revenue"),
            params=(
                CatalogParam(
                    ref=f"{read_id}.query.granularity",
                    name="granularity",
                    source=ParamSource.QUERY,
                    type="choice",
                    required=False,
                    choices=("day", "week", "month"),
                    default="day",
                ),
            ),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref=f"{read_id}.field.total_revenue",
                    path="data.total_revenue",
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
    planner = _OmitOptionalDefaultChoicePlannerPort(
        plan=plan,
        eligible_read_id=read_id,
    )
    data_access = _DataAccessPort({read_id: {"data": [{"total_revenue": "14.00"}]}})
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog),
        data_access_port=data_access,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue?",
            run_id="run_optional_default_choice_omitted",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", (
        result.error,
        result.answer,
    )
    assert data_access.requests == [
        {
            "endpointName": read_id,
            "args": {f"{read_id}.query.granularity": "day"},
        }
    ]


def test_lookup_cutover_runs_single_fact_plan_then_execution_then_response_rendering():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="metric_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="metric_read",
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
                        input_relation="metric_rows",
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
        catalog=_catalog(
            EndpointRead(
                id="metric_read",
                endpoint_name="metric_read",
                resource_names=("metric read",),
                row_paths=(
                    RowPath(
                        id="root",
                        path="",
                        cardinality=RowCardinality.ONE,
                    ),
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
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
        ),
        responses={
            "metric_read": {
                "data": [{"location_name": "Location Alpha", "metric_total": "125.00"}]
            }
        },
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was the metric total at Location Alpha?",
            run_id="run_direct",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "Location Alpha: 125.00"
    assert ports.planner_model_port.calls == 6
    assert result.rendered_fact.rows == (  # type: ignore[union-attr]
        {"answer_1": "Location Alpha", "answer_2": Decimal("125.00")},
    )


def _sales_revenue_read(read_id: str) -> EndpointRead:
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        resource_names=("sales", "revenue"),
        catalog_endpoint=_catalog_endpoint_metadata(
            endpoint_name=read_id,
            namespace_path=("sales",),
            route_path_template="/v1/sales/revenue/",
            handler_ref="tests.SalesRevenueView",
            domain_resource_names=("sales", "revenue"),
        ),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref=f"{read_id}.field.total_revenue",
                path="data.total_revenue",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _catalog_endpoint_metadata(
    *,
    endpoint_name: str,
    namespace_path: tuple[str, ...],
    route_path_template: str,
    handler_ref: str,
    domain_resource_names: tuple[str, ...],
) -> CatalogEndpointMetadata:
    return CatalogEndpointMetadata(
        catalog_endpoint_key=f"django_sales_{endpoint_name}:test",
        endpoint_name=endpoint_name,
        framework_kind="django_drf",
        source_namespace_kind="django_app",
        source_namespace_path=namespace_path,
        route_method="GET",
        route_path_template=route_path_template,
        handler_ref=handler_ref,
        domain_resource_names=domain_resource_names,
    )


@dataclass
class _LineageRecorder:
    steps: list[RunStepWrite] = field(default_factory=list)
    catalog_endpoints: list[CatalogEndpointWrite] = field(default_factory=list)
    source_reads: list[SourceReadWrite] = field(default_factory=list)
    answered_results: list[AnsweredRunResultWrite] = field(default_factory=list)
    terminal_results: list[FactualTerminalRunResultWrite] = field(default_factory=list)
    runtime_error_results: list[RuntimeErrorResultWrite] = field(default_factory=list)
    model_call_audits: list[object] = field(default_factory=list)
    program_invocations: list[object] = field(default_factory=list)

    def record_step(self, step: RunStepWrite) -> RunStepWrite:
        self.steps.append(step)
        return step

    def record_catalog_endpoint(
        self,
        catalog_endpoint: CatalogEndpointWrite,
    ) -> CatalogEndpointWrite:
        self.catalog_endpoints.append(catalog_endpoint)
        return catalog_endpoint

    def record_step_with_source_context(
        self,
        step: RunStepWrite,
        catalog_endpoints: tuple[CatalogEndpointWrite, ...],
        source_reads: tuple[SourceReadWrite, ...],
    ) -> RunStepWrite:
        self.catalog_endpoints.extend(catalog_endpoints)
        self.steps.append(step)
        self.source_reads.extend(source_reads)
        return step

    def record_answered_result(
        self,
        answered_result: AnsweredRunResultWrite,
    ) -> AnsweredRunResultWrite:
        self.answered_results.append(answered_result)
        return answered_result

    def record_factual_terminal_result(
        self,
        terminal_result: FactualTerminalRunResultWrite,
    ) -> FactualTerminalRunResultWrite:
        self.terminal_results.append(terminal_result)
        return terminal_result

    def record_runtime_error_result(
        self,
        runtime_error: RuntimeErrorResultWrite,
    ) -> RuntimeErrorResultWrite:
        self.runtime_error_results.append(runtime_error)
        return runtime_error

    def record_model_call_audit(self, audit: object) -> object:
        self.model_call_audits.append(audit)
        return audit

    def record_program_invocation(self, invocation: object) -> object:
        self.program_invocations.append(invocation)
        return invocation


class _FailingRuntimeErrorLineageRecorder(_LineageRecorder):
    def record_runtime_error_result(
        self,
        runtime_error: RuntimeErrorResultWrite,
    ) -> RuntimeErrorResultWrite:
        del runtime_error
        raise LineagePersistenceUnavailable("runtime error lineage unavailable")


class _FailingModelCallAuditLineageRecorder(_LineageRecorder):
    def record_model_call_audit(self, audit: object) -> object:
        del audit
        raise LineagePersistenceUnavailable("model call audit unavailable")


class _ProviderAuthError(RuntimeError):
    code = common_error_codes.LLM_API_AUTHENTICATION_ERROR
    context = {"provider": "opencode", "reason": "authentication failed"}


class _ProviderAuthFailurePlannerPort:
    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise _ProviderAuthError("authentication failed")


def _read_eligibility_fact_context(
    prompt: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    requested_facts = _prompt_json_section(prompt, "Requested facts")["requested_facts"]
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for fact in requested_facts:
        fact_id = str(fact.get("requested_fact_id") or "")
        answer_request = fact.get("answer_request") or {}
        answer_population = answer_request.get("answer_population") or {}
        output[fact_id] = {
            "answer_output_ids": tuple(
                str(item.get("answer_output_id") or "")
                for item in fact.get("answer_outputs") or ()
                if str(item.get("answer_output_id") or "")
            ),
            "membership_test_ids": tuple(
                str(item.get("test_id") or "")
                for item in answer_population.get("membership_tests") or ()
                if str(item.get("test_id") or "")
            ),
        }
    return output


def _test_retained_read_candidate(
    *,
    card: dict[str, object],
    fact_context: dict[str, tuple[str, ...]],
    field_paths: tuple[str, ...],
    retention_basis: str,
) -> dict[str, object]:
    del fact_context
    source_candidate_id = str(card["source_candidate_id"])
    read_id = str(card["read_id"])
    tokens_by_path = {
        str(field.get("path") or ""): str(field.get("evidence_token") or "")
        for row in card.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    }
    field_tokens = [
        tokens_by_path[field_path]
        for field_path in field_paths
        if tokens_by_path.get(field_path)
    ]
    return {
        "source_candidate_id": source_candidate_id,
        "read_id": read_id,
        "relevant_row_path_tokens": [],
        "relevant_field_tokens": field_tokens,
        "retention_basis": retention_basis,
        "retention_decision": "RETAIN",
    }


def _read_eligibility_assessments_by_requested_fact(
    *,
    requested_fact_id: str,
    fact_context: dict[str, tuple[str, ...]],
    candidate_cards: list[dict[str, object]],
    retained_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    retained_by_source_id = {
        str(candidate["source_candidate_id"]): candidate
        for candidate in retained_candidates
    }
    del fact_context
    read_candidate_reviews = []
    for card in candidate_cards:
        source_candidate_id = str(card["source_candidate_id"])
        retained = retained_by_source_id.get(source_candidate_id)
        if retained is not None:
            read_candidate_reviews.append(retained)
            continue
        read_candidate_reviews.append(
            {
                "source_candidate_id": source_candidate_id,
                "read_id": str(card["read_id"]),
                "relevant_row_path_tokens": [],
                "relevant_field_tokens": [],
                "retention_basis": "This read was not retained for the requested fact.",
                "retention_decision": "DROP",
            }
        )
    return [
        {
            "requested_fact_id": requested_fact_id,
            "read_candidate_reviews": read_candidate_reviews,
        }
    ]


@dataclass
class _ReadEligibilityPlannerPort(_PlannerPort):
    eligible_read_id: str = ""
    read_eligibility_prompt: str = ""

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: Any = None,
        tool_specs: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name != "submit_read_eligibility":
            return super().generate(
                provider=provider,
                prompt=prompt,
                max_thinking_tokens=max_thinking_tokens,
                system_prompt=system_prompt,
                output_mode=output_mode,
                tool_specs=tool_specs,
            )
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.prompts.append(prompt)
        self.read_eligibility_prompt = prompt
        read_cards = _prompt_json_section(prompt, "Candidate API reads")[
            "requested_fact_read_candidates"
        ]
        facts_by_id = _read_eligibility_fact_context(prompt)
        requested_fact_assessments = []
        for fact_group in read_cards:
            requested_fact_id = fact_group["requested_fact_id"]
            fact_context = facts_by_id[requested_fact_id]
            candidate_cards = list(fact_group["read_candidates"])
            retained_candidates = []
            for card in candidate_cards:
                read_id = card["read_id"]
                eligible = read_id == self.eligible_read_id
                field_paths = [
                    field["path"]
                    for row in card.get("response_rows") or ()
                    if isinstance(row, dict)
                    for field in row.get("fields") or ()
                    if isinstance(field, dict) and field.get("path")
                ]
                if eligible and field_paths:
                    retained_candidates.append(
                        _test_retained_read_candidate(
                            card=card,
                            fact_context=fact_context,
                            field_paths=tuple(field_paths),
                            retention_basis=(
                                "total_revenue provides the requested revenue."
                            ),
                        )
                    )
            requested_fact_assessments.extend(
                _read_eligibility_assessments_by_requested_fact(
                    requested_fact_id=requested_fact_id,
                    fact_context=fact_context,
                    candidate_cards=candidate_cards,
                    retained_candidates=retained_candidates,
                )
            )
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_read_eligibility",
                    "arguments": {
                        "requested_fact_assessments": requested_fact_assessments,
                    },
                },
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


@dataclass
class _DocstringAmbiguousReadEligibilityPlannerPort(_ReadEligibilityPlannerPort):
    ambiguous_read_id: str = ""
    required_docstring_text: str = ""

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: Any = None,
        tool_specs: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name != "submit_read_eligibility":
            return super().generate(
                provider=provider,
                prompt=prompt,
                max_thinking_tokens=max_thinking_tokens,
                system_prompt=system_prompt,
                output_mode=output_mode,
                tool_specs=tool_specs,
            )
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.prompts.append(prompt)
        self.read_eligibility_prompt = prompt
        read_cards = _prompt_json_section(prompt, "Candidate API reads")[
            "requested_fact_read_candidates"
        ]
        facts_by_id = _read_eligibility_fact_context(prompt)
        requested_fact_assessments = []
        for fact_group in read_cards:
            requested_fact_id = fact_group["requested_fact_id"]
            fact_context = facts_by_id[requested_fact_id]
            candidate_cards = list(fact_group["read_candidates"])
            retained_candidates = []
            for card in candidate_cards:
                read_id = card["read_id"]
                field_paths = [
                    field["path"]
                    for row in card.get("response_rows") or ()
                    if isinstance(row, dict)
                    for field in row.get("fields") or ()
                    if isinstance(field, dict) and field.get("path")
                ]
                docstring_matches = (
                    read_id == self.ambiguous_read_id
                    and self.required_docstring_text in str(card.get("docstring") or "")
                )
                if docstring_matches and field_paths:
                    retained_candidates.append(
                        _test_retained_read_candidate(
                            card=card,
                            fact_context=fact_context,
                            field_paths=tuple(field_paths),
                            retention_basis=(
                                "The docstring and fields show plausible support."
                            ),
                        )
                    )
            requested_fact_assessments.extend(
                _read_eligibility_assessments_by_requested_fact(
                    requested_fact_id=requested_fact_id,
                    fact_context=fact_context,
                    candidate_cards=candidate_cards,
                    retained_candidates=retained_candidates,
                )
            )
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_read_eligibility",
                    "arguments": {
                        "requested_fact_assessments": requested_fact_assessments,
                    },
                },
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


@dataclass
class _OmitOptionalDefaultChoicePlannerPort(_ReadEligibilityPlannerPort):
    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: Any = None,
        tool_specs: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name != "submit_source_binding":
            return super().generate(
                provider=provider,
                prompt=prompt,
                max_thinking_tokens=max_thinking_tokens,
                system_prompt=system_prompt,
                output_mode=output_mode,
                tool_specs=tool_specs,
            )
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.prompts.append(prompt)
        self.source_binding_selection_prompt = prompt
        binding_target_id = source_binding_target_id_for_candidate(
            prompt,
            requested_fact_id="fact_1",
            source_candidate_id="source_1",
            plan_shape="list_rows",
        )
        arguments = source_binding_payload_for_one_call(
            {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": {
                                "population_binding_id": (
                                    "pop.source_1.candidate_population"
                                ),
                                "intent_text": "What was total sales revenue?",
                                "match_basis_explanation": (
                                    "The selected source provides the requested revenue."
                                ),
                            },
                            "fulfillment_decisions": {
                                "answer_1": {
                                    "match_basis_explanation": (
                                        "answer_1 is fulfilled by the total_revenue field."
                                    ),
                                    "fulfillment_choice_id": (
                                        "source_1.data.total_revenue"
                                    ),
                                }
                            },
                            "param_decisions": {},
                            "row_predicate_reviews": {},
                            "finite_choice_param_reviews": {},
                        },
                    },
                }
            },
            prompt=prompt,
        )
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_source_binding",
                    "arguments": arguments,
                },
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


def test_lookup_unresolved_named_entity_returns_resource_specific_clarification():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="buyer feedback reasons for store choice",
                answer_subject="buyer feedback reasons",
                parts=("buyer feedback reasons",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "Nextgen",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "store",
                        "resolved_value_text": "Nextgen",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "location can identify Nextgen because value_meaning_hint is store.",
                                "term": "location",
                            }
                        ],
                    }
                ],
            ),
            "submit_grounding": {
                "known_time_resolutions": {},
                "known_input_bindings": {
                    "fact_1_entity_1": {
                        "selected_option_id": "bind_fact_1_entity_1_1",
                        "input_value": "Nextgen",
                        "result_kind": "canonical_identity",
                        "matched_field_ref": "field.name",
                        "selection_basis": "The selected location name field is the supplied store reference.",
                    }
                },
            },
        }
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                EndpointRead(
                    id="locations",
                    endpoint_name="list_location_list",
                    resource_names=("location",),
                    params=(
                        CatalogParam(
                            ref="list_location_list.query.name",
                            name="name",
                            source=ParamSource.QUERY,
                            type="string",
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
                            type="string",
                        ),
                        CatalogField(
                            ref="field.name",
                            path="data.name",
                            row_path_id="data",
                            type="string",
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
                            context_field_refs=("field.name",),
                        ),
                    ),
                )
            )
        ),
        data_access_port=_DataAccessPort({"list_location_list": {"data": []}}),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question=(
                "In the available buyer and order data, is there a field that "
                "records explicit buyer feedback reasons for store choice between "
                "ABC Mall and Nextgen?"
            ),
            run_id="run_unresolved_named_entity_clarification",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "NEEDS_CLARIFICATION", result
    assert (
        result.answer == 'I could not find store "Nextgen". Which store should I use?'
    )
    assert planner.tool_names == [
        "submit_question_contract_outcome",
        "submit_query_enrichment",
        "submit_grounding",
    ]


def test_lookup_grounding_keeps_identity_list_resolver_visible_with_noisy_entity_terms():
    class _NoisyResolverPlannerPort(_ToolNamePlannerPort):
        def generate(self, **kwargs):  # type: ignore[no-untyped-def]
            tool_specs = kwargs.get("tool_specs") or ()
            tool_name = tool_specs[0].name if tool_specs else ""
            if tool_name == "submit_source_binding":
                from tests.lookup.source_binding_helpers import (
                    source_candidate_answer_population,
                    source_candidate_with_fields,
                    source_fulfills_for_candidate,
                    source_binding_target_id_for_candidate,
                )

                prompt = str(kwargs.get("prompt") or "")
                candidate = source_candidate_with_fields(
                    prompt,
                    required=("staff_id",),
                )
                binding_target_id = source_binding_target_id_for_candidate(
                    prompt,
                    requested_fact_id="fact_1",
                    source_candidate_id=str(candidate["source_candidate_id"]),
                    plan_shape="list_rows",
                )
                arguments = {
                    "outcome": {
                        "kind": "source_bindings",
                        "bindings_for_fact_1": {
                            "plan_shape": "list_rows",
                            "primary": {
                                "binding_target_id": binding_target_id,
                                "answer_population": source_candidate_answer_population(
                                    prompt,
                                    source_candidate_id=candidate[
                                        "source_candidate_id"
                                    ],
                                ),
                                "fulfillment_decisions": source_fulfills_for_candidate(
                                    candidate,
                                    field_ids=("staff_id",),
                                ),
                                "param_decisions": {},
                            },
                        },
                    }
                }
                arguments = source_binding_payload_for_one_call(
                    arguments,
                    prompt=prompt,
                )
                self.calls += 1
                self.system_prompts.append(str(kwargs.get("system_prompt") or ""))
                self.prompts.append(prompt)
                self.tool_names.append(tool_name)
                self.source_binding_selection_prompt = prompt
                return {
                    "answer": json.dumps(
                        {"tool": tool_name, "arguments": arguments},
                        default=str,
                    ),
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "thinkingTokens": 0,
                        "costUsd": 0,
                    },
                }
            return super().generate(**kwargs)

    planner = _NoisyResolverPlannerPort(
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="Jane Doe staff ID",
                answer_subject="staff ID",
                answer_expression_family="list_rows",
                parts=("staff ID",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "Jane Doe",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "staff member",
                        "resolved_value_text": "Jane Doe",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("staff",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "staff can identify Jane Doe because value_meaning_hint is staff member.",
                                "term": "staff",
                            },
                        ],
                    }
                ],
            ),
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1"],
                            "pattern": "list_rows",
                            "source": {
                                "kind": "read",
                                "read_id": "staff_list",
                            },
                            "output_fields": [{"field_id": "staff_id"}],
                        }
                    ],
                }
            },
        }
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                _required_identity_detail_read(
                    "staff_detail", "get_staff_detail", "staff"
                ),
                _required_identity_detail_read(
                    "area_detail", "get_area_detail", "area"
                ),
                _required_identity_detail_read(
                    "cash_deposit_detail",
                    "get_cash_deposit_detail",
                    "cash_deposit",
                ),
                _required_identity_detail_read(
                    "deal_detail", "get_deal_detail", "deal"
                ),
                _required_identity_detail_read(
                    "look_detail", "get_look_detail", "look"
                ),
                _staff_identity_list_read(),
            )
        ),
        data_access_port=_DataAccessPort(
            {
                "list_staff_list": {
                    "data": [
                        {
                            "staff_id": "staff-1",
                            "full_name": "Jane Doe",
                        }
                    ]
                }
            }
        ),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is Jane Doe's staff ID?",
            run_id="run_noisy_entity_resolver_terms",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert planner.tool_names == [
        "submit_question_contract_outcome",
        "submit_query_enrichment",
        "submit_grounding",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
        "submit_pattern_fact_plan",
    ]
    assert ports.data_access_port.requests == [
        {
            "endpointName": "list_staff_list",
            "args": {},
        },
        {
            "endpointName": "list_staff_list",
            "args": {},
        },
    ]


def test_lookup_grounding_executes_ambiguous_resolver_routes_before_source_binding():
    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                row_path_ids=("data",),
            ),
        ),
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="sales at ABC Mall",
                answer_subject="sales",
                parts=("sales",),
                answer_output_role="ROW_COUNT",
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "ABC Mall",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "store location",
                        "resolved_value_text": "ABC Mall",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("sales",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "location can identify ABC Mall because value_meaning_hint is store location.",
                                "term": "location",
                            },
                        ],
                    }
                ],
            ),
            "submit_source_binding": {
                "outcome": {
                    "kind": "impossible",
                    "blocked_facts": [
                        {
                            "requested_fact_id": "fact_1",
                            "basis": "policy_access",
                            "evidence_refs": ["policy:sales_records_restricted"],
                            "reviewed_read_ids": ["sales"],
                            "nearest_fields": [
                                {"read_id": "sales", "field_id": "sale_id"}
                            ],
                            "explanation": "The test stops after grounding resolves the named entity.",
                        }
                    ],
                }
            },
        },
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                _identity_list_read(
                    "shade_store_availability",
                    "get_merch_shade_store_availability",
                    "store",
                    extra_identity_refs=("location",),
                ),
                _identity_list_read("store_list", "list_store_list", "store"),
                _identity_list_read(
                    "deal_location_limits",
                    "get_deal_location_limits",
                    "deal_location_limit",
                    extra_identity_refs=("deal", "location"),
                ),
                _identity_list_read("deal_list", "list_deal_list", "deal"),
                _identity_list_read("location_list", "list_location_list", "location"),
                EndpointRead(
                    id="sales",
                    endpoint_name="list_sale_list",
                    resource_names=("sales",),
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    params=(
                        CatalogParam(
                            ref="list_sale_list.query.location_id",
                            name="location_id",
                            source=ParamSource.QUERY,
                            type="string",
                            entity_target=EntityKeyComponentTarget(
                                entity_kind="location",
                                key_id="primary_key",
                                component_id="location_id",
                            ),
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.sale_id",
                            path="data.sale_id",
                            row_path_id="data",
                            type="string",
                        ),
                    ),
                    facts=(
                        CatalogFact(
                            ref="sales.records",
                            availability=CatalogFactAvailability.POLICY_BLOCKED,
                            read_id="sales",
                            proof_refs=("policy:sales_records_restricted",),
                        ),
                    ),
                ),
            )
        ),
        data_access_port=_DataAccessPort(
            {
                "get_merch_shade_store_availability": {"data": []},
                "list_store_list": {"data": []},
                "get_deal_location_limits": {"data": []},
                "list_location_list": {
                    "data": [{"location_id": "loc-1", "name": "ABC Mall"}]
                },
            }
        ),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What were sales at ABC Mall?",
            run_id="run_store_location_resolver_ambiguity",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert planner.tool_names == [
        "submit_question_contract_outcome",
        "submit_query_enrichment",
        "submit_grounding",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
    ]
    requested_endpoints = {
        item["endpointName"] for item in ports.data_access_port.requests
    }
    assert "list_location_list" in requested_endpoints
    assert len(requested_endpoints) <= 3


def test_lookup_runtime_records_grounding_resolver_source_reads() -> None:
    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="staff_list",
            ),
        ),
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="Jane Doe staff ID",
                answer_subject="staff ID",
                answer_expression_family="list_rows",
                parts=("staff ID",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "Jane Doe",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "staff member",
                        "resolved_value_text": "Jane Doe",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("staff",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "staff can identify Jane Doe because value_meaning_hint is staff member.",
                                "term": "staff",
                            },
                        ],
                    }
                ],
            ),
            "submit_grounding": {
                "known_time_resolutions": {},
                "known_input_bindings": {
                    "fact_1_entity_1": {
                        "selected_option_id": "bind_fact_1_entity_1_1",
                        "input_value": "Jane Doe",
                        "result_kind": "canonical_identity",
                        "matched_field_ref": "staff.field.full_name",
                        "selection_basis": "The staff resolver returns the staff identity named by Jane Doe.",
                    }
                },
            },
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1"],
                            "pattern": "list_rows",
                            "source": {
                                "kind": "read",
                                "read_id": "staff_list",
                            },
                            "output_fields": [{"field_id": "staff_id"}],
                        }
                    ],
                }
            },
        },
    )
    lineage = _LineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_catalog(_staff_identity_list_read())),
        data_access_port=_DataAccessPort(
            {
                "list_staff_list": {
                    "data": [
                        {
                            "staff_id": "staff-1",
                            "full_name": "Jane Doe",
                        }
                    ]
                }
            }
        ),
        planner_model_port=planner,
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_grounding_source_read_lineage",
            recorder=lineage,
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is Jane Doe's staff ID?",
            run_id="run_grounding_source_read_lineage",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", (result, result.error)
    source_reads_by_step_key = {step.step_id: step.step_key for step in lineage.steps}
    assert [
        (
            source_reads_by_step_key[item.step_id],
            item.catalog_endpoint_id,
            item.row_count,
        )
        for item in lineage.source_reads
    ] == [
        (
            RunStepKey.GROUNDING,
            lineage.catalog_endpoints[0].catalog_endpoint_id,
            1,
        ),
        (
            RunStepKey.EXECUTE,
            lineage.catalog_endpoints[0].catalog_endpoint_id,
            1,
        ),
    ]
    latest_steps_by_id = {step.step_id: step for step in lineage.steps}
    assert [
        item.payload
        for step in latest_steps_by_id.values()
        if step.step_key is RunStepKey.GROUNDING
        for item in step_semantic_items_from_json(step.output_summary_json)
        if item.kind == "grounding_result"
    ] == [
        {
            "input_id": "fact_1_entity_1",
            "input_text": "Jane Doe",
            "resolver_read_id": "staff_list",
            "resolver_label": "Staff List",
            "entity_kind": "staff",
            "key_id": "staff_key",
            "key_components": [{"component_id": "staff_id", "value": "staff-1"}],
            "matched_label": "Jane Doe",
        }
    ]


def test_lookup_runtime_fails_closed_on_grounding_resolver_source_read_failure() -> (
    None
):
    planner = _ToolNamePlannerPort(
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="Jane Doe staff ID",
                answer_subject="staff ID",
                parts=("staff ID",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "Jane Doe",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "staff member",
                        "resolved_value_text": "Jane Doe",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("staff",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "staff can identify Jane Doe because value_meaning_hint is staff member.",
                                "term": "staff",
                            },
                        ],
                    }
                ],
            ),
            "submit_grounding": {
                "known_time_resolutions": {},
                "known_input_bindings": {
                    "fact_1_entity_1": {
                        "selected_option_id": "bind_fact_1_entity_1_1",
                        "input_value": "Jane Doe",
                        "result_kind": "canonical_identity",
                        "matched_field_ref": "staff.field.full_name",
                        "selection_basis": "The staff resolver returns the staff identity named by Jane Doe.",
                    }
                },
            },
        },
    )
    lineage = _LineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_catalog(_staff_identity_list_read())),
        data_access_port=_StatusDataAccessPort(
            {
                "list_staff_list": {
                    "endpointName": "list_staff_list",
                    "responseStatus": 500,
                    "responseBody": {"error": "database unavailable"},
                }
            }
        ),
        planner_model_port=planner,
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_grounding_source_read_failure",
            recorder=lineage,
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is Jane Doe's staff ID?",
            run_id="run_grounding_source_read_failure",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "FAILED", result
    assert result.error == "framework_adapter_failed"
    source_reads_by_step_key = {step.step_id: step.step_key for step in lineage.steps}
    assert [
        (source_reads_by_step_key[item.step_id], item.status, item.row_count)
        for item in lineage.source_reads
    ] == [(RunStepKey.GROUNDING, SourceReadStatus.FAILED, None)]
    assert lineage.runtime_error_results
    assert lineage.runtime_error_results[0].error.failed_step_id in {
        step.step_id for step in lineage.steps if step.step_key is RunStepKey.GROUNDING
    }


def test_lookup_runtime_fails_closed_on_grounding_missing_catalog_endpoint() -> None:
    planner = _ToolNamePlannerPort(
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="Jane Doe staff ID",
                answer_subject="staff ID",
                parts=("staff ID",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "Jane Doe",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "staff member",
                        "resolved_value_text": "Jane Doe",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("staff",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "staff can identify Jane Doe because value_meaning_hint is staff member.",
                                "term": "staff",
                            },
                        ],
                    }
                ],
            ),
            "submit_grounding": {
                "known_time_resolutions": {},
                "known_input_bindings": {
                    "fact_1_entity_1": {
                        "selected_option_id": "bind_fact_1_entity_1_1",
                        "input_value": "Jane Doe",
                        "result_kind": "canonical_identity",
                        "matched_field_ref": "staff.field.full_name",
                        "selection_basis": "The staff resolver returns the staff identity named by Jane Doe.",
                    }
                },
            },
        },
    )
    lineage = _LineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(_staff_identity_list_read_without_catalog_endpoint())
        ),
        data_access_port=_DataAccessPort({}),
        planner_model_port=planner,
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_grounding_missing_catalog_endpoint",
            recorder=lineage,
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What is Jane Doe's staff ID?",
            run_id="run_grounding_missing_catalog_endpoint",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "FAILED", result
    assert result.error == "framework_adapter_failed"
    assert not lineage.source_reads
    assert lineage.runtime_error_results


def test_model_turn_failure_preserves_provider_error_when_terminal_lineage_fails() -> (
    None
):
    lineage = _FailingRuntimeErrorLineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_catalog()),
        data_access_port=_DataAccessPort({}),
        planner_model_port=_ProviderAuthFailurePlannerPort(),
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_provider_auth_lineage_failure",
            recorder=lineage,
        ),
        lineage_required=True,
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many orders came in today?",
            run_id="run_provider_auth_lineage_failure",
            tenant_id="tenant_1",
            provider_preferences={"provider": "opencode", "modelKey": "opencode:x"},
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "provider_authentication_failed"
    assert lineage.steps[0].error_json["errorCode"] == "provider_authentication_failed"


def test_model_turn_failure_preserves_provider_error_when_audit_lineage_fails() -> None:
    lineage = _FailingModelCallAuditLineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_catalog()),
        data_access_port=_DataAccessPort({}),
        planner_model_port=_ProviderAuthFailurePlannerPort(),
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_provider_auth_audit_failure",
            recorder=lineage,
        ),
        lineage_required=True,
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many orders came in today?",
            run_id="run_provider_auth_audit_failure",
            tenant_id="tenant_1",
            provider_preferences={"provider": "opencode", "modelKey": "opencode:x"},
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "provider_authentication_failed"
    assert lineage.runtime_error_results


def test_model_turn_failure_records_text_runtime_error_message() -> None:
    lineage = _LineageRecorder()
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_catalog()),
        data_access_port=_DataAccessPort({}),
        planner_model_port=_ProviderAuthFailurePlannerPort(),
        lineage_step_sink=LineageRuntimeStepSink(
            run_id="run_provider_auth_text_error",
            recorder=lineage,
        ),
        lineage_required=True,
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many orders came in today?",
            run_id="run_provider_auth_text_error",
            tenant_id="tenant_1",
            provider_preferences={"provider": "opencode", "modelKey": "opencode:x"},
        ),
        ports,
    )

    assert result.status == "FAILED"
    assert result.error == "provider_authentication_failed"
    assert lineage.runtime_error_results
    message = lineage.runtime_error_results[0].error.message
    assert isinstance(message, str)
    assert "provider_authentication_failed" in message
    assert "opencode" in message


def test_lookup_cutover_runs_combined_source_and_split_planning_turns():
    ports = _ports(
        plan=FactPlan(
            outcome=_answer_plan(
                relations=(
                    Relation(
                        id="rows",
                        source=RelationSource(
                            kind=SourceKind.API_READ,
                            read_id="sales",
                        ),
                        fields=(
                            RelationField(
                                field_id="amount",
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
                            fields=(ProjectField(source="amount"),),
                        ),
                        output_relation="answer_rows",
                    ),
                ),
                result_projection=ResultProjection(
                    relation_outputs=(
                        RelationResultOutput(
                            id="amount",
                            relation_id="answer_rows",
                            field_id="amount",
                        ),
                    )
                ),
            )
        ),
        catalog=_catalog(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sales",),
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
                ),
            )
        ),
        responses={"list_sale_list": {"data": [{"amount": "50000"}]}},
    )

    def source_binding_payload(*, prompt, tool_specs):
        del tool_specs
        binding_target_id = source_binding_target_id_for_candidate(
            prompt,
            requested_fact_id="fact_1",
            source_candidate_id="source_1",
            plan_shape="list_rows",
        )
        return source_binding_payload_for_one_call(
            {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": {
                                "population_binding_id": "pop.source_1.candidate_population",
                                "intent_text": "sales",
                                "match_basis_explanation": "sales defines the source population",
                            },
                            "fulfillment_decisions": {
                                "answer_1": {
                                    "match_basis_explanation": (
                                        "answer_1 is fulfilled by source_1.data.amount "
                                        "because that source evidence provides the "
                                        "requested output."
                                    ),
                                    "fulfillment_choice_id": "source_1.data.amount",
                                }
                            },
                            "param_decisions": {},
                        },
                    },
                }
            },
            prompt=prompt,
        )

    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                answer_value_fields=("amount",),
            ),
        ),
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="sales",
                answer_expression_family="list_rows",
                parts=("sales",),
            ),
            "submit_source_binding": source_binding_payload,
            "submit_source_alignment_reviews": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "pattern": "list_rows",
                            "source": {"kind": "read", "read_id": "sales"},
                            "output_fields": [{"field_id": "amount"}],
                        }
                    ],
                }
            },
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1"],
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": [{"field_id": "amount"}],
                        }
                    ],
                }
            },
        },
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=ports.relation_catalog_port,
        data_access_port=ports.data_access_port,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="How much sales did we make?",
            run_id="run_split_planning_turns",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert planner.tool_names == [
        "submit_question_contract_outcome",
        "submit_query_enrichment",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
        "submit_pattern_fact_plan",
    ]


def test_lookup_source_binding_uses_first_class_fulfillment_usage():
    ports = _ports(
        plan=FactPlan(
            outcome=_answer_plan(
                relations=(
                    Relation(
                        id="rows",
                        source=RelationSource(
                            kind=SourceKind.API_READ,
                            read_id="sales",
                        ),
                        fields=(
                            RelationField(
                                field_id="staff_name",
                                roles=(FieldBindingRole.OUTPUT,),
                            ),
                            RelationField(
                                field_id="amount",
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
                                ProjectField(source="staff_name"),
                                ProjectField(source="amount"),
                            ),
                        ),
                        output_relation="answer_rows",
                    ),
                ),
                result_projection=ResultProjection(
                    relation_outputs=(
                        RelationResultOutput(
                            id="staff_name",
                            relation_id="answer_rows",
                            field_id="staff_name",
                        ),
                        RelationResultOutput(
                            id="amount",
                            relation_id="answer_rows",
                            field_id="amount",
                        ),
                    )
                ),
            )
        ),
        catalog=_catalog(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sales",),
                row_paths=(
                    RowPath(
                        id="data",
                        path="data",
                        cardinality=RowCardinality.MANY,
                    ),
                ),
                fields=(
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
            )
        ),
        responses={
            "list_sale_list": {"data": [{"staff_name": "Amina", "amount": "1500"}]}
        },
        question_contract=_question_contract_for(
            "fact_1",
            description="staff and sales amount",
            subject_text="staff",
            binding_target_ids=("answer_1", "answer_2"),
        ),
    )

    def source_binding_payload(*, prompt, tool_specs):
        del tool_specs
        binding_target_id = source_binding_target_id_for_candidate(
            prompt,
            requested_fact_id="fact_1",
            source_candidate_id="source_1",
            plan_shape="list_rows",
        )
        return source_binding_payload_for_one_call(
            {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": {
                                "population_binding_id": "pop.source_1.candidate_population",
                                "intent_text": "staff and sales amount",
                                "match_basis_explanation": "staff and sales amount defines the source population",
                            },
                            "fulfillment_decisions": {
                                "answer_1": {
                                    "match_basis_explanation": "answer_1 is fulfilled by source_1.data.staff_name because that evidence provides the requested staff value.",
                                    "fulfillment_choice_id": "source_1.data.staff_name",
                                },
                                "answer_2": {
                                    "match_basis_explanation": "answer_2 is fulfilled by source_1.data.amount because that evidence provides the requested sales amount value.",
                                    "fulfillment_choice_id": "source_1.data.amount",
                                },
                            },
                            "param_decisions": {},
                        },
                    },
                }
            },
            prompt=prompt,
        )

    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                answer_value_fields=("staff_name",),
            ),
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                measured_value_fields=("amount",),
            ),
        ),
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="staff and sales amount",
                answer_subject="staff",
                parts=("staff", "sales amount"),
                answer_expression_family="list_rows",
            ),
            "submit_source_binding": source_binding_payload,
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1", "answer_2"],
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": [
                                {"field_id": "staff_name"},
                                {"field_id": "amount"},
                            ],
                        }
                    ],
                }
            },
        },
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=ports.relation_catalog_port,
        data_access_port=ports.data_access_port,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="List staff and sales amount.",
            run_id="run_fulfillment_usage_contract",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert "Amina" in result.answer
    assert "1500" in result.answer


def test_lookup_rejects_param_distinct_bindings_for_one_selected_plan_member():
    read_id = "sales_summary"
    planner = _ToolNamePlannerPort(
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="total sales revenue",
                answer_subject="sales",
                answer_expression_family="scalar_value",
                parts=("total sales revenue",),
            ),
            "submit_source_alignment_reviews": _first_source_strategy_selection_payload,
            "submit_source_binding": _duplicate_selected_candidate_binding_payload,
            "submit_pattern_fact_plan": _location_grouped_total_revenue_plan,
        },
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id=read_id,
                answer_value_fields=("total_revenue",),
            ),
        ),
    )
    data_access = _DataAccessPort(
        {
            read_id: {
                "data": [
                    {"total_revenue": "14.00"},
                ]
            }
        }
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_selected_plan_boundary_catalog(read_id)),
        data_access_port=data_access,
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What was total sales revenue by location?",
            run_id="run_reject_param_distinct_selected_member_bindings",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "FAILED", result
    assert data_access.requests == []


def _first_source_strategy_selection_payload(prompt: str, **_: Any) -> dict[str, Any]:
    return plan_selection_payload_from_fact_plan({}, prompt=prompt)


def _duplicate_selected_candidate_binding_payload(
    prompt: str, **_: Any
) -> dict[str, Any]:
    source_candidate_id, options = _source_candidate_param_options(
        prompt,
        param_id="group_by",
    )
    support_set_id = _first_fulfillment_choice_id(
        prompt,
        source_candidate_id=source_candidate_id,
    )
    date_decision = _param_decision_for_value(
        options,
        "date",
        population_intent="sales grouped by date",
    )
    location_decision = _param_decision_for_value(
        options,
        "location",
        population_intent="sales grouped by location",
    )
    invocation = {
        "requested_fact_id": "fact_1",
        "source_candidate_id": source_candidate_id,
        "answer_population": {
            "population_binding_id": f"pop.{source_candidate_id}.candidate_population",
            "intent_text": "total sales revenue",
            "match_basis_explanation": "The selected source returns total sales revenue.",
        },
        "fulfillment_decisions": {
            "answer_1": {
                "match_basis_explanation": (
                    "answer_1 is fulfilled by the selected total revenue field."
                ),
                "fulfillment_choice_id": support_set_id,
            }
        },
        "param_decisions": _param_decisions_for_prompt(
            {"group_by": date_decision},
            prompt=prompt,
        ),
    }
    duplicate = {
        **invocation,
        "param_decisions": _param_decisions_for_prompt(
            {"group_by": location_decision},
            prompt=prompt,
        ),
    }
    return {
        "outcome": {
            "kind": "source_bindings",
            "source_invocations": [invocation, duplicate],
        }
    }


def _source_candidate_param_options(
    prompt: str,
    *,
    param_id: str,
) -> tuple[str, dict[str, Any]]:
    for source_candidate_id, param_options in _source_candidate_param_decision_options(
        prompt
    ).items():
        if param_id in param_options:
            return source_candidate_id, param_options[param_id]
    raise AssertionError(f"source candidate param missing from prompt: {param_id}")


def _first_fulfillment_choice_id(
    prompt: str,
    *,
    source_candidate_id: str,
) -> str:
    support_sets = _source_candidate_fulfillment_support_sets(prompt)[
        source_candidate_id
    ]
    return str(support_sets[0]["fulfillment_choice_id"])


def _param_decision_for_value(
    options: dict[str, Any],
    value: str,
    *,
    population_intent: str,
) -> dict[str, str]:
    return {
        "population_intent": population_intent,
        "match_basis_explanation": f"{value} is the selected argument value.",
        "param_decision_id": options["bind_decision_ids"][value],
    }


def _location_grouped_total_revenue_plan(prompt: str, **_: Any) -> dict[str, Any]:
    source_binding_id = _bound_source_id_for_param_value(
        prompt,
        param_id="group_by",
        value="location",
    )
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "pattern": "direct_field_value",
                    "source_binding_id": source_binding_id,
                    "output_field": {"field_id": "total_revenue"},
                }
            ],
        }
    }


def _bound_source_id_for_param_value(
    prompt: str,
    *,
    param_id: str,
    value: str,
) -> str:
    bound_sources = _planner_prompt_json_section(
        prompt,
        label="Bound sources",
    )["bound_sources"]
    for bound_source in bound_sources:
        for bound_param in bound_source.get("bound_params") or ():
            if (
                str(bound_param.get("param_id") or "") == param_id
                and str(bound_param.get("value") or "") == value
            ):
                return str(bound_source["source_binding_id"])
    raise AssertionError(f"bound source missing for {param_id}={value}")


def _selected_plan_boundary_catalog(read_id: str) -> RelationCatalog:
    return _catalog(
        EndpointRead(
            id=read_id,
            endpoint_name=read_id,
            resource_names=("sales", "revenue"),
            params=(
                CatalogParam(
                    ref=f"{read_id}.query.group_by",
                    name="group_by",
                    source=ParamSource.QUERY,
                    type="choice",
                    required=True,
                    choices=("date", "location"),
                ),
            ),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref=f"{read_id}.field.total_revenue",
                    path="data.total_revenue",
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


def test_lookup_plan_selection_uses_backend_projected_candidates():
    class _PlanSelectionPlannerPort:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(
            self,
            *,
            provider: str,
            prompt: str,
            max_thinking_tokens: int,
            system_prompt: str = "",
            output_mode: Any = None,
            tool_specs: tuple[Any, ...] = (),
        ) -> dict[str, Any]:
            del provider, max_thinking_tokens, system_prompt, output_mode
            self.prompts.append(prompt)
            tool_name = tool_specs[0].name if tool_specs else ""
            if tool_name == "submit_question_contract_outcome":
                arguments = _question_contract_decision(
                    _question_contract_response(
                        subject="total sales",
                        answer_subject="sales",
                        answer_expression_family="scalar_value",
                        parts=("total sales",),
                        demand_text="sales",
                    )
                )
            elif tool_name == "submit_query_enrichment":
                arguments = _query_enrichment_payload_from_prompt(prompt)
            elif tool_name == "submit_read_eligibility":
                return read_eligibility_response_for_retained_fields(
                    prompt,
                    answer_value_fields=("total_sales",),
                )
            elif tool_name == "submit_source_binding":
                binding_target_id = source_binding_target_id_for_candidate(
                    prompt,
                    requested_fact_id="fact_1",
                    source_candidate_id="source_1",
                    plan_shape="direct_field_value",
                )
                source_binding_arguments = {
                    "outcome": {
                        "kind": "source_bindings",
                        "bindings_for_fact_1": {
                            "plan_shape": "direct_field_value",
                            "primary": {
                                "binding_target_id": binding_target_id,
                                "answer_population": {
                                    "population_binding_id": "pop.source_1.candidate_population",
                                    "intent_text": "total sales",
                                    "match_basis_explanation": "total sales defines the source population",
                                },
                                "fulfillment_decisions": {
                                    "answer_1": {
                                        "match_basis_explanation": "answer_1 is fulfilled by source_1.root.total_sales because that evidence is the requested scalar value.",
                                        "fulfillment_choice_id": "source_1.root.total_sales",
                                    }
                                },
                                "param_decisions": {},
                            },
                        },
                    }
                }
                arguments = source_binding_payload_for_one_call(
                    source_binding_arguments,
                    prompt=prompt,
                )
            elif tool_name == "submit_source_alignment_reviews":
                arguments = plan_selection_payload_from_fact_plan(
                    {
                        "outcome": {
                            "kind": "fact_plan",
                            "answers": [
                                {
                                    "requested_fact_id": "fact_1",
                                    "pattern": "direct_field_value",
                                    "source": {
                                        "kind": "read",
                                        "read_id": "sales_summary",
                                    },
                                    "output_field": {"field_id": "total_sales"},
                                }
                            ],
                        }
                    },
                    prompt=prompt,
                )
            elif tool_name == "submit_pattern_fact_plan":
                arguments = {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "fact_1",
                                "answer_output_ids": ["answer_1"],
                                "pattern": "direct_field_value",
                                "source_binding_id": "sb_1",
                                "output_field": {"field_id": "total_sales"},
                            }
                        ],
                    }
                }
            else:
                raise AssertionError(f"unexpected tool: {tool_name}")
            return {
                "answer": json.dumps({"tool": tool_name, "arguments": arguments}),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }

    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                EndpointRead(
                    id="sales_summary",
                    endpoint_name="list_sales_summary",
                    resource_names=("sales summary",),
                    row_paths=(
                        RowPath(
                            id="root",
                            path="",
                            cardinality=RowCardinality.ONE,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.total_sales",
                            path="total_sales",
                            row_path_id="root",
                            type="decimal",
                        ),
                    ),
                )
            )
        ),
        data_access_port=_DataAccessPort(
            {"list_sales_summary": {"total_sales": "9000"}}
        ),
        planner_model_port=_PlanSelectionPlannerPort(),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What are total sales?",
            run_id="run_plan_selection_candidates",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert "9000" in result.answer


def test_lookup_cutover_uses_relation_as_read_instance_and_derives_grain():
    data_access = _DataAccessPort(
        {
            "list_sale_list": {
                "data": [
                    {
                        "sale_id": "sale_1",
                        "location_id": "store_1",
                        "amount": "125.00",
                        "sold_at": "2026-05-07T10:00:00+03:00",
                    }
                ]
            },
            "list_store_list": {"data": [{"store_id": "store_1", "name": "ABC Mall"}]},
        }
    )
    result = run_lookup_question(
        LookupRequest(
            question="How much sales did ABC Mall make today?",
            run_id="run_relation_owned_read",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-07",
                timezone="Africa/London",
            ),
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_sales_and_store_catalog()),
            data_access_port=data_access,
            planner_model_port=_RawPlannerPort(
                _pattern_fact_plan_payload(
                    requested_fact_id="rf_amount",
                    answer_output_ids=("amount",),
                    read_id="list_sale_list",
                    output_fields=({"field_id": "amount"},),
                ),
                question_contract=_question_contract_for(
                    "rf_amount",
                    description="sales amount at store today",
                    subject_text="sales",
                    binding_target_ids=("amount",),
                    known_inputs=(
                        RequestedFactLiteralInput(
                            id="today",
                            source=KnownInputSource.QUESTION_CONTEXT,
                            text="today",
                            role=LiteralInputRole.TIME_VALUE,
                            resolved_value_text="today",
                        ),
                    ),
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "125.00"
    assert data_access.requests == [
        {
            "endpointName": "list_sale_list",
            "args": {
                "list_sale_list.query.start_date": "2026-05-07",
                "list_sale_list.query.end_date": "2026-05-07",
            },
        }
    ]


def test_lookup_model_turns_send_business_system_prompt_and_question_frame():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="sales_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="sales",
                    ),
                    fields=(
                        RelationField(
                            field_id="amount",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_amount",
                    spec=ProjectSpec(
                        input_relation="sales_rows",
                        fields=(ProjectField(source="amount"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="amount",
                        relation_id="answer_rows",
                        field_id="amount",
                    ),
                )
            ),
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_amount",
                    answer_output_id="amount",
                    result_output_id="amount",
                ),
            ),
        )
    )
    planner = _PlannerPort(
        plan,
        question_contract=_question_contract_for(
            "rf_amount",
            description="sales amount",
            subject_text="sales",
            binding_target_ids=("amount",),
        ),
    )
    result = run_lookup_question(
        LookupRequest(
            question="How much sales did we make?",
            run_id="run_prompt_frame",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
            host=HostPromptContext(
                organization_name="Shopify",
                about_api=(
                    "The Shopify API helps merchants work with online and "
                    "in-person commerce operations."
                ),
            ),
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(
                    EndpointRead(
                        id="sales",
                        endpoint_name="list_sale_list",
                        resource_names=("sales",),
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
                        ),
                        facts=(
                            CatalogFact(ref="sales.amount", field_ref="field.amount"),
                        ),
                    )
                )
            ),
            data_access_port=_DataAccessPort(
                {"list_sale_list": {"data": [{"amount": "50000"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert len(planner.system_prompts) == 6
    assert all(
        prompt.startswith("You are an Fervis runtime for Shopify.\n\nAbout the API:\n")
        for prompt in planner.system_prompts
    ), [
        (index, system_prompt[:120], planner.prompts[index][:120])
        for index, system_prompt in enumerate(planner.system_prompts)
    ]
    assert all(
        "The Shopify API helps merchants work with online and in-person commerce operations."
        in prompt
        for prompt in planner.system_prompts
    )
    question_contract_prompt = next(
        prompt
        for prompt in planner.prompts
        if "We are currently on the question contract step." in prompt
    )
    assert question_contract_prompt.startswith("Current question:\n")
    assert "Your task is to author" in question_contract_prompt
    assert "We are currently on the pattern fact planning step." in _fact_plan_prompt(
        planner
    )


def test_lookup_cutover_returns_rendered_machine_truth_without_model_synthesis():
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
        catalog=_catalog(
            EndpointRead(
                id="metric_read",
                endpoint_name="metric_read",
                resource_names=("metric read",),
                row_paths=(
                    RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
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
        ),
        responses={
            "metric_read": {
                "data": [{"location_name": "Location Alpha", "metric_total": "125.00"}]
            }
        },
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=ports.relation_catalog_port,
        data_access_port=ports.data_access_port,
        planner_model_port=ports.planner_model_port,
    )

    result = run_lookup_question(
        LookupRequest(question="What was the metric total?", run_id="run_truth"),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "Location Alpha: 125.00"


def test_lookup_cutover_no_data_is_tied_to_fulfilled_requested_fact():
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_open_ticket",
                description="open ticket",
                answer_subject=RequestedFactAnswerSubject(subject_text="tickets"),
                required_for="open ticket",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="ticket_id", role="ANSWER_VALUE"),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_open_ticket",
                    answer_output_id="ticket_id",
                    result_output_id="ticket_id",
                ),
            ),
            relations=(
                Relation(
                    id="ticket_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="tickets",
                    ),
                    fields=(
                        RelationField(
                            field_id="ticket_id",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="ticket_rows",
                        fields=(ProjectField(source="ticket_id"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="ticket_id",
                        relation_id="answer_rows",
                        field_id="ticket_id",
                    ),
                )
            ),
        )
    )
    ports = _ports(
        plan=plan,
        catalog=RelationCatalog(
            reads=(
                EndpointRead(
                    id="tickets",
                    endpoint_name="list_tickets",
                    resource_names=("tickets",),
                    row_paths=(
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    ),
                    fields=(
                        CatalogField(
                            ref="field.ticket_id",
                            path="data.ticket_id",
                            row_path_id="data",
                            type="string",
                        ),
                    ),
                    params=(
                        CatalogParam(
                            ref="query.priority",
                            name="priority",
                            source=ParamSource.QUERY,
                            type="string",
                            choices=("P1",),
                        ),
                    ),
                    facts=(
                        CatalogFact(
                            ref="ticket.id",
                            availability=CatalogFactAvailability.AVAILABLE,
                            field_ref="field.ticket_id",
                        ),
                    ),
                    pagination=PaginationMetadata(
                        mode=PaginationMode.NONE,
                        completeness_policy=CompletenessPolicy.COMPLETE,
                    ),
                ),
            )
        ),
        responses={"list_tickets": {"data": []}},
        question_contract=question_contract,
    )

    result = run_lookup_question(
        LookupRequest(
            question="Which open P1 tickets are there?",
            run_id="run_no_data_requested_fact",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert result.fact_result.outcome.kind == OutcomeKind.NO_DATA
    details = result.rendered_fact.details  # type: ignore[union-attr]
    empty = details["emptyRelation"]  # type: ignore[index]
    assert empty["requestedFactIds"] == ["fact_1"]
    assert "read:list_tickets" in empty["proofRefs"]


def _required_identity_detail_read(
    read_id: str,
    endpoint_name: str,
    entity_ref: str,
) -> EndpointRead:
    return EndpointRead(
        id=read_id,
        endpoint_name=endpoint_name,
        method="GET",
        resource_names=(entity_ref,),
        params=(
            CatalogParam(
                ref=f"{endpoint_name}.path.{entity_ref}_id",
                name=f"{entity_ref}_id",
                source=ParamSource.PATH,
                type="string",
                required=True,
                entity_target=EntityKeyComponentTarget(
                    entity_kind=entity_ref,
                    key_id="primary_key",
                    component_id=f"{entity_ref}_id",
                ),
            ),
        ),
        row_paths=(RowPath(id="root", path="", cardinality=RowCardinality.ONE),),
        fields=(
            CatalogField(
                ref=f"{endpoint_name}.field.{entity_ref}_id",
                path=f"{entity_ref}_id",
                row_path_id="root",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="primary_key",
                entity_kind=entity_ref,
                components=(
                    CandidateKeyComponent(
                        id=f"{entity_ref}_id",
                        field_ref=f"{endpoint_name}.field.{entity_ref}_id",
                    ),
                ),
                primary=True,
            ),
        ),
    )


def _staff_identity_list_read() -> EndpointRead:
    return EndpointRead(
        id="staff_list",
        endpoint_name="list_staff_list",
        method="GET",
        resource_names=("staff",),
        catalog_endpoint=_catalog_endpoint_metadata(
            endpoint_name="list_staff_list",
            namespace_path=("tests",),
            route_path_template="/v1/staff/",
            handler_ref="tests.StaffListView",
            domain_resource_names=("staff",),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="staff.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="staff.field.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="staff_key",
                entity_kind="staff",
                components=(
                    CandidateKeyComponent(
                        id="staff_id",
                        field_ref="staff.field.staff_id",
                    ),
                ),
                primary=True,
                context_field_refs=("staff.field.full_name",),
            ),
        ),
    )


def _staff_identity_list_read_without_catalog_endpoint() -> EndpointRead:
    return replace(_staff_identity_list_read(), catalog_endpoint=None)


def _identity_list_read(
    read_id: str,
    endpoint_name: str,
    entity_ref: str,
    *,
    extra_identity_refs: tuple[str, ...] = (),
) -> EndpointRead:
    identity_refs = (entity_ref, *extra_identity_refs)
    return EndpointRead(
        id=read_id,
        endpoint_name=endpoint_name,
        method="GET",
        resource_names=(entity_ref,),
        catalog_endpoint=_catalog_endpoint_metadata(
            endpoint_name=endpoint_name,
            namespace_path=("tests",),
            route_path_template=f"/v1/{endpoint_name}/",
            handler_ref=f"tests.{endpoint_name}.View",
            domain_resource_names=(entity_ref,),
        ),
        params=(
            CatalogParam(
                ref=f"{endpoint_name}.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            *(
                CatalogField(
                    ref=f"{endpoint_name}.field.{identity_ref}_id",
                    path=f"data.{identity_ref}_id",
                    row_path_id="data",
                    type="string",
                )
                for identity_ref in identity_refs
            ),
            CatalogField(
                ref=f"{endpoint_name}.field.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="primary_key",
                entity_kind=entity_ref,
                components=(
                    CandidateKeyComponent(
                        id=f"{entity_ref}_id",
                        field_ref=f"{endpoint_name}.field.{entity_ref}_id",
                    ),
                ),
                primary=True,
                context_field_refs=(f"{endpoint_name}.field.name",),
            ),
        ),
        entity_references=tuple(
            EntityReference(
                id=f"{identity_ref}_reference",
                target_entity_kind=identity_ref,
                target_key_id="primary_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id=f"{identity_ref}_id",
                        local_field_ref=f"{endpoint_name}.field.{identity_ref}_id",
                    ),
                ),
                context_field_refs=(f"{endpoint_name}.field.name",),
            )
            for identity_ref in extra_identity_refs
        ),
    )


@dataclass
class _LimitFailure:
    status: str
    error: str
    result_data: dict[str, Any]
    usage: dict[str, Any]
