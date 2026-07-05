from __future__ import annotations

import pytest

from fervis.lookup.question_contract.schema import (
    build_answer_request_contract_schema,
)
from fervis.lookup.question_contract.parser import parse_question_contract
from fervis.lookup.question_contract.model import (
    validate_question_contract_against_question,
)
from fervis.lookup.question_contract.tools import (
    ANSWER_REQUEST_CONTRACT_TOOL_NAME,
)
from tests.lookup.orchestrator._helpers import *  # noqa: F403


def _question_contract_payload(
    *,
    subject: str = "sales",
    answer_subject: str | None = None,
    parts: tuple[str, ...] = ("sales",),
) -> dict[str, object]:
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [],
        "answer_requests": [
            {
                "answer_fact": subject,
                "answer_expression": {"family": "scalar_aggregate"},
                "answer_subject": _answer_subject_payload(answer_subject or subject),
                "answer_population": default_answer_population(
                    description=subject,
                    subject_text=answer_subject or subject,
                    instance_interpretation=RequestedFactAnswerSubject(
                        subject_text=answer_subject or subject
                    ).instance_interpretation,
                ).to_question_contract_dict(),
                "answer_outputs": [{"description": part} for part in parts],
                "used_question_inputs": [],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }


def test_question_contract_schema_declares_inputs_before_answer_requests():
    schema = build_answer_request_contract_schema()

    assert "oneOf" not in schema
    assert list(schema["properties"]).index(  # type: ignore[index]
        "question_inputs"
    ) < list(
        schema["properties"]  # type: ignore[index]
    ).index("answer_requests")


def test_question_contract_accepts_semantic_answer_subject_not_copied_from_question():
    question = "Where did Jane Doe work on her first two shifts?"
    payload = _question_contract_payload(
        subject="work location for Jane Doe's first two shifts",
        answer_subject="staff shift",
        parts=("the location where she worked",),
    )
    request_payload = payload["answer_requests"][0]
    assert isinstance(request_payload, dict)

    result = parse_question_contract(
        tool_name=ANSWER_REQUEST_CONTRACT_TOOL_NAME,
        payload=payload,
        question_context=question,
        question_context_texts=(question,),
    )
    validate_question_contract_against_question(
        result.outcome,
        question=question,
        context_texts=(question,),
    )

    assert (
        result.outcome.requested_facts[0].answer_subject.subject_text == "staff shift"
    )


def test_lookup_question_contract_cannot_short_circuit_into_clarification():
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
                            field_id="store_name",
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
                        fields=(ProjectField(source="store_name", output="store"),),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="store",
                        relation_id="answer_rows",
                        field_id="store",
                    ),
                )
            ),
        )
    )
    planner = _ClarificationBiasedPlannerPort(plan=plan)
    result = run_lookup_question(
        LookupRequest(
            question="Which store brought in highest sales?",
            run_id="run_no_catalog_blind_clarification",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(
                    EndpointRead(
                        id="metric_read",
                        endpoint_name="metric_read",
                        resource_names=("sales", "store"),
                        row_paths=(
                            RowPath(
                                id="data",
                                path="data",
                                cardinality=RowCardinality.MANY,
                            ),
                        ),
                        fields=(
                            CatalogField(
                                ref="field.data.store_name",
                                path="data.store_name",
                                row_path_id="data",
                                type="string",
                            ),
                        ),
                    )
                )
            ),
            data_access_port=_DataAccessPort(
                {"metric_read": {"data": [{"store_name": "ABC Mall"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "ABC Mall"
    assert planner.tool_names == [
        "submit_answer_request_contract",
        "submit_query_enrichment",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
        "submit_pattern_fact_plan",
    ]


def test_lookup_carries_answer_subject_instance_interpretation_to_source_binding():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": {
                "kind": "question_contract",
                "answer_requests_count": 1,
                "question_inputs": [],
                "answer_requests": [
                    {
                        "answer_fact": "in-person sales this month",
                        "answer_expression": {"family": "scalar_aggregate"},
                        "answer_subject": {
                            "subject_text": "sales",
                            "instance_interpretation": {
                                "kind": "NORMAL_BUSINESS_INSTANCE"
                            },
                        },
                        "answer_outputs": [{"description": "amount"}],
                        "used_question_inputs": [],
                    }
                ],
                "question_input_inventory_check": {
                    "all_input_like_phrases_declared": True,
                },
            },
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="sales",
                output_fields=({"field_id": "amount"},),
            ),
        }
    )
    result = run_lookup_question(
        LookupRequest(
            question="How much were in-person sales this month?",
            run_id="run_subject_instance_interpretation",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
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
                    )
                )
            ),
            data_access_port=_DataAccessPort(
                {"list_sale_list": {"data": [{"amount": "100.00"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    source_binding_prompt = _source_binding_prompt(planner)
    assert '"kind": "NORMAL_BUSINESS_INSTANCE"' in source_binding_prompt
    assert (
        "Answer over ordinary business instances of 'sales' as they are normally "
        "understood in business operations and reporting."
    ) in source_binding_prompt


@dataclass
class _ExactQuestionContractPlannerPort:
    payload: dict[str, object]
    tool_names: list[str] = field(default_factory=list)

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: object = None,
        tool_specs: tuple[object, ...] = (),
    ) -> dict[str, object]:
        del provider, prompt, max_thinking_tokens, system_prompt, output_mode
        desired_tool_name = (
            "submit_missing_input_clarification"
            if self.payload.get("kind") == "needs_clarification"
            else "submit_answer_request_contract"
        )
        offered = {tool.name for tool in tool_specs}
        tool_name = desired_tool_name if desired_tool_name in offered else ""
        self.tool_names.append(tool_name)
        if tool_name != desired_tool_name:
            raise AssertionError(f"unexpected tool: {tool_name}")
        return {
            "answer": json.dumps(
                {
                    "tool": tool_name,
                    "arguments": self.payload,
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


def test_lookup_question_contract_clarification_stops_after_question_contract_turn():
    planner = _ExactQuestionContractPlannerPort(
        payload={
            "kind": "needs_clarification",
            "missing": [
                {
                    "type": "answer_definition",
                    "source_text": "check",
                    "entity_type": "",
                    "why_context_is_insufficient": (
                        "The question does not say what business fact or metric "
                        "should be checked."
                    ),
                }
            ],
            "clarification_question": "What should I check?",
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="Hey, can you check?",
            run_id="run_question_contract_missing_answer_definition",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status == "NEEDS_CLARIFICATION", result
    assert result.answer == "What should I check?"
    assert planner.tool_names == ["submit_missing_input_clarification"]
    details = result.rendered_fact.details  # type: ignore[union-attr]
    clarification = details["clarifications"][0]  # type: ignore[index]
    assert clarification["basis"] == "missing_answer_metric"
    assert clarification["question"] == "What should I check?"
    assert clarification["ambiguousMetricPhrase"] == "check"
    assert "requested_fact:question_contract" in clarification["evidenceRefs"]


def test_lookup_question_contract_can_terminally_request_missing_target_reference():
    planner = _ExactQuestionContractPlannerPort(
        payload={
            "kind": "needs_clarification",
            "missing": [
                {
                    "type": "target_reference",
                    "source_text": "her",
                    "entity_type": "staff",
                    "why_context_is_insufficient": (
                        "No resolved staff reference is present in conversation "
                        "resolution or active clarification context."
                    ),
                }
            ],
            "clarification_question": "Which staff member do you mean?",
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="What were her sales last week?",
            run_id="run_question_contract_missing_target_reference",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status == "NEEDS_CLARIFICATION", result
    assert result.answer == "Which staff member do you mean?"
    assert planner.tool_names == ["submit_missing_input_clarification"]
    details = result.rendered_fact.details  # type: ignore[union-attr]
    clarification = details["clarifications"][0]  # type: ignore[index]
    assert clarification["basis"] == "unsupported_reference"
    assert clarification["knownInputId"] == "question_contract:her"
    assert "known_input:question_contract:her" in clarification["evidenceRefs"]


def test_lookup_question_contract_rejects_false_inventory_check():
    payload = _question_contract_payload()
    answer_request = payload["answer_requests"][0]
    assert isinstance(answer_request, dict)
    payload["question_input_inventory_check"] = {
        "all_input_like_phrases_declared": False,
    }
    with pytest.raises(
        ValueError, match="all_input_like_phrases_declared must be true"
    ):
        parse_question_contract(
            tool_name=ANSWER_REQUEST_CONTRACT_TOOL_NAME,
            payload=payload,
            question_context="How many sales happened this month?",
        )

    planner = _ExactQuestionContractPlannerPort(payload=payload)

    result = run_lookup_question(
        LookupRequest(
            question="How many sales happened this month?",
            run_id="run_question_contract_false_inventory_check",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED", result
    assert result.error == "planning_failed"


def test_lookup_clarification_answer_prompt_frame_uses_memory_cards_not_global_prior_section():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(),
        }
    )
    prior_artifact = build_fact_artifact(
        artifact_id="prior_clarification",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal=FactOutcome.NEEDS_CLARIFICATION.value,
                clarification_questions=("Which store do you mean?",),
                proof={"kind": "needs_clarification"},
            ),
        ),
        source_question="The store I referred to, how much sales did we make yesterday?",
    )

    result = run_lookup_question(
        LookupRequest(
            question="ABC Mall",
            conversation_context={"factArtifacts": [prior_artifact.to_dict()]},
            run_id="run_active_clarification_prompt_frame",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status in {"FAILED", "NEEDS_CLARIFICATION", "COMPLETED"}
    prompt = planner.prompts[0]
    assert prompt.startswith("Current question:\nABC Mall")
    assert "Active clarification prior question:" not in prompt
    assert "The store I referred to, how much sales did we make yesterday?" in prompt
    assert "ABC Mall" in prompt
    assert "We are currently on the conversation resolution step." in prompt
    assert "Context sources:" in prompt


def test_lookup_multi_step_clarification_prompt_frame_uses_memory_cards_not_global_prior_section():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(),
        }
    )
    first_clarification = build_fact_artifact(
        artifact_id="clarification_store",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal=FactOutcome.NEEDS_CLARIFICATION.value,
                clarification_questions=("Which store do you mean?",),
                proof={"kind": "needs_clarification"},
            ),
        ),
        source_question="How much sales did we make?",
    )
    second_clarification = build_fact_artifact(
        artifact_id="clarification_date",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal=FactOutcome.NEEDS_CLARIFICATION.value,
                clarification_questions=("Which date do you mean?",),
                proof={"kind": "needs_clarification"},
            ),
        ),
        source_question="ABC Mall",
    )

    result = run_lookup_question(
        LookupRequest(
            question="yesterday",
            conversation_context={
                "factArtifacts": [
                    second_clarification.to_dict(),
                    first_clarification.to_dict(),
                ]
            },
            run_id="run_multi_step_active_clarification_prompt_frame",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status in {"FAILED", "NEEDS_CLARIFICATION", "COMPLETED"}
    prompt = planner.prompts[0]
    assert prompt.startswith("Current question:\nyesterday")
    assert "Active clarification prior question:" not in prompt
    assert "How much sales did we make?" in prompt
    assert "Context sources:" in prompt


def test_lookup_active_clarification_accepts_known_inputs_from_prior_question():
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_sales",
                description="sales amount for ABC Mall yesterday",
                answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="amount",
                    ),
                ),
                known_inputs=(
                    _known_reference_input(
                        "store",
                        "ABC Mall",
                        value_meaning_hint="location",
                    ),
                    _known_time_input("date", "yesterday"),
                ),
            ),
        )
    )
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_sales",
                    answer_output_id="amount",
                    render_output_id="amount",
                ),
            ),
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
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="amount",
                        relation_id="answer_rows",
                        field_id="amount",
                    ),
                )
            ),
        )
    )
    prior_artifact = build_fact_artifact(
        artifact_id="prior_clarification",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal=FactOutcome.NEEDS_CLARIFICATION.value,
                clarification_questions=("Which store do you mean?",),
                proof={"kind": "needs_clarification"},
            ),
        ),
        source_question="The store I referred to, how much sales did we make yesterday?",
    )

    def clarification_resolution(prompt: str) -> dict[str, object]:
        source = _first_context_source(prompt)
        return _conversation_resolution_clause_payload(
            prompt=prompt,
            current_question="ABC Mall",
            integrated_question="How much sales did we make at ABC Mall yesterday?",
            actual_text="ABC Mall",
            selected_sources=(source,),
            dependencies=[],
            requested_value_frame={
                "current_value_surface": {
                    "text": "ABC Mall",
                    "kind": "no_value_request",
                },
                "context_frame_choices": [],
            },
        )

    ports = _ports(
        plan=plan,
        catalog=_catalog(
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
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="string",
                        identity=IdentityMetadata(
                            entity_ref="location",
                            identity_field="location_id",
                            primary_key=True,
                            stable=True,
                            display_fields=("field.location_name",),
                        ),
                    ),
                    CatalogField(
                        ref="field.location_name",
                        path="data.name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sales",),
                params=(
                    CatalogParam(
                        ref="sales.query.location_id",
                        name="location_id",
                        source=ParamSource.QUERY,
                        type="string",
                        identity=IdentityMetadata(
                            entity_ref="location",
                            identity_field="location_id",
                            primary_key=True,
                        ),
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
                facts=(CatalogFact(ref="sales.amount", field_ref="field.amount"),),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
        responses={
            "list_location_list": {
                "data": [{"location_id": "loc_bbs", "name": "ABC Mall"}]
            },
            "list_sale_list": {"data": [{"amount": "50000"}]},
        },
        question_contract=question_contract,
        conversation_resolution=clarification_resolution,
    )

    result = run_lookup_question(
        LookupRequest(
            question="ABC Mall",
            conversation_context={"factArtifacts": [prior_artifact.to_dict()]},
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-09",
                timezone="Africa/London",
            ),
            run_id="run_active_clarification_prior_input",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "50000"


def test_lookup_follow_up_memory_stays_out_of_question_contract_prompt():
    artifact = build_fact_artifact(
        artifact_id="run_prior_total",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.value(
                address="value.sales_total",
                value={"type": "decimal", "value": "125.00"},
                display="ABC Mall sales yesterday",
                derivation={"source": "prior_result"},
            ),
        ),
        source_question="How much sales did ABC Mall make yesterday?",
    )
    plan = FactPlan(
        outcome=_answer_plan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="metric_total",
                    render_output_id="metric_total",
                ),
            ),
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
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="metric_total",
                        relation_id="answer_rows",
                        field_id="metric_total",
                    ),
                )
            ),
        )
    )
    planner = _QuestionIntentAwarePlannerPort(plan=plan)

    result = run_lookup_question(
        LookupRequest(
            question="How much is that in total?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
            run_id="run_question_intent_memory",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort(
                {
                    "metric_read": {
                        "data": [
                            {
                                "location_id": "location_alpha",
                                "metric_total": "125.00",
                            }
                        ]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.prompts)
    resolution_prompt = planner.prompts[0]
    question_prompt = planner.prompts[1]
    assert "Context sources:" in resolution_prompt
    assert '"source_id":' in resolution_prompt
    assert "run_prior_total.value.sales_total" not in resolution_prompt
    assert "Available prior answer candidates:" not in resolution_prompt
    assert "prior_reference_candidates" not in resolution_prompt
    assert "proofRefs" not in resolution_prompt
    assert "Conversation resolution annotations:" in question_prompt
    assert '"value_frames":' in question_prompt
    assert '"memory_id": "run_prior_total.value.sales_total"' not in question_prompt
    assert "proofRefs" not in question_prompt


def test_lookup_resolved_follow_up_reaches_query_enrichment_as_raw_question_with_overlay():
    prior_artifact = {
        "artifactId": "run_prior_total",
        "outcome": "answered",
        "sourceQuestion": "How much money did we make yesterday?",
        "provenance": {
            "question_contract": {
                "kind": "question_contract",
                "answer_requests_count": 1,
                "question_inputs": [],
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "money made yesterday",
                        "answer_expression": {"family": "scalar_aggregate"},
                        "answer_subject": _answer_subject_payload("the day before"),
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "total sales amount",
                                "requested_value_frame": "total sales amount",
                            }
                        ],
                        "used_question_inputs": [],
                    }
                ],
                "question_input_inventory_check": {
                    "all_input_like_phrases_declared": True,
                },
            }
        },
        "addresses": [
            {
                "address": "relation.answer_1_rows",
                "kind": "relation",
                "source": {
                    "kind": "operation_output",
                    "relationId": "answer_1_rows",
                },
                "fieldCoverage": {"amount": "answer_1_rows.amount"},
                "completeness": {
                    "status": "complete",
                    "pagination": "not_paginated",
                    "rowCount": 1,
                },
            },
            {
                "address": "row.answer_1_rows.1",
                "kind": "row",
                "relation": "relation.answer_1_rows",
                "values": {"amount": {"type": "decimal", "value": "13579.00"}},
            },
        ],
    }
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: (
                lambda prompt: _conversation_resolution_payload_using_memory(
                    prompt,
                    integrated_question="How much money did we make the day before yesterday?",
                    actual_text="the day before",
                )
            ),
            "submit_answer_request_contract": {
                "kind": "question_contract",
                "answer_requests_count": 1,
                "question_inputs": [
                    {
                        "input_ref": "input_period",
                        "source": "question_context",
                        "kind": "literal_text",
                        "role": "time_value",
                        "source_text": "the day before",
                        "resolved_value_text": "the day before",
                        "inventory_check": {
                            "why_this_is_an_input": ("the day before is a time scope")
                        },
                    }
                ],
                "answer_requests": [
                    {
                        "answer_fact": "total sales amount for the day before",
                        "answer_expression": {"family": "scalar_aggregate"},
                        "answer_subject": _answer_subject_payload("the day before"),
                        "answer_population": {
                            "population_label": "sales for the day before",
                            "counted_unit": "sale",
                            "membership_tests": [
                                {
                                    "test_id": "subject_identity",
                                    "kind": "SUBJECT_IDENTITY",
                                    "polarity": "MUST_PASS",
                                    "test_question": (
                                        "Does the row/value represent a sale?"
                                    ),
                                },
                                {
                                    "test_id": "normal_instance",
                                    "kind": "NORMAL_INSTANCE_GUARD",
                                    "polarity": "MUST_PASS",
                                    "test_question": (
                                        "Is this an ordinary domain instance of a sale?"
                                    ),
                                },
                            ],
                        },
                        "answer_outputs": [
                            {
                                "description": "total sales amount",
                            }
                        ],
                        "used_question_inputs": ["input_period"],
                    }
                ],
                "question_input_inventory_check": {
                    "all_input_like_phrases_declared": True,
                },
            },
            "submit_query_enrichment": _query_enrichment_payload(("sales",)),
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="How about the day before?",
            conversation_context={"factArtifacts": [prior_artifact]},
            run_id="run_resolved_follow_up",
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-15",
                timezone="Africa/London",
            ),
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert result.status in {"FAILED", "NEEDS_CLARIFICATION", "COMPLETED"}
    query_prompt = (
        planner.prompts[2] if len(planner.prompts) > 2 else planner.prompts[-1]
    )
    assert "Current question:\nHow about the day before?" in query_prompt
    if len(planner.prompts) > 2:
        assert "Conversation resolution annotations:" in query_prompt
    assert "the requested business fact for that date" not in query_prompt


def test_lookup_question_contract_is_catalog_blind_before_impossible_planning():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(
                subject="full card numbers used in buyer payments",
                parts=("full card numbers",),
            ),
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "impossible",
                    "blocked_facts": [
                        {
                            "requested_fact_id": "fact_1",
                            "basis": "policy_access",
                            "evidence_refs": [
                                "endpoint_docs:card_numbers_not_returned"
                            ],
                            "reviewed_read_ids": ["payments"],
                            "nearest_fields": [
                                {
                                    "read_id": "payments",
                                    "field_id": "card_txn_code_last_four",
                                }
                            ],
                            "explanation": "The payment read exposes only last-four card evidence, not full card numbers.",
                        }
                    ],
                }
            },
        },
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="payments",
            ),
        ),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            RelationCatalog(
                reads=(
                    EndpointRead(
                        id="payments",
                        endpoint_name="list_buyer_payments",
                        resource_names=("payments",),
                        row_paths=(
                            RowPath(
                                id="data",
                                path="data",
                                cardinality=RowCardinality.MANY,
                            ),
                        ),
                        fields=(
                            CatalogField(
                                ref="field.card_txn_code_last_four",
                                path="data.card_txn_code_last_four",
                                row_path_id="data",
                                type="string",
                            ),
                        ),
                        facts=(
                            CatalogFact(
                                ref="payment.card.full_number",
                                availability=CatalogFactAvailability.POLICY_BLOCKED,
                                field_ref="field.card_txn_code_last_four",
                                read_id="payments",
                                proof_refs=("endpoint_docs:card_numbers_not_returned",),
                            ),
                        ),
                        source_metadata={
                            "description": "Returns payment records. Full card numbers are never returned."
                        },
                    ),
                )
            )
        ),
        data_access_port=_DataAccessPort({}),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What are the full card numbers used in buyer payments?",
            run_id="run_card_impossible",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert planner.tool_names == [
        "submit_answer_request_contract",
        "submit_query_enrichment",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
        "submit_pattern_fact_plan",
    ]
    question_contract_prompt = planner.prompts[0]
    assert "card_txn_code_last_four" not in question_contract_prompt
    assert "Full card numbers are never returned" not in question_contract_prompt
    assert result.status == "COMPLETED"
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    blocked = result.rendered_fact.details["blockedRequirements"][0]  # type: ignore[union-attr,index]
    assert blocked["kind"] == "policy"
    assert blocked["requiredFor"] == "full card numbers used in buyer payments"
    assert blocked["nearestFields"] == [
        {
            "readId": "payments",
            "fieldId": "card_txn_code_last_four",
        }
    ]
    assert blocked["reviewedReadIds"] == ["payments"]
    assert set(blocked["proofRefs"]) == {
        "endpoint_docs:card_numbers_not_returned",
        "payments.card_txn_code_last_four",
        "payments",
    }
    assert ports.data_access_port.requests == []


def test_lookup_source_binding_impossible_accepts_source_evidence_handles():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(
                subject="full card numbers used in buyer payments",
                parts=("full card numbers",),
            ),
            "submit_source_binding": {
                "outcome": {
                    "kind": "impossible",
                    "blocked_facts": [
                        {
                            "requested_fact_id": "fact_1",
                            "basis": "policy_access",
                            "evidence_refs": [
                                "endpoint_docs:card_numbers_not_returned"
                            ],
                            "reviewed_read_ids": ["payments"],
                            "nearest_fields": [
                                {
                                    "read_id": "payments",
                                    "field_id": "card_txn_code_last_four",
                                }
                            ],
                            "explanation": "The payment read exposes only last-four card evidence, not full card numbers.",
                        }
                    ],
                }
            },
        },
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="payments",
            ),
        ),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            RelationCatalog(
                reads=(
                    EndpointRead(
                        id="payments",
                        endpoint_name="list_buyer_payments",
                        resource_names=("payments",),
                        row_paths=(
                            RowPath(
                                id="data",
                                path="data",
                                cardinality=RowCardinality.MANY,
                            ),
                        ),
                        fields=(
                            CatalogField(
                                ref="field.card_txn_code_last_four",
                                path="data.card_txn_code_last_four",
                                row_path_id="data",
                                type="string",
                            ),
                        ),
                        facts=(
                            CatalogFact(
                                ref="payment.card.full_number",
                                availability=CatalogFactAvailability.POLICY_BLOCKED,
                                field_ref="field.card_txn_code_last_four",
                                read_id="payments",
                                proof_refs=("endpoint_docs:card_numbers_not_returned",),
                            ),
                        ),
                        source_metadata={
                            "description": "Returns payment records. Full card numbers are never returned."
                        },
                    ),
                )
            )
        ),
        data_access_port=_DataAccessPort({}),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What are the full card numbers used in buyer payments?",
            run_id="run_card_impossible_source_evidence",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert planner.tool_names == [
        "submit_answer_request_contract",
        "submit_query_enrichment",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
    ]
    assert result.status == "COMPLETED", result
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    blocked = result.rendered_fact.details["blockedRequirements"][0]  # type: ignore[union-attr,index]
    assert blocked["kind"] == "policy"
    assert blocked["nearestFields"] == [
        {
            "readId": "payments",
            "fieldId": "card_txn_code_last_four",
        }
    ]
    assert blocked["reviewedReadIds"] == ["payments"]
    assert set(blocked["proofRefs"]) == {
        "endpoint_docs:card_numbers_not_returned",
        "payments.card_txn_code_last_four",
        "payments",
    }
    assert ports.data_access_port.requests == []


def test_lookup_source_binding_impossible_proves_selected_candidate_surface():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(
                subject="product sales report target values",
                answer_subject="product sales reports",
                parts=("target values",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("product", "sales")),
            "submit_source_binding": {
                "outcome": {
                    "kind": "impossible",
                    "blocked_facts": [
                        {
                            "requested_fact_id": "fact_1",
                            "basis": "policy_access",
                            "evidence_refs": [
                                "policy:product_target_values_1",
                                "policy:product_target_values_2",
                                "policy:product_target_values_3",
                            ],
                            "reviewed_read_ids": [
                                "product_sales_1",
                                "product_sales_2",
                                "product_sales_3",
                            ],
                            "nearest_fields": [
                                {
                                    "read_id": f"product_sales_{index}",
                                    "field_id": "product_name",
                                }
                                for index in range(1, 4)
                            ],
                            "explanation": "The selected product sales reads expose sales values, not product target values.",
                        }
                    ],
                }
            },
        },
        read_eligibility_retention_specs=tuple(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id=f"product_sales_{index}",
                row_path_ids=("data",),
            )
            for index in range(1, 4)
        ),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            RelationCatalog(
                reads=tuple(
                    _targetless_product_sales_read(index) for index in range(1, 7)
                )
            )
        ),
        data_access_port=_DataAccessPort({}),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question=(
                "Do our product sales reports expose target values so we can tell "
                "whether a product beat target?"
            ),
            run_id="run_product_target_impossible_selected_surface",
            tenant_id="tenant_1",
            max_catalog_reads_per_fact=3,
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert planner.tool_names == [
        "submit_answer_request_contract",
        "submit_query_enrichment",
        "submit_read_eligibility",
        "submit_source_alignment_reviews",
        "submit_source_binding",
    ]
    assert result.status == "COMPLETED", result
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    blocked = result.rendered_fact.details["blockedRequirements"][0]  # type: ignore[union-attr,index]
    assert blocked["kind"] == "policy"
    assert blocked["reviewedReadIds"] == [
        "product_sales_1",
        "product_sales_2",
        "product_sales_3",
    ]
    assert set(blocked["proofRefs"]) == {
        "product_sales_1.product_name",
        "policy:product_target_values_1",
        "product_sales_2.product_name",
        "policy:product_target_values_2",
        "product_sales_3.product_name",
        "policy:product_target_values_3",
        "product_sales_1",
        "product_sales_2",
        "product_sales_3",
    }
    assert ports.data_access_port.requests == []


def test_lookup_plan_shape_impossible_accepts_bound_source_evidence_handles():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_answer_request_contract": _question_contract_payload(
                subject="full card numbers used in buyer payments",
                parts=("full card numbers",),
            ),
            "submit_pattern_fact_plan": {
                "outcome": {
                    "kind": "impossible",
                    "blocked_facts": [
                        {
                            "requested_fact_id": "fact_1",
                            "basis": "policy_access",
                            "evidence_refs": [
                                "endpoint_docs:full_card_numbers_hidden",
                            ],
                            "reviewed_read_ids": ["payments"],
                            "nearest_fields": [
                                {
                                    "read_id": "payments",
                                    "field_id": "card_txn_code",
                                }
                            ],
                            "explanation": "The bound source exposes only card transaction codes, not full card numbers.",
                        }
                    ],
                }
            },
        },
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="payments",
            ),
        ),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            RelationCatalog(
                reads=(
                    EndpointRead(
                        id="payments",
                        endpoint_name="list_buyer_payments",
                        resource_names=("payments",),
                        row_paths=(
                            RowPath(
                                id="data",
                                path="data",
                                cardinality=RowCardinality.MANY,
                            ),
                        ),
                        fields=(
                            CatalogField(
                                ref="field.card_txn_code",
                                path="data.card_txn_code",
                                row_path_id="data",
                                type="string",
                            ),
                        ),
                        facts=(
                            CatalogFact(
                                ref="payment.card.full_number",
                                availability=CatalogFactAvailability.POLICY_BLOCKED,
                                field_ref="field.card_txn_code",
                                read_id="payments",
                                proof_refs=("endpoint_docs:full_card_numbers_hidden",),
                            ),
                        ),
                        source_metadata={
                            "description": "Returns buyer payments. Full card numbers are never returned."
                        },
                    ),
                )
            )
        ),
        data_access_port=_DataAccessPort({}),
        planner_model_port=planner,
    )

    result = run_lookup_question(
        LookupRequest(
            question="What are the full card numbers used in buyer payments?",
            run_id="run_plan_shape_impossible_source_evidence",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
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
    assert result.fact_result.outcome.kind == OutcomeKind.IMPOSSIBLE
    blocked = result.rendered_fact.details["blockedRequirements"][0]  # type: ignore[union-attr,index]
    assert blocked["kind"] == "policy"
    assert blocked["reviewedReadIds"] == ["payments"]
    assert set(blocked["proofRefs"]) == {
        "endpoint_docs:full_card_numbers_hidden",
        "payments.card_txn_code",
        "payments",
    }
    assert ports.data_access_port.requests == []


def _targetless_product_sales_read(index: int) -> EndpointRead:
    return EndpointRead(
        id=f"product_sales_{index}",
        endpoint_name=f"list_product_sales_{index}",
        resource_names=("product", "sales"),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref=f"field.product_name_{index}",
                path="data.product_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref=f"field.revenue_{index}",
                path="data.revenue",
                row_path_id="data",
                type="number",
            ),
        ),
        source_metadata={
            "description": (
                "Product sales report with product revenue and sales values. "
                "Does not expose target values."
            )
        },
        facts=(
            CatalogFact(
                ref=f"product_sales_{index}.target_values",
                availability=CatalogFactAvailability.POLICY_BLOCKED,
                field_ref=f"field.product_name_{index}",
                read_id=f"product_sales_{index}",
                proof_refs=(f"policy:product_target_values_{index}",),
            ),
        ),
    )
