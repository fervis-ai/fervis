from __future__ import annotations

from decimal import Decimal
import json

from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole

from tests.lookup.orchestrator._helpers import *  # noqa: F403
from tests.lookup.prompt_sections import prompt_section_payload


def test_lookup_cutover_executes_take_limit_value_use_as_proof_link():
    known_inputs = (
        _known_result_limit_input(
            "result_limit",
            "top 2",
            value_text="2",
        ),
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_top_totals",
                description="top metric totals",
                answer_subject=RequestedFactAnswerSubject(subject_text="metric totals"),
                answer_expression=_requested_fact_answer_expression(
                    RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
                    known_inputs=known_inputs,
                ),
                answer_outputs=(
                    RequestedFactAnswerOutput(id="location_name", role="ANSWER_VALUE"),
                ),
                known_inputs=known_inputs,
            ),
        )
    )
    planner = _RawPlannerPort(
        {
            "outcome": {
                "kind": "fact_plan",
                "answers": [
                    {
                        "requested_fact_id": "fact_1",
                        "pattern": "aggregate_by_group",
                        "aggregate_choice": {
                            "group_field_ids": ("location_id",),
                            "metric_field_id": "metric_total",
                            "metric_function": "sum",
                        },
                        "source_hint": {"kind": "read", "read_id": "metric_read"},
                    }
                ],
            }
        },
        question_contract=question_contract,
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="metric_read",
                answer_value_fields=("location_id",),
                group_key_fields=("location_id",),
            ),
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="metric_read",
                measured_value_fields=("metric_total",),
            ),
        ),
    )
    result = run_lookup_question(
        LookupRequest(question="Give me the top 2 metric totals."),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_metric_catalog()),
            data_access_port=_DataAccessPort(
                {
                    "metric_read": {
                        "data": [
                            {
                                "location_id": "loc_a",
                                "location_name": "A",
                                "metric_total": "10.00",
                            },
                            {
                                "location_id": "loc_b",
                                "location_name": "B",
                                "metric_total": "30.00",
                            },
                            {
                                "location_id": "loc_c",
                                "location_name": "C",
                                "metric_total": "20.00",
                            },
                        ]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    assert "30.00" in result.answer
    assert "20.00" in result.answer
    assert "10.00" not in result.answer
    assert result.rendered_fact is not None
    assert "known_input:fact_1_limit_1" in result.rendered_fact.proof_refs


def test_lookup_cutover_list_rows_projected_identity_field_becomes_memory_identity_set():
    planner = _ToolNamePlannerPort(
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="sale ids",
                answer_subject="sale IDs",
                answer_expression_family="list_rows",
                parts=("sale ids",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sale",)),
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
                                "read_id": "list_sale_list",
                            },
                            "output_fields": [{"field_id": "sale_id"}],
                        }
                    ],
                }
            },
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="List sale IDs.",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_sale_detail_catalog()),
            data_access_port=_DataAccessPort(
                {
                    "list_sale_list": {
                        "results": [
                            {"sale_id": "sale_1"},
                            {"sale_id": "sale_2"},
                        ]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    relation_address = next(
        item
        for item in result.fact_addresses
        if item["kind"] == "relation" and item["address"] == "relation.answer_1_rows"
    )
    row_addresses = [
        item
        for item in result.fact_addresses
        if item["kind"] == "row" and item["relation"] == "relation.answer_1_rows"
    ]

    assert result.status == "COMPLETED", result
    assert relation_address["grainKeys"] == ["sale_id"]
    assert [item["identity"] for item in row_addresses] == [
        {"sale_id": "sale_1"},
        {"sale_id": "sale_2"},
    ]


def test_lookup_cutover_grounded_named_entity_is_stored_as_memory_identity():
    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                known_input_resolver_results=(
                    ("fact_1_entity_1", "location:primary_key"),
                ),
            ),
        ),
        source_binding_invocation_overrides=(
            {
                "requested_fact_id": "fact_1",
                "resolved_input_applications": (
                    {
                        "value_id": (
                            "grounded_fact_1_entity_1_location_primary_key_"
                            "location_id_loc_1"
                        ),
                        "value_component": "canonical_key",
                        "target_kind": "request_parameter",
                        "target_id": "location_id",
                    },
                ),
            },
        ),
        responses={
            "submit_question_contract_outcome": _question_contract_response(
                subject="sales at ABC Mall",
                answer_expression_family="list_rows",
                parts=("sales",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "ABC Mall",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "location",
                        "resolved_value_text": "ABC Mall",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("sale",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": "ABC Mall is an instance of location.",
                                "term": "location",
                            }
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
                                "read_id": "sales",
                            },
                            "output_fields": [{"field_id": "sale_id"}],
                        }
                    ],
                }
            },
        },
    )
    result = run_lookup_question(
        LookupRequest(
            question="What were sales at ABC Mall?",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_location_sales_catalog()),
            data_access_port=_DataAccessPort(
                {
                    "list_location_list": {
                        "data": [{"location_id": "loc_1", "name": "ABC Mall"}]
                    },
                    "list_sale_list": {
                        "data": [{"sale_id": "sale_1", "location_id": "loc_1"}]
                    },
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", result
    entity_addresses = [
        item
        for item in result.fact_addresses
        if item["kind"] == "entity" and item["resource"] == "location"
    ]
    assert entity_addresses == [
        {
            "address": (
                "entity.grounded_fact_1_entity_1_location_primary_key_location_id_loc_1"
            ),
            "kind": "entity",
            "resource": "location",
            "keyId": "primary_key",
            "referenceText": "ABC Mall",
            "identity": {"location_id": "loc_1"},
            "evidence": {"stepIds": ["known_input:fact_1_entity_1"]},
        }
    ]


def test_lookup_cutover_standalone_question_does_not_activate_old_identity_set():
    artifact = build_fact_artifact(
        artifact_id="run_prior_stores",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.relation(
                address="relation.stores",
                source={"kind": "api_read", "identityType": "store"},
                grain_keys=("store_id",),
                completeness={
                    "status": "complete",
                    "pagination": "all_pages",
                    "rowCount": 2,
                },
                row_addresses=("row.stores.1", "row.stores.2"),
            ),
            FactAddress.row(
                address="row.stores.1",
                relation="relation.stores",
                identity={"store_id": "store_1"},
            ),
            FactAddress.row(
                address="row.stores.2",
                relation="relation.stores",
                identity={"store_id": "store_2"},
            ),
        ),
    )
    planner = _StandaloneIdentitySetPlannerPort()

    result = run_lookup_question(
        LookupRequest(
            question="Show sales.",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_identity_set_sales_catalog()),
            data_access_port=_IdentitySetDataAccessPort(responses={}),
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED"
    source_prompt = planner.prompts[2]
    assert (
        "mem.run_prior_stores.relation.stores.identity_set.store_id"
        not in source_prompt
    )
    assert "binding_values" not in source_prompt


def test_lookup_orchestrator_repeated_named_target_does_not_reuse_inactive_memory_identity():
    artifact = build_fact_artifact(
        artifact_id="run_prior_location",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                key_id="primary_key",
                reference_text="ABC Mall",
                identity={"location_id": "loc_1"},
            ),
        ),
    )
    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="sales",
                known_input_resolver_results=(
                    ("fact_1_entity_1", "location:primary_key"),
                ),
            ),
        ),
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: (
                lambda prompt: _conversation_resolution_payload_from_prompt(prompt)
            ),
            "submit_question_contract_outcome": _question_contract_response(
                subject="sales at ABC Mall",
                answer_expression_family="list_rows",
                parts=("sales",),
                question_inputs=(
                    {
                        "kind": KnownInputKind.LITERAL.value,
                        "source": "question_context",
                        "source_text": "ABC Mall",
                        "role": LiteralInputRole.REFERENCE_VALUE.value,
                        "value_meaning_hint": "location",
                        "resolved_value_text": "ABC Mall",
                    },
                ),
            ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("sale",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "basis": (
                                    "location can identify ABC Mall because "
                                    "value_meaning_hint is location."
                                ),
                                "term": "location",
                            }
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
                                "read_id": "sales",
                            },
                            "output_fields": [{"field_id": "sale_id"}],
                        }
                    ],
                }
            },
        },
    )
    data_access = _DataAccessPort(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_other", "name": "Other Mall"}]
            },
            "list_sale_list": {
                "data": [
                    {
                        "sale_id": "sale_1",
                        "location_id": "loc_1",
                    }
                ]
            },
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="What were sales at ABC Mall?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_location_sales_catalog()),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "NEEDS_CLARIFICATION", (
        result,
        result,
    )
    assert data_access.requests == [
        {
            "endpointName": "list_location_list",
            "args": {"list_location_list.query.name": "ABC Mall"},
        }
    ]


@dataclass
class _StandaloneIdentitySetPlannerPort:
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        conversation_resolution_payload = _conversation_resolution_payload_from_prompt(
            prompt
        )
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            self.calls += 1
            self.prompts.append(prompt)
            self.system_prompts.append(system_prompt)
            return {
                "answer": json.dumps(
                    {
                        "tool": tool_name,
                        "arguments": conversation_resolution_payload,
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
        if tool_name != "submit_question_contract_outcome":
            del provider, max_thinking_tokens, output_mode
            self.calls += 1
            self.prompts.append(prompt)
            self.system_prompts.append(system_prompt)
            if tool_name == "submit_query_enrichment":
                arguments = _query_enrichment_payload(("sale",))
            elif tool_name == "submit_read_eligibility":
                return read_eligibility_response_for_retained_fields(
                    prompt,
                    answer_value_fields=("sale_id",),
                )
            elif tool_name == "submit_source_binding":
                assert _source_candidate_with_param(prompt, param_id="store_id") is None
                arguments = {"outcome": {"kind": "impossible", "blocked_facts": []}}
            else:
                raise AssertionError(f"unexpected tool: {tool_name}")
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
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return {
            "answer": json.dumps(
                {
                    "tool": tool_name,
                    "arguments": _question_contract_decision(
                        _question_contract_response(
                            subject="sales",
                            parts=("sales",),
                            demand_text="sales",
                        )
                    ),
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


def test_lookup_cutover_coalesces_identical_api_reads_for_multiple_row_relations():
    plan = FactPlan(
        outcome=_answer_plan(
            relations=(
                Relation(
                    id="summary_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="record_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
                Relation(
                    id="detail_rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="record_read",
                    ),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="summary_rows",
                        outputs=(
                            NamedExpression(
                                output_field="name", expression=FieldRef("name")
                            ),
                        ),
                    ),
                    output_relation="answer_rows",
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="name",
                        relation_id="answer_rows",
                        field_id="name",
                    ),
                )
            ),
        )
    )
    data_access = _DataAccessPort(
        {
            "record_read": {
                "data": {
                    "summaries": [{"name": "Summary A"}],
                    "details": [{"name": "Detail A"}],
                }
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(question="What is the summary?", run_id="run_shared_source"),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(
                    EndpointRead(
                        id="record_read",
                        endpoint_name="record_read",
                        resource_names=("summary", "record"),
                        row_paths=(
                            RowPath(
                                id="summaries",
                                path="data.summaries",
                                cardinality=RowCardinality.MANY,
                            ),
                            RowPath(
                                id="details",
                                path="data.details",
                                cardinality=RowCardinality.MANY,
                            ),
                        ),
                        fields=(
                            CatalogField(
                                ref="field.summaries.name",
                                path="data.summaries.name",
                                row_path_id="summaries",
                                type="string",
                            ),
                            CatalogField(
                                ref="field.details.name",
                                path="data.details.name",
                                row_path_id="details",
                                type="string",
                            ),
                        ),
                        pagination=PaginationMetadata(
                            mode=PaginationMode.NONE,
                            completeness_policy=CompletenessPolicy.COMPLETE,
                        ),
                    )
                )
            ),
            data_access_port=data_access,
            planner_model_port=_PlannerPort(
                plan,
                read_eligibility_retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="record_read",
                        row_path_ids=("summaries",),
                        answer_value_fields=("data.summaries.name",),
                    ),
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact.rows == ({"answer_1": "Summary A"},)
    assert data_access.requests == [{"endpointName": "record_read", "args": {}}]


def test_lookup_cutover_preserves_scalar_memory_proofs_for_undefined_compute():
    artifact = build_fact_artifact(
        artifact_id="run_prior_totals",
        outcome=FactOutcome.ANSWERED,
        source_question="What were the prior current and previous totals?",
        source_answer="The prior current total was 10 and the previous total was 0.",
        addresses=(
            FactAddress.value(
                address="value.current",
                value={"type": "number", "value": "10"},
                evidence=EvidenceRef(step_ids=("prior_current",)),
            ),
            FactAddress.value(
                address="value.previous",
                value={"type": "number", "value": "0"},
                evidence=EvidenceRef(step_ids=("prior_previous",)),
            ),
            FactAddress.relation(
                address="relation.context",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                grain_keys=("row_id",),
                completeness={
                    "status": "complete",
                    "setKind": "observed",
                    "pagination": "not_paginated",
                },
                row_addresses=("row.context.1",),
            ),
            FactAddress.row(
                address="row.context.1",
                relation="relation.context",
                grain={"row_id": "context"},
                values={"label": FactAddressValue(type="string", value="prior totals")},
            ),
        ),
    )
    question_contract = _question_contract_for(
        "rf_answer",
        description="percentage increase from prior totals",
        subject_text="percentage increase",
        binding_target_ids=("ratio",),
    )

    result = run_lookup_question(
        LookupRequest(
            question="What percentage increase is that?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_RawPlannerPort(
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "rf_answer",
                                "answer_output_ids": ["ratio"],
                                "pattern": "computed_scalar",
                                "scalar_inputs": [
                                    {
                                        "input_id": "current",
                                        "value_id": ("run_prior_totals.value.current"),
                                    },
                                    {
                                        "input_id": "previous",
                                        "value_id": ("run_prior_totals.value.previous"),
                                    },
                                ],
                                "expression": [
                                    {"input_id": "current"},
                                    {"input_id": "previous"},
                                    {"operator": "divide"},
                                ],
                                "output": {"scalar_id": "ratio", "label": "ratio"},
                            }
                        ],
                    }
                },
                question_contract=question_contract,
                conversation_resolution=lambda prompt: (
                    _conversation_resolution_payload_using_memories(
                        prompt,
                        contextualized_question=(
                            "What percentage increase is there from the prior current "
                            "and previous totals?"
                        ),
                        memories=(
                            {
                                "actual_text": "percentage increase",
                            },
                            {
                                "actual_text": "that",
                            },
                        ),
                    )
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.answer == "division_by_zero"
    assert result.fact_outcome_addresses[0]["evidence"]["stepIds"] == [
        "memory:run_prior_totals.value.current",
        "prior_current",
        "memory:run_prior_totals.value.previous",
        "prior_previous",
        "answer_1_rows_compute",
    ]


def test_lookup_cutover_computes_across_single_cell_prior_answer_relations():
    artifact = build_fact_artifact(
        artifact_id="run_two_date_totals",
        outcome=FactOutcome.ANSWERED,
        source_question="What were the prior daily revenue answers?",
        source_answer="Revenue was 4000.00 on 2025-11-01 and 5000.00 on 2025-11-02.",
        addresses=(
            FactAddress.relation(
                address="relation.answer_1_rows",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                field_coverage={
                    "revenue_on_2025_11_01": ("answer_1_rows.revenue_on_2025_11_01")
                },
                completeness={
                    "status": "complete",
                    "setKind": "observed",
                    "pagination": "not_paginated",
                },
                row_addresses=("row.answer_1_rows.1",),
                evidence=EvidenceRef(step_ids=("prior_day_1",)),
            ),
            FactAddress.row(
                address="row.answer_1_rows.1",
                relation="relation.answer_1_rows",
                values={
                    "revenue_on_2025_11_01": FactAddressValue(
                        type="number",
                        value="4000.00",
                    )
                },
                evidence=EvidenceRef(step_ids=("prior_day_1",)),
            ),
            FactAddress.relation(
                address="relation.answer_2_rows",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                field_coverage={
                    "revenue_on_2025_11_02": ("answer_2_rows.revenue_on_2025_11_02")
                },
                completeness={
                    "status": "complete",
                    "setKind": "observed",
                    "pagination": "not_paginated",
                },
                row_addresses=("row.answer_2_rows.1",),
                evidence=EvidenceRef(step_ids=("prior_day_2",)),
            ),
            FactAddress.row(
                address="row.answer_2_rows.1",
                relation="relation.answer_2_rows",
                values={
                    "revenue_on_2025_11_02": FactAddressValue(
                        type="number",
                        value="5000.00",
                    )
                },
                evidence=EvidenceRef(step_ids=("prior_day_2",)),
            ),
        ),
    )
    first_relation_id = "run_two_date_totals.relation.answer_1_rows"
    second_relation_id = "run_two_date_totals.relation.answer_2_rows"
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="total across the two prior daily revenue answers",
                answer_subject=RequestedFactAnswerSubject(subject_text="total"),
                answer_outputs=(
                    RequestedFactAnswerOutput(id="total", role="ANSWER_VALUE"),
                ),
            ),
        )
    )
    result = run_lookup_question(
        LookupRequest(
            question="How much is that in total?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_RawPlannerPort(
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "rf_answer",
                                "answer_output_ids": ["total"],
                                "pattern": "computed_scalar",
                                "scalar_inputs": [
                                    {
                                        "input_id": "day_1",
                                        "value_id": (
                                            f"{first_relation_id}.value."
                                            "revenue_on_2025_11_01"
                                        ),
                                    },
                                    {
                                        "input_id": "day_2",
                                        "value_id": (
                                            f"{second_relation_id}.value."
                                            "revenue_on_2025_11_02"
                                        ),
                                    },
                                ],
                                "expression": [
                                    {"input_id": "day_1"},
                                    {"input_id": "day_2"},
                                    {"operator": "add"},
                                ],
                                "output": {"scalar_id": "total", "label": "total"},
                            }
                        ],
                    }
                },
                question_contract=question_contract,
                conversation_resolution=lambda prompt: (
                    _conversation_resolution_payload_using_memories(
                        prompt,
                        contextualized_question="How much are the two prior daily revenue answers in total?",
                        memories=(
                            {
                                "actual_text": "that",
                            },
                            {
                                "actual_text": "total",
                            },
                        ),
                    )
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", (
        result,
        result,
    )
    assert result.rendered_fact is not None
    assert result.rendered_fact.scalars == {"total": Decimal("9000.00")}


def test_lookup_cutover_computes_increase_across_multi_row_prior_answer_relation():
    artifact = build_fact_artifact(
        artifact_id="run_two_date_totals",
        outcome=FactOutcome.ANSWERED,
        source_question="What were the prior daily revenue answers?",
        source_answer="Revenue was 4000.00 on 2025-11-01 and 5000.00 on 2025-11-02.",
        addresses=(
            FactAddress.relation(
                address="relation.answer_1_rows",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                grain_keys=("sold_at",),
                field_coverage={
                    "sold_at": "answer_1_rows.sold_at",
                    "total_made": "answer_1_rows.total_made",
                },
                completeness={
                    "status": "complete",
                    "setKind": "observed",
                    "pagination": "not_paginated",
                },
                row_addresses=("row.answer_1_rows.1", "row.answer_1_rows.2"),
                evidence=EvidenceRef(step_ids=("prior_totals",)),
            ),
            FactAddress.row(
                address="row.answer_1_rows.1",
                relation="relation.answer_1_rows",
                grain={"sold_at": "2025-11-01T09:00:00Z"},
                values={
                    "sold_at": FactAddressValue(
                        type="datetime",
                        value="2025-11-01T09:00:00Z",
                    ),
                    "total_made": FactAddressValue(
                        type="number",
                        value="4000.00",
                    ),
                },
                evidence=EvidenceRef(step_ids=("prior_day_1",)),
            ),
            FactAddress.row(
                address="row.answer_1_rows.2",
                relation="relation.answer_1_rows",
                grain={"sold_at": "2025-11-02T09:00:00Z"},
                values={
                    "sold_at": FactAddressValue(
                        type="datetime",
                        value="2025-11-02T09:00:00Z",
                    ),
                    "total_made": FactAddressValue(
                        type="number",
                        value="5000.00",
                    ),
                },
                evidence=EvidenceRef(step_ids=("prior_day_2",)),
            ),
        ),
    )
    relation_id = "run_two_date_totals.relation.answer_1_rows"
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="increase across the prior daily revenue answers",
                answer_subject=RequestedFactAnswerSubject(subject_text="increase"),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="increase",
                        role="ANSWER_VALUE",
                    ),
                ),
            ),
        )
    )
    result = run_lookup_question(
        LookupRequest(
            question="What is the increase between those values?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_RawPlannerPort(
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "rf_answer",
                                "answer_output_ids": ["increase"],
                                "pattern": "computed_scalar",
                                "scalar_inputs": [
                                    {
                                        "input_id": "previous",
                                        "value_id": (
                                            f"{relation_id}.value."
                                            "row.answer_1_rows.1.total_made"
                                        ),
                                    },
                                    {
                                        "input_id": "current",
                                        "value_id": (
                                            f"{relation_id}.value."
                                            "row.answer_1_rows.2.total_made"
                                        ),
                                    },
                                ],
                                "expression": [
                                    {"input_id": "current"},
                                    {"input_id": "previous"},
                                    {"operator": "subtract"},
                                ],
                                "output": {
                                    "scalar_id": "increase",
                                    "label": "increase",
                                },
                            }
                        ],
                    }
                },
                question_contract=question_contract,
                conversation_resolution=lambda prompt: (
                    _conversation_resolution_payload_using_memory(
                        prompt,
                        contextualized_question="What is the increase across the prior daily revenue answers?",
                        actual_text="that",
                        source_kind="row_set",
                    )
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", (
        result,
        result,
    )
    assert result.rendered_fact is not None
    assert result.rendered_fact.scalars == {"increase": Decimal("1000.00")}


def test_lookup_cutover_executes_memory_relation_field_bindings():
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
                values={"quantity": FactAddressValue(type="number", value=7)},
            ),
        ),
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="quantities for prior referenced items",
                answer_subject=RequestedFactAnswerSubject(subject_text="quantities"),
                answer_outputs=(
                    RequestedFactAnswerOutput(id="sku", role="ANSWER_VALUE"),
                    RequestedFactAnswerOutput(id="quantity", role="ANSWER_VALUE"),
                ),
            ),
        )
    )
    result = run_lookup_question(
        LookupRequest(
            question="What quantities were those?",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog()),
            data_access_port=_DataAccessPort({}),
            planner_model_port=_RawPlannerPort(
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "rf_answer",
                                "answer_output_ids": ["sku", "quantity"],
                                "pattern": "list_rows",
                                "source": {
                                    "kind": "memory_relation",
                                    "memory_relation_id": (
                                        "run_prior_items.relation.items"
                                    ),
                                },
                                "output_fields": [
                                    {"field_id": "sku"},
                                    {"field_id": "quantity"},
                                ],
                            }
                        ],
                    }
                },
                question_contract=question_contract,
                conversation_resolution=lambda prompt: (
                    _conversation_resolution_payload_using_memories(
                        prompt,
                        contextualized_question="What quantities were the prior referenced items?",
                        memories=(
                            {
                                "actual_text": "those",
                            },
                        ),
                    )
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", (
        result,
        result,
    )
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == ({"answer_1": "SKU-1", "answer_2": 7},)


def test_lookup_cutover_can_fetch_same_prior_scope_for_additional_fields():
    scope_fingerprint = json.dumps(
        {
            "endpointArgs": {
                "sales_read.query.location_id": "loc_westlands",
                "sales_read.query.start_date": "2025-12-03",
                "sales_read.query.end_date": "2025-12-03",
                "sales_read.query.include_items": True,
            },
            "endpointArgProofRefs": {
                "sales_read.query.location_id": ["known_input:location"],
                "sales_read.query.start_date": ["known_input:date"],
                "sales_read.query.end_date": ["known_input:date"],
                "sales_read.query.include_items": ["source_param:include_items"],
            },
            "rowFilters": [],
        },
        sort_keys=True,
    )
    prior_artifact = build_fact_artifact(
        artifact_id="run_prior_sales",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.relation(
                address="relation.answer_1_rows",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                grain_keys=("staff_name", "product_name"),
                field_coverage={
                    "staff_name": "answer_1_rows.staff_name",
                    "product_name": "answer_1_rows.product_name",
                },
                completeness={
                    "status": "complete",
                    "setKind": "observation",
                    "rowCount": 1,
                    "pagination": "terminal",
                    "scopeFingerprint": scope_fingerprint,
                },
                row_addresses=("row.answer_1_rows.1",),
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
            FactAddress.row(
                address="row.answer_1_rows.1",
                relation="relation.answer_1_rows",
                grain={"staff_name": "Amina", "product_name": "Lipstick"},
                values={
                    "staff_name": FactAddressValue(type="string", value="Amina"),
                    "product_name": FactAddressValue(
                        type="string",
                        value="Lipstick",
                    ),
                },
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
        ),
        source_question=(
            "List salespeople and products sold at Westlands Beauty Hub on "
            "December 3, 2025."
        ),
        source_answer="Amina sold Lipstick.",
    )
    planner = _SameScopeReadPlannerPort(
        prior_reference_id="run_prior_sales.relation.answer_1_rows",
        source_read_id="sales_read",
        source_field_id="shade_name",
    )
    data_access = _DataAccessPort(
        {
            "sales_read": {
                "data": [
                    {
                        "staff_name": "Amina",
                        "product_name": "Lipstick",
                        "shade_name": "Ruby",
                    }
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="Can you show the shade names too?",
            conversation_context={"factArtifacts": [prior_artifact.to_dict()]},
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_same_scope_sales_catalog()),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (
        result,
        planner.prompts,
    )
    assert result.answer == "Ruby"
    source_binding_prompt = next(
        prompt for prompt in planner.prompts if "Candidate evidence sources:" in prompt
    )
    source_payload = _prompt_json_section(
        source_binding_prompt,
        label="Candidate evidence sources",
    )
    memory_candidates = [
        candidate
        for fact_sources in source_payload.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict) and context.get("kind") == "memory_sources"
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    ]
    same_scope_candidate = next(
        candidate
        for candidate in memory_candidates
        if candidate.get("kind") == "same_scope_api_read"
        and candidate.get("read_id") == "sales_read"
    )
    response_rows = same_scope_candidate["response_rows"]
    assert response_rows == [
        {
            "path": "data",
            "cardinality": "many",
            "fields": [
                {"field_id": "staff_name", "path": "data.staff_name", "type": "string"},
                {
                    "field_id": "product_name",
                    "path": "data.product_name",
                    "type": "string",
                },
                {"field_id": "shade_name", "path": "data.shade_name", "type": "string"},
            ],
        }
    ]
    assert same_scope_candidate["population_bindings"][0]["population_binding_id"] == (
        "pop.source_2.prior_scope_replay"
    )
    assert same_scope_candidate["population_bindings"][0]["kind"] == (
        "prior_scope_replay"
    )
    assert data_access.requests == [
        {
            "endpointName": "sales_read",
            "args": {
                "sales_read.query.end_date": "2025-12-03",
                "sales_read.query.include_items": True,
                "sales_read.query.location_id": "loc_westlands",
                "sales_read.query.start_date": "2025-12-03",
            },
        }
    ]


def test_lookup_cutover_same_scope_candidate_survives_catalog_starvation():
    prior_artifact = _same_scope_prior_sales_artifact()
    planner = _SameScopeReadPlannerPort(
        prior_reference_id="run_prior_sales.relation.answer_1_rows",
        source_read_id="sales_read",
        source_field_id="shade_name",
    )
    data_access = _DataAccessPort(
        {
            "sales_read": {
                "data": [
                    {
                        "staff_name": "Amina",
                        "product_name": "Lipstick",
                        "shade_name": "Ruby",
                    }
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="Can you show the shade names too?",
            conversation_context={"factArtifacts": [prior_artifact.to_dict()]},
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
            max_catalog_reads_per_fact=1,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_same_scope_starved_catalog()),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.prompts, result.error)
    assert result.answer == "Ruby"
    assert data_access.requests == [
        {
            "endpointName": "sales_read",
            "args": {
                "sales_read.query.end_date": "2025-12-03",
                "sales_read.query.include_items": True,
                "sales_read.query.location_id": "loc_westlands",
                "sales_read.query.start_date": "2025-12-03",
            },
        }
    ]


def test_lookup_cutover_requires_source_binding_answer_output_fulfillment():
    prior_artifact = _same_scope_prior_sales_artifact()
    planner = _MissingFulfillmentPlannerPort(
        prior_reference_id="run_prior_sales.relation.answer_1_rows",
        source_candidate_id="run_prior_sales.relation.answer_1_rows",
        field_id="product_name",
    )
    data_access = _DataAccessPort(
        {
            "sales_read": {
                "data": [
                    {
                        "staff_name": "Amina",
                        "product_name": "Lipstick",
                        "shade_name": "Ruby",
                    }
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="Can you show the shade names too?",
            conversation_context={"factArtifacts": [prior_artifact.to_dict()]},
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_same_scope_sales_catalog()),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED"
    assert data_access.requests == []


def test_lookup_cutover_rejects_unavailable_fact_plan_source_fields():
    prior_artifact = _same_scope_prior_sales_artifact()
    planner = _UnavailableSourceFieldPlannerPort(
        prior_reference_id="run_prior_sales.relation.answer_1_rows",
        source_read_id="sales_read",
        candidate_field_id="shade_name",
        planned_field_id="missing_field",
    )
    data_access = _DataAccessPort(
        {
            "sales_read": {
                "data": [
                    {
                        "staff_name": "Amina",
                        "product_name": "Lipstick",
                        "shade_name": "Ruby",
                    }
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="Can you show the shade names too?",
            conversation_context={"factArtifacts": [prior_artifact.to_dict()]},
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_same_scope_sales_catalog()),
            data_access_port=data_access,
            planner_model_port=planner,
        ),
    )

    assert result.status == "FAILED"
    assert data_access.requests == []


def test_lookup_cutover_executes_set_difference_without_python_or_old_phases():
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(
            _catalog(
                _variant_read("candidate_read", include_name=True),
                _variant_read("observed_read", include_name=False),
            )
        ),
        data_access_port=_DataAccessPort(
            {
                "candidate_read": {
                    "data": [
                        {"variant_id": "variant_1", "variant_name": "Variant One"},
                        {"variant_id": "variant_2", "variant_name": "Variant Two"},
                    ]
                },
                "observed_read": {
                    "data": [
                        {"variant_id": "variant_1"},
                    ]
                },
            }
        ),
        planner_model_port=_RawPlannerPort(
            {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1"],
                            "pattern": "set_difference",
                            "candidate": {
                                "source": {
                                    "kind": "read",
                                    "read_id": "candidate_read",
                                },
                                "identity_fields": ["variant_id"],
                            },
                            "observed": {
                                "source": {
                                    "kind": "read",
                                    "read_id": "observed_read",
                                },
                                "identity_fields": ["variant_id"],
                            },
                        }
                    ],
                }
            },
            question_contract=_question_contract_for(
                "fact_1",
                description="variant name",
                subject_text="item variants",
                binding_target_ids=("answer_1",),
                answer_output_role="ANSWER_VALUE",
            ),
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="Which item variants were not observed?",
            run_id="run_coverage",
            tenant_id="tenant_1",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED"
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == (
        {
            "answer_1": {
                "entityKind": "variant",
                "keyId": "primary_key",
                "components": {"variant_id": "variant_2"},
            }
        },
    )


def test_lookup_cutover_resolves_generated_calendar_dates_from_time_values():
    question_contract = _question_contract_for(
        "rf_answer",
        description="days in the last 3 days",
        subject_text="days",
        binding_target_ids=("day",),
        answer_expression_family=RequestedFactAnswerExpressionFamily.LIST_ROWS,
        known_inputs=(_known_time_input("month", "last 3 days"),),
    )
    ports = LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(_catalog()),
        data_access_port=_DataAccessPort({}),
        planner_model_port=_RawPlannerPort(
            {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "rf_answer",
                            "answer_output_ids": ["day"],
                            "pattern": "list_rows",
                            "source": {
                                "kind": "calendar",
                                "calendar_id": "calendar_days",
                            },
                            "output_fields": [{"field_id": CALENDAR_DATE_FIELD_ID}],
                        }
                    ],
                }
            },
            question_contract=question_contract,
            source_binding_invocation_overrides=(
                {
                    "requested_fact_id": "fact_1",
                    "resolved_input_applications": (
                        {
                            "value_id": "grounded_fact_1_time_1",
                            "value_component": "start",
                            "target_kind": "request_parameter",
                            "target_id": "interval_start",
                        },
                        {
                            "value_id": "grounded_fact_1_time_1",
                            "value_component": "end",
                            "target_kind": "request_parameter",
                            "target_id": "interval_end",
                        },
                    ),
                },
            ),
        ),
    )

    result = run_lookup_question(
        LookupRequest(
            question="Which days are in the last 3 days?",
            run_id="run_calendar",
            tenant_id="tenant_1",
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-03",
                timezone="Africa/London",
            ),
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        ports,
    )

    assert result.status == "COMPLETED"
    assert result.rendered_fact.rows == (  # type: ignore[union-attr]
        {"answer_1": "2026-05-01"},
        {"answer_1": "2026-05-02"},
        {"answer_1": "2026-05-03"},
    )


def _list_rows_fact_plan(
    *,
    read_id: str = "",
    field_id: str = "",
    source_kind: str = "read",
) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "requested_fact_id": "fact_1",
        "pattern": "list_rows",
    }
    if read_id:
        answer["source"] = {
            "kind": source_kind,
            "read_id": read_id,
        }
    if field_id:
        answer["output_fields"] = [{"field_id": field_id}]
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [answer],
        }
    }


@dataclass
class _SameScopeReadPlannerPort:
    prior_reference_id: str
    source_read_id: str
    source_field_id: str
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        is_conversation_resolution = bool(
            _offered_conversation_resolution_tool_names(tool_specs)
        )
        if is_conversation_resolution:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="Show shade names for the prior referenced salespeople and products sold.",
                actual_text="the shade names too",
                source_kind="row_set",
            )
            tool_name = CONVERSATION_RESOLUTION_TOOL_NAME
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
        else:
            tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_response(
                    subject="shade names too",
                    answer_expression_family="list_rows",
                    parts=("shade names",),
                    demand_text="shade",
                )
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload(("sale",))
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                read_id=self.source_read_id,
                answer_value_fields=(self.source_field_id,),
            )
        elif tool_name == "submit_source_binding":
            arguments = source_binding_payload_from_fact_plan(
                _list_rows_fact_plan(
                    read_id=self.source_read_id,
                    field_id=self.source_field_id,
                    source_kind="same_scope_api_read",
                ),
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _list_rows_fact_plan(
                    read_id=self.source_read_id,
                    field_id=self.source_field_id,
                    source_kind="same_scope_api_read",
                ),
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
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": _required_output_fields_from_prompt(
                                prompt,
                                fallback_field_ids=(self.source_field_id,),
                            ),
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
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


@dataclass
class _SameScopeReadOutputPlannerPort:
    prior_reference_id: str
    source_candidate_id: str
    field_id: str
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="Show shade names for the prior referenced salespeople and products sold.",
                actual_text="the shade names too",
                source_kind="row_set",
            )
        elif tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_response(
                    subject="shade names too",
                    answer_expression_family="list_rows",
                    parts=("shade names",),
                    demand_text="shade",
                )
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload(("shade",))
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                answer_value_fields=(self.field_id,),
            )
        elif tool_name == "submit_source_binding":
            payload = _prompt_json_section(
                prompt,
                label="Candidate evidence sources",
            )
            candidate = source_candidate_with_fields(
                payload,
                kind="same_scope_api_read",
                required=(self.field_id,),
                forbidden=(),
            )
            candidate_id = str(candidate["source_candidate_id"])
            binding_target_id = source_binding_target_id_for_candidate(
                prompt,
                requested_fact_id="fact_1",
                source_candidate_id=candidate_id,
            )
            arguments = source_binding_payload_for_one_call(
                {
                    "outcome": {
                        "kind": "source_bindings",
                        "source_invocations": [
                            {
                                "requested_fact_id": "fact_1",
                                "source_candidate_id": candidate_id,
                                "answer_population": source_candidate_answer_population(
                                    prompt,
                                    binding_target_id=binding_target_id,
                                ),
                                "param_decisions": {},
                            }
                        ],
                    }
                },
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _list_rows_fact_plan(
                    read_id="sales_read",
                    field_id=self.field_id,
                    source_kind="same_scope_api_read",
                ),
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
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": _required_output_fields_from_prompt(
                                prompt,
                                fallback_field_ids=(self.field_id,),
                            ),
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
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


@dataclass
class _IdentitySetPlannerPort:
    prior_reference_id: str
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        is_conversation_resolution = bool(
            _offered_conversation_resolution_tool_names(tool_specs)
        )
        if is_conversation_resolution:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="Show sales for the prior referenced stores.",
                actual_text="those stores",
                source_kind="row_set",
            )
            tool_name = CONVERSATION_RESOLUTION_TOOL_NAME
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
        else:
            tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_response(
                    subject="sales for those stores",
                    answer_expression_family="list_rows",
                    parts=("sales",),
                    demand_text="sales",
                    question_inputs=(
                        {
                            "kind": "row_set_reference",
                            "reference_text": "those stores",
                            "source": "conversation_resolution",
                            "resolved_input_ref": "cr_input_1",
                        },
                    ),
                )
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload(("sale",))
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                answer_value_fields=("store_id",),
            )
        elif tool_name == "submit_source_binding":
            candidate = _source_candidate_with_param(prompt, param_id="store_id")
            assert candidate is not None
            bind_option = _bind_option_for_param(candidate, param_id="store_id")
            param_decisions = {
                "store_id": {
                    "population_intent": "sales for those stores",
                    "match_basis_explanation": (
                        f"{bind_option['meaning']} This matches sales for those stores "
                        "because those stores are the prior complete identity set."
                    ),
                    "param_decision_id": bind_option["param_decision_id"],
                }
            }
            source_binding_arguments = {
                "outcome": {
                    "kind": "source_bindings",
                    "source_invocations": [
                        {
                            "requested_fact_id": "fact_1",
                            "source_candidate_id": candidate["source_candidate_id"],
                            "answer_population": {
                                "population_binding_id": _candidate_binding_surface(
                                    candidate
                                )["population_bindings"][0]["population_binding_id"],
                                "intent_text": "sales for those stores",
                                "match_basis_explanation": (
                                    "sales for those stores defines the source population"
                                ),
                            },
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                candidate=candidate,
                                field_ids=("store_id",),
                            ),
                            "param_decisions": param_decisions,
                        }
                    ],
                }
            }
            arguments = source_binding_payload_for_one_call(
                source_binding_arguments,
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _list_rows_fact_plan(
                    read_id="sales_read",
                    field_id="store_id",
                ),
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
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": _required_output_fields_from_prompt(
                                prompt,
                                fallback_field_ids=("store_id",),
                            ),
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
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


@dataclass
class _SaleDetailIdentitySetPlannerPort:
    prior_reference_id: str
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        is_conversation_resolution = bool(
            _offered_conversation_resolution_tool_names(tool_specs)
        )
        if is_conversation_resolution:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="Show product names for the prior referenced sales.",
                actual_text="those sales",
                source_kind="row_set",
            )
            tool_name = CONVERSATION_RESOLUTION_TOOL_NAME
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
        else:
            tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_response(
                    subject="those sales",
                    answer_expression_family="list_rows",
                    parts=("product names",),
                    demand_text="product",
                    question_inputs=(
                        {
                            "kind": "row_set_reference",
                            "reference_text": "those sales",
                            "source": "conversation_resolution",
                            "resolved_input_ref": "cr_input_1",
                        },
                    ),
                )
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload(("sale",))
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                read_id="get_sale_detail",
                answer_value_fields=("product_name",),
            )
        elif tool_name == "submit_source_binding":
            candidate = _source_candidate_with_param(
                prompt,
                param_id="sale_id",
                read_id="get_sale_detail",
            )
            assert candidate is not None
            assert _candidate_read_id(candidate) == "get_sale_detail"
            bind_option = _bind_option_for_param(candidate, param_id="sale_id")
            param_decisions = {
                "sale_id": {
                    "population_intent": "those sales",
                    "match_basis_explanation": (
                        f"{bind_option['meaning']} This matches those sales "
                        "because those sales are the prior complete identity set."
                    ),
                    "param_decision_id": bind_option["param_decision_id"],
                }
            }
            source_binding_arguments = {
                "outcome": {
                    "kind": "source_bindings",
                    "source_invocations": [
                        {
                            "requested_fact_id": "fact_1",
                            "source_candidate_id": candidate["source_candidate_id"],
                            "answer_population": {
                                "population_binding_id": _candidate_binding_surface(
                                    candidate
                                )["population_bindings"][0]["population_binding_id"],
                                "intent_text": "those sales",
                                "match_basis_explanation": (
                                    "those sales defines the source population"
                                ),
                            },
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                candidate,
                                field_ids=("product_name",),
                            ),
                            "param_decisions": param_decisions,
                        }
                    ],
                }
            }
            arguments = source_binding_payload_for_one_call(
                source_binding_arguments,
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _list_rows_fact_plan(
                    read_id="get_sale_detail",
                    field_id="product_name",
                ),
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
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": _required_output_fields_from_prompt(
                                prompt,
                                fallback_field_ids=("product_name",),
                            ),
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
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


@dataclass
class _MissingFulfillmentPlannerPort:
    prior_reference_id: str
    source_candidate_id: str
    field_id: str
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="Show shade names for the prior referenced salespeople and products sold.",
                actual_text="the shade names too",
                source_kind="row_set",
            )
        elif tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_response(
                    subject="shade names too",
                    answer_expression_family="list_rows",
                    parts=("shade names",),
                    demand_text="shade",
                )
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload(("shade",))
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                answer_value_fields=(self.field_id,),
            )
        elif tool_name == "submit_source_binding":
            arguments = source_binding_payload_for_one_call(
                {
                    "outcome": {
                        "kind": "source_bindings",
                        "source_invocations": [
                            {
                                "requested_fact_id": "fact_1",
                                "source_candidate_id": self.source_candidate_id,
                                "answer_population": {
                                    "population_binding_id": f"pop.{self.source_candidate_id}.candidate_population",
                                    "intent_text": "shade names too",
                                    "match_basis_explanation": "shade names too defines the source population",
                                },
                                "param_decisions": {},
                            }
                        ],
                    }
                },
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _list_rows_fact_plan(),
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
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": [{"field_id": self.field_id}],
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
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


@dataclass
class _UnavailableSourceFieldPlannerPort:
    prior_reference_id: str
    source_read_id: str
    candidate_field_id: str
    planned_field_id: str
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)

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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="Show shade names for the prior referenced salespeople and products sold.",
                actual_text="the shade names too",
                source_kind="row_set",
            )
        elif tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(
                _question_contract_response(
                    subject="shade names too",
                    answer_expression_family="list_rows",
                    parts=("shade names",),
                )
            )
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload(("shade",))
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                read_id=self.source_read_id,
                answer_value_fields=(self.candidate_field_id,),
            )
        elif tool_name == "submit_source_binding":
            candidate = _same_scope_source_candidate(
                prompt,
                read_id=self.source_read_id,
                field_id=self.candidate_field_id,
            )
            arguments = source_binding_payload_for_one_call(
                {
                    "outcome": {
                        "kind": "source_bindings",
                        "source_invocations": [
                            {
                                "requested_fact_id": "fact_1",
                                "source_candidate_id": candidate["source_candidate_id"],
                                "answer_population": {
                                    "population_binding_id": candidate[
                                        "population_bindings"
                                    ][0]["population_binding_id"],
                                    "intent_text": "shade names too",
                                    "match_basis_explanation": "shade names too defines the source population",
                                },
                                "fulfillment_decisions": source_fulfills_for_candidate(
                                    candidate=candidate,
                                    field_ids=(self.candidate_field_id,),
                                ),
                                "param_decisions": {},
                            }
                        ],
                    }
                },
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                _list_rows_fact_plan(),
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
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": _required_output_fields_from_prompt(
                                prompt,
                                fallback_field_ids=(self.candidate_field_id,),
                            ),
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
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


def _same_scope_prior_sales_artifact() -> FactArtifact:
    scope_fingerprint = json.dumps(
        {
            "endpointArgs": {
                "sales_read.query.location_id": "loc_westlands",
                "sales_read.query.start_date": "2025-12-03",
                "sales_read.query.end_date": "2025-12-03",
                "sales_read.query.include_items": "true",
            },
            "endpointArgProofRefs": {
                "sales_read.query.location_id": ["known_input:location"],
                "sales_read.query.start_date": ["known_input:date"],
                "sales_read.query.end_date": ["known_input:date"],
                "sales_read.query.include_items": ["source_param:include_items"],
            },
            "rowFilters": [],
        },
        sort_keys=True,
    )
    return build_fact_artifact(
        artifact_id="run_prior_sales",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.relation(
                address="relation.answer_1_rows",
                source={"kind": RelationSourceKind.OPERATION_OUTPUT.value},
                grain_keys=("staff_name", "product_name"),
                field_coverage={
                    "staff_name": "answer_1_rows.staff_name",
                    "product_name": "answer_1_rows.product_name",
                },
                completeness={
                    "status": "complete",
                    "setKind": "observation",
                    "rowCount": 1,
                    "pagination": "terminal",
                    "scopeFingerprint": scope_fingerprint,
                },
                row_addresses=("row.answer_1_rows.1",),
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
            FactAddress.row(
                address="row.answer_1_rows.1",
                relation="relation.answer_1_rows",
                grain={"staff_name": "Amina", "product_name": "Lipstick"},
                values={
                    "staff_name": FactAddressValue(type="string", value="Amina"),
                    "product_name": FactAddressValue(
                        type="string",
                        value="Lipstick",
                    ),
                },
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
        ),
        source_question=(
            "List salespeople and products sold at Westlands Beauty Hub on "
            "December 3, 2025."
        ),
        source_answer="Amina sold Lipstick.",
    )


def _same_scope_source_candidate(
    prompt: str,
    *,
    read_id: str,
    field_id: str,
) -> dict[str, Any]:
    if "Selected source invocations:\n" in prompt:
        payload = _prompt_json_section(prompt, label="Selected source invocations")
        candidates = [
            candidate
            for candidate in payload.get("source_invocations") or ()
            if isinstance(candidate, dict)
        ]
    else:
        payload = _prompt_json_section(prompt, label="Candidate evidence sources")
        candidates = _all_source_candidates(payload)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("kind") not in {None, "same_scope_api_read"}:
            continue
        if _candidate_read_id(candidate) not in {"", read_id}:
            continue
        if "Selected source invocations:\n" in prompt or _candidate_has_field(
            candidate,
            field_id,
        ):
            return candidate
    raise AssertionError("same_scope_api_read candidate not found")


def _source_candidate_with_param(
    prompt: str,
    *,
    param_id: str,
    read_id: str = "",
) -> dict[str, Any] | None:
    for candidate in _requested_fact_source_candidates(prompt):
        if (
            read_id
            and _candidate_read_id(candidate)
            and _candidate_read_id(candidate) != read_id
        ):
            continue
        if any(
            isinstance(param, dict) and param.get("param_id") == param_id
            for param in _candidate_binding_surface(candidate).get("params") or ()
        ):
            return candidate
    return None


def _bind_option_for_param(
    candidate: dict[str, Any],
    *,
    param_id: str,
) -> dict[str, Any]:
    for param in _candidate_binding_surface(candidate).get("params") or ():
        if not isinstance(param, dict) or param.get("param_id") != param_id:
            continue
        for option in param.get("decision_options") or ():
            if isinstance(option, dict) and option.get("decision") == "bind":
                return option
    raise AssertionError(f"bind option not found for {param_id}")


def _requested_fact_source_candidates(prompt: str) -> list[dict[str, Any]]:
    if "Selected source invocations:\n" in prompt:
        payload = _prompt_json_section(prompt, label="Selected source invocations")
        return [
            candidate
            for candidate in payload.get("source_invocations") or ()
            if isinstance(candidate, dict)
        ]
    payload = _prompt_json_section(prompt, label="Candidate evidence sources")
    return _all_source_candidates(payload)


def _all_source_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        item
        for key in (
            "memory_source_candidates",
            "utility_source_candidates",
            "value_source_candidates",
        )
        for item in payload.get(key) or ()
        if isinstance(item, dict)
    )
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        candidates.extend(_source_options_for_fact_sources(fact_sources))
    return candidates


def _candidate_has_field(candidate: dict[str, Any], field_id: str) -> bool:
    for item in _candidate_binding_surface(candidate).get("evidence_items") or ():
        if not isinstance(item, dict):
            continue
        if item.get("field_id") == field_id:
            return True
    for item in candidate.get("fields") or ():
        if not isinstance(item, dict):
            continue
        if item.get("field_id") == field_id or item.get("id") == field_id:
            return True
    for support_set in (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if not isinstance(support_set, dict):
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for key in (
                "metric_measure_evidence",
                "row_count_basis_evidence",
                "scope_evidence",
                "group_key_evidence",
            ):
                for item in slot.get(key) or ():
                    if isinstance(item, dict) and item.get("field_id") == field_id:
                        return True
    return False


def _required_output_fields_from_prompt(
    prompt: str,
    *,
    fallback_field_ids: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    try:
        evidence_payload = _prompt_json_section(
            prompt,
            label="Required fulfillment evidence",
        )
    except (AssertionError, ValueError):
        evidence_payload = {}
    field_ids = tuple(
        dict.fromkeys(
            str(item.get("field_id") or "")
            for requirement in evidence_payload.get("required_fulfillment_evidence")
            or ()
            if isinstance(requirement, dict)
            for item in requirement.get("must_use_evidence") or ()
            if isinstance(item, dict) and str(item.get("field_id") or "")
        )
    )
    if not field_ids:
        field_ids = tuple(
            dict.fromkeys(field_id for field_id in fallback_field_ids if field_id)
        )
    if not field_ids:
        raise AssertionError("pattern prompt must expose required fulfillment evidence")
    return [{"field_id": field_id} for field_id in field_ids]


def _candidate_binding_surface(candidate: dict[str, Any]) -> dict[str, Any]:
    surface = candidate.get("binding_surface")
    if isinstance(surface, dict):
        return surface
    if candidate.get("kind") not in {"new_api_read", "same_scope_api_read"}:
        return candidate
    output = {
        key: candidate[key]
        for key in (
            "applied_filters",
            "bound_params",
            "source_invocations",
            "population_bindings",
            "params",
            "population_roles",
        )
        if key in candidate
    }
    if "fulfillment_choices" in candidate:
        output["fulfillment_support_sets"] = candidate["fulfillment_choices"]
    fields = [
        field
        for row in candidate.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    ]
    if fields:
        output["fields"] = fields
    return output


def _candidate_read_id(candidate: dict[str, Any]) -> str:
    read_id = str(candidate.get("read_id") or "")
    if read_id:
        return read_id
    read_contract = candidate.get("read_contract")
    if isinstance(read_contract, dict):
        return str(read_contract.get("read_id") or "")
    return ""


def _source_options_for_fact_sources(
    fact_sources: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        candidate
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
    )


def _prompt_json_section(prompt: str, *, label: str) -> dict[str, Any]:
    return prompt_section_payload(prompt, label)


def _primary_key(
    entity_kind: str,
    component_id: str,
    field_ref: str,
    *,
    context_field_refs: tuple[str, ...] = (),
) -> tuple[CandidateKey, ...]:
    return (
        CandidateKey(
            id="primary_key",
            entity_kind=entity_kind,
            components=(CandidateKeyComponent(id=component_id, field_ref=field_ref),),
            primary=True,
            context_field_refs=context_field_refs,
        ),
    )


def _entity_target(entity_kind: str, component_id: str) -> EntityKeyComponentTarget:
    return EntityKeyComponentTarget(
        entity_kind=entity_kind,
        key_id="primary_key",
        component_id=component_id,
    )


def _entity_reference(
    entity_kind: str,
    component_id: str,
    field_ref: str,
) -> tuple[EntityReference, ...]:
    return (
        EntityReference(
            id=f"{entity_kind}_reference",
            target_entity_kind=entity_kind,
            target_key_id="primary_key",
            components=(
                EntityReferenceComponent(
                    target_component_id=component_id,
                    local_field_ref=field_ref,
                ),
            ),
        ),
    )


def _identity_set_sales_catalog() -> RelationCatalog:
    return _catalog(
        EndpointRead(
            id="sales",
            endpoint_name="list_sale_list",
            resource_names=("sale",),
            params=(
                CatalogParam(
                    ref="sales.query.store_id",
                    name="store_id",
                    source=ParamSource.QUERY,
                    type="uuid",
                    entity_target=_entity_target("store", "store_id"),
                ),
            ),
            row_paths=(
                RowPath(id="results", path="results", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="sales.field.sale_id",
                    path="results.sale_id",
                    row_path_id="results",
                    type="uuid",
                ),
                CatalogField(
                    ref="sales.field.store_id",
                    path="results.store_id",
                    row_path_id="results",
                    type="uuid",
                ),
            ),
            candidate_keys=_primary_key("sale", "sale_id", "sales.field.sale_id"),
            entity_references=_entity_reference(
                "store", "store_id", "sales.field.store_id"
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
        EndpointRead(
            id="stores",
            endpoint_name="list_store_list",
            resource_names=("store",),
            row_paths=(
                RowPath(id="results", path="results", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="stores.field.store_id",
                    path="results.store_id",
                    row_path_id="results",
                    type="uuid",
                ),
            ),
            candidate_keys=_primary_key("store", "store_id", "stores.field.store_id"),
        ),
    )


def _location_sales_catalog() -> RelationCatalog:
    return _catalog(
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
                    ref="locations.field.location_id",
                    path="data.location_id",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="locations.field.name",
                    path="data.name",
                    row_path_id="data",
                    type="string",
                ),
            ),
            candidate_keys=_primary_key(
                "location",
                "location_id",
                "locations.field.location_id",
                context_field_refs=("locations.field.name",),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
        EndpointRead(
            id="sales",
            endpoint_name="list_sale_list",
            resource_names=("sale",),
            params=(
                CatalogParam(
                    ref="sales.query.location_id",
                    name="location_id",
                    source=ParamSource.QUERY,
                    type="string",
                    entity_target=_entity_target("location", "location_id"),
                ),
            ),
            row_paths=(
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="sales.field.sale_id",
                    path="data.sale_id",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="sales.field.location_id",
                    path="data.location_id",
                    row_path_id="data",
                    type="string",
                ),
            ),
            candidate_keys=_primary_key("sale", "sale_id", "sales.field.sale_id"),
            entity_references=_entity_reference(
                "location", "location_id", "sales.field.location_id"
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
    )


@dataclass
class _IdentitySetDataAccessPort:
    responses: dict[str, dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        value = str(args["sales.query.store_id"])
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": self.responses[value],
            "truncated": False,
            "pageCount": 1,
        }


@dataclass
class _SaleDetailDataAccessPort:
    responses: dict[str, dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        value = str(args["sale_detail.path.sale_id"])
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": self.responses[value],
            "truncated": False,
            "pageCount": 1,
        }


def _sale_detail_catalog() -> RelationCatalog:
    return _catalog(
        EndpointRead(
            id="list_sale_list",
            endpoint_name="list_sale_list",
            resource_names=("sale",),
            row_paths=(
                RowPath(id="results", path="results", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="sale_list.field.sale_id",
                    path="results.sale_id",
                    row_path_id="results",
                    type="uuid",
                ),
            ),
            candidate_keys=_primary_key("sale", "sale_id", "sale_list.field.sale_id"),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
        EndpointRead(
            id="get_sale_detail",
            endpoint_name="get_sale_detail",
            resource_names=("sale",),
            params=(
                CatalogParam(
                    ref="sale_detail.path.sale_id",
                    name="sale_id",
                    source=ParamSource.PATH,
                    type="uuid",
                    required=True,
                    entity_target=_entity_target("sale", "sale_id"),
                ),
            ),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="items", path="items", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="sale_detail.field.sale_id",
                    path="sale_id",
                    row_path_id="root",
                    type="uuid",
                ),
                CatalogField(
                    ref="sale_detail.field.product_name",
                    path="items.product_name",
                    row_path_id="items",
                    type="string",
                ),
            ),
            candidate_keys=_primary_key("sale", "sale_id", "sale_detail.field.sale_id"),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        ),
    )


def _same_scope_sales_catalog() -> RelationCatalog:
    return _catalog(
        EndpointRead(
            id="sales_read",
            endpoint_name="sales_read",
            resource_names=("sale",),
            params=(
                CatalogParam(
                    ref="sales_read.query.location_id",
                    name="location_id",
                    source=ParamSource.QUERY,
                    type="string",
                ),
                CatalogParam(
                    ref="sales_read.query.start_date",
                    name="start_date",
                    source=ParamSource.QUERY,
                    type="date",
                ),
                CatalogParam(
                    ref="sales_read.query.end_date",
                    name="end_date",
                    source=ParamSource.QUERY,
                    type="date",
                ),
                CatalogParam(
                    ref="sales_read.query.include_items",
                    name="include_items",
                    source=ParamSource.QUERY,
                    type="boolean",
                ),
            ),
            row_paths=(
                RowPath(id="root", path="", cardinality=RowCardinality.ONE),
                RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
            ),
            fields=(
                CatalogField(
                    ref="sales_read.field.staff_name",
                    path="data.staff_name",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="sales_read.field.product_name",
                    path="data.product_name",
                    row_path_id="data",
                    type="string",
                ),
                CatalogField(
                    ref="sales_read.field.shade_name",
                    path="data.shade_name",
                    row_path_id="data",
                    type="string",
                ),
            ),
            pagination=PaginationMetadata(
                mode=PaginationMode.NONE,
                completeness_policy=CompletenessPolicy.COMPLETE,
            ),
        )
    )


def _same_scope_starved_catalog() -> RelationCatalog:
    return _catalog(_shade_lookup_decoy_read(), *_same_scope_sales_catalog().reads)


def _shade_lookup_decoy_read() -> EndpointRead:
    return EndpointRead(
        id="shade_lookup_read",
        endpoint_name="shade_lookup_read",
        resource_names=("shade",),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="shade_lookup_read.field.shade_name",
                path="data.shade_name",
                row_path_id="data",
                type="string",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )
