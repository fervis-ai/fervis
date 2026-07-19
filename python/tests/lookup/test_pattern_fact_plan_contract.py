from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import json
import re
from typing import Any

from fervis.lookup.orchestration.pipeline import run_lookup_question
from fervis.lookup.orchestration.request import LookupRequest, LookupRuntimePorts
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    CONVERSATION_RESOLUTION_TOOL_NAMES,
)
from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    CompletenessPolicy,
    EndpointRead,
    FieldRequirement,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.question_contract import (
    RequestedFactAnswerSubject,
    default_answer_population,
)
from fervis.memory.addresses import (
    EvidenceRef,
    FactAddress,
    FactAddressValue,
    RelationSourceKind,
)
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from tests.lookup.source_binding_helpers import (
    bound_fact_plan_payload_from_fact_plan,
    plan_selection_payload_from_fact_plan,
    source_candidate_answer_population,
    source_candidate_with_fields,
    source_candidate_with_kind,
    source_fulfills_fields_for_candidate,
    source_fulfills_for_candidate,
    source_binding_payload_for_one_call,
    source_binding_payload_from_fact_plan,
    source_binding_target_id_for_candidate,
)
from tests.lookup.orchestrator._payloads import (
    ReadEligibilityRetentionSpec,
    _question_contract_decision,
    _query_enrichment_payload,
    _query_enrichment_payload_from_prompt,
    _conversation_resolution_clause_payload,
    read_eligibility_response_from_prompt,
    read_eligibility_response_from_fact_plan,
)
from tests.lookup.orchestrator._runtime_ports import (
    _grounding_payload_from_prompt,
)
from tests.lookup.prompt_sections import prompt_section_payload


def _conversation_resolution_tool_name_for_payload(payload: dict[str, Any]) -> str:
    del payload
    return CONVERSATION_RESOLUTION_TOOL_NAME


def _primary_key(
    entity_kind: str,
    component_id: str,
    field_ref: str,
) -> tuple[CandidateKey, ...]:
    return (
        CandidateKey(
            id="primary_key",
            entity_kind=entity_kind,
            components=(CandidateKeyComponent(id=component_id, field_ref=field_ref),),
            primary=True,
        ),
    )


def test_pattern_contract_source_binding_uses_opaque_source_handles_end_to_end():
    result = run_lookup_question(
        LookupRequest(
            question="List location metric totals.",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        _ports(
            question_contract=_question_contract(
                fact_description="location metric totals",
                answer_outputs=("location", "metric_total"),
            ),
            fact_plan=_fact_plan_answer(
                pattern="list_rows",
                source={"kind": "read", "read_id": "metric_read"},
                answer_output_ids=("answer_1", "answer_2"),
                output_fields=(
                    {"field_id": "location_name"},
                    {"field_id": "metric_total"},
                ),
            ),
            planner_port_cls=_OpaqueSourceHandlePlannerPort,
            responses={
                "metric_read": {
                    "data": [
                        {
                            "location_id": "location_alpha",
                            "location_name": "Location Alpha",
                            "metric_total": "125.00",
                        }
                    ]
                }
            },
        ),
    )

    assert result.status == "COMPLETED", result
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == (
        {"answer_1": "Location Alpha", "answer_2": Decimal("125.00")},
    )


def test_pattern_contract_fact_plan_prompt_uses_bound_sources_end_to_end():
    data_access = _DataAccessPort(
        {
            "records_read": {
                "data": [
                    {"record_id": "record_1"},
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many completed store records are there?",
            provider_preferences={"provider": "fake", "modelKey": "FAKE"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_records_catalog_with_optional_params()),
            data_access_port=data_access,
            planner_model_port=_FactPlanPromptInspectingPlannerPort(
                question_contract=_question_contract(
                    fact_description="count of completed store records",
                    subject_text="records",
                    answer_outputs=("count",),
                    answer_output_roles=("ROW_COUNT",),
                ),
                fact_plan=_fact_plan_answer(
                    pattern="aggregate_scalar",
                    source={
                        "kind": "read",
                        "read_id": "records_read",
                        "param_bindings": (
                            {"param_id": "channel", "value": "STORE"},
                            {"param_id": "status", "value": "COMPLETED"},
                        ),
                    },
                    answer_output_ids=("answer_1",),
                    metric={
                        "kind": "count_records",
                        "count_basis": {
                            "kind": "row_population",
                            "row_path_id": "data",
                            "row_cardinality": "many",
                        },
                        "label": "count",
                    },
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert data_access.requests == [
        {
            "endpointName": "records_read",
            "args": {
                "records_read.query.channel": "STORE",
                "records_read.query.status": "COMPLETED",
            },
        }
    ]


def test_pattern_contract_anthropic_uses_canonical_source_binding_end_to_end():
    data_access = _DataAccessPort(
        {
            "records_read": {
                "data": [
                    {"record_id": "record_1"},
                    {"record_id": "record_2"},
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many completed store records are there?",
            provider_preferences={"provider": "anthropic", "modelKey": "HAIKU"},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_records_catalog_with_optional_params()),
            data_access_port=data_access,
            planner_model_port=_CanonicalSourceBindingPlannerPort(
                question_contract=_question_contract(
                    fact_description="count of completed store records",
                    subject_text="records",
                    answer_outputs=("count",),
                    answer_output_roles=("ROW_COUNT",),
                ),
                fact_plan=_fact_plan_answer(
                    pattern="aggregate_scalar",
                    source={
                        "kind": "read",
                        "read_id": "records_read",
                        "param_bindings": (
                            {"param_id": "channel", "value": "STORE"},
                            {"param_id": "status", "value": "COMPLETED"},
                        ),
                    },
                    answer_output_ids=("answer_1",),
                    metric={
                        "kind": "count_records",
                        "count_basis": {
                            "kind": "row_population",
                            "row_path_id": "data",
                            "row_cardinality": "many",
                        },
                        "label": "count",
                    },
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert data_access.requests == [
        {
            "endpointName": "records_read",
            "args": {
                "records_read.query.channel": "STORE",
                "records_read.query.status": "COMPLETED",
            },
        }
    ]
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == ({"count": 2},)


def test_pattern_contract_gpt_keeps_canonical_source_binding_end_to_end():
    data_access = _DataAccessPort(
        {
            "records_read": {
                "data": [
                    {"record_id": "record_1"},
                    {"record_id": "record_2"},
                ]
            }
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="How many completed store records are there?",
            provider_preferences={
                "provider": "openai",
                "modelKey": "GPT_5_4_MINI",
            },
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_records_catalog_with_optional_params()),
            data_access_port=data_access,
            planner_model_port=_CanonicalSourceBindingPlannerPort(
                question_contract=_question_contract(
                    fact_description="count of completed store records",
                    subject_text="records",
                    answer_outputs=("count",),
                    answer_output_roles=("ROW_COUNT",),
                ),
                fact_plan=_fact_plan_answer(
                    pattern="aggregate_scalar",
                    source={
                        "kind": "read",
                        "read_id": "records_read",
                        "param_bindings": (
                            {"param_id": "channel", "value": "STORE"},
                            {"param_id": "status", "value": "COMPLETED"},
                        ),
                    },
                    answer_output_ids=("answer_1",),
                    metric={
                        "kind": "count_records",
                        "count_basis": {
                            "kind": "row_population",
                            "row_path_id": "data",
                            "row_cardinality": "many",
                        },
                        "label": "count",
                    },
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert data_access.requests == [
        {
            "endpointName": "records_read",
            "args": {
                "records_read.query.channel": "STORE",
                "records_read.query.status": "COMPLETED",
            },
        }
    ]


def test_pattern_contract_same_scope_candidate_keeps_field_params_together_end_to_end():
    prior_scope_without_items = json.dumps(
        {
            "endpointArgs": {
                "sales_read.query.include_items": "false",
                "sales_read.query.location_id": "loc_westlands",
            },
            "endpointArgProofRefs": {
                "sales_read.query.include_items": ["source_param:include_items"],
                "sales_read.query.location_id": ["known_input:location"],
            },
            "rowFilters": [],
        },
        sort_keys=True,
    )
    prior_scope_with_items = json.dumps(
        {
            "endpointArgs": {
                "sales_read.query.include_items": True,
                "sales_read.query.location_id": "loc_westlands",
            },
            "endpointArgProofRefs": {
                "sales_read.query.include_items": ["source_param:include_items"],
                "sales_read.query.location_id": ["known_input:location"],
            },
            "rowFilters": [],
        },
        sort_keys=True,
    )
    prior_relation_id = "prior_sales.relation.answer_1_rows"
    prior_artifact = build_fact_artifact(
        artifact_id="prior_sales",
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
                    "scopeFingerprint": (
                        f"{prior_scope_without_items}|{prior_scope_with_items}"
                    ),
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
        source_question="List salespeople and products sold at Westlands.",
        source_answer="Amina sold Lipstick.",
    )
    data_access = _DataAccessPort(
        {
            "sales_read": {
                "data": [
                    {
                        "sale_id": "sale_1",
                        "staff_name": "Amina",
                        "product_name": "Lipstick",
                        "items": [{"shade_name": "Ruby"}],
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
            relation_catalog_port=_CatalogPort(_same_scope_items_catalog()),
            data_access_port=data_access,
            planner_model_port=_SameScopeFieldPlannerPort(
                question_contract={
                    "kind": "question_contract",
                    "answer_requests_count": 1,
                    "question_inputs": [],
                    "answer_requests": [
                        {
                            "answer_fact": "shade names",
                            "answer_expression": {"family": "list_rows"},
                            "answer_subject": _answer_subject_payload("shade names"),
                            "answer_population": default_answer_population(
                                description="shade names",
                                subject_text="shade names",
                                instance_interpretation=RequestedFactAnswerSubject(
                                    subject_text="shade names"
                                ).instance_interpretation,
                            ).to_question_contract_dict(),
                            "answer_outputs": [
                                {"description": "shade names", "role": "ANSWER_VALUE"}
                            ],
                            "used_question_inputs": [],
                        }
                    ],
                },
                field_id="shade_name",
                conversation_resolution=lambda prompt: (
                    _conversation_resolution_payload_using_memories(
                        prompt,
                        contextualized_question="Show shade names for the prior products.",
                    )
                ),
            ),
        ),
    )

    assert result.status == "COMPLETED", result
    assert data_access.requests == [
        {
            "endpointName": "sales_read",
            "args": {
                "sales_read.query.include_items": True,
                "sales_read.query.location_id": "loc_westlands",
            },
        }
    ]
    assert result.rendered_fact is not None
    assert result.rendered_fact.rows == ({"answer_1": "Ruby"},)


def _ports(
    *,
    question_contract: dict[str, Any],
    fact_plan: dict[str, Any],
    responses: dict[str, Any],
    catalog: RelationCatalog | None = None,
    planner_port_cls: type[_PlannerPort] | None = None,
    conversation_resolution: Any = None,
    query_enrichment: dict[str, Any] | None = None,
    read_eligibility_retention_specs: (
        tuple[ReadEligibilityRetentionSpec, ...] | None
    ) = None,
) -> LookupRuntimePorts:
    planner_cls = planner_port_cls or _PlannerPort
    return LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog or _metric_catalog()),
        data_access_port=_DataAccessPort(responses),
        planner_model_port=planner_cls(
            question_contract=question_contract,
            fact_plan=fact_plan,
            conversation_resolution=conversation_resolution,
            query_enrichment=query_enrichment,
            read_eligibility_retention_specs=read_eligibility_retention_specs,
        ),
    )


def _question_contract(
    *,
    fact_description: str,
    answer_outputs: tuple[str, ...],
    answer_output_roles: tuple[str, ...] | None = None,
    subject_text: str | None = None,
    known_inputs: tuple[dict[str, Any], ...] = (),
    split_answer_outputs: bool = True,
) -> dict[str, Any]:
    output_descriptions = (
        list(answer_outputs) if split_answer_outputs else [", ".join(answer_outputs)]
    )
    output_roles = answer_output_roles or tuple(
        "ANSWER_VALUE" for _ in output_descriptions
    )
    if len(output_roles) != len(output_descriptions):
        raise ValueError("answer output roles must match answer outputs")
    question_inputs = [
        {
            "input_ref": f"input_{index}",
            "source": "question_context",
            "kind": item["kind"],
            **(
                {"reference_text": item["text"]}
                if item["kind"] == "row_set_reference"
                else {
                    "source_text": item["text"],
                    "role": item["role"],
                    "resolved_value_text": item["resolved_value_text"],
                }
            ),
            "inventory_check": {
                "why_this_is_an_input": f"{item['text']} is a declared question input"
            },
            **(
                {"value_meaning_hint": item["value_meaning_hint"]}
                if item.get("value_meaning_hint")
                else {}
            ),
            **(
                {"resolved_input_ref": item["resolved_input_ref"]}
                if item.get("resolved_input_ref")
                else {}
            ),
        }
        for index, item in enumerate(known_inputs, start=1)
    ]
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": question_inputs,
        "answer_requests": [
            {
                "answer_fact": fact_description,
                "answer_subject": _answer_subject_payload(
                    subject_text or fact_description
                ),
                "answer_population": _answer_population_payload(
                    description=fact_description,
                    subject_text=subject_text or fact_description,
                ),
                "answer_outputs": [
                    {"description": description, "role": role}
                    for description, role in zip(
                        output_descriptions,
                        output_roles,
                        strict=True,
                    )
                ],
                "used_question_inputs": [
                    f"input_{index}"
                    for index, _item in enumerate(known_inputs, start=1)
                ],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }


def _question_contract_for_fact_plan(
    payload: dict[str, Any],
    *,
    fact_plan: dict[str, Any],
) -> dict[str, Any]:
    family_by_fact_id = _answer_expression_family_by_fact_id(fact_plan)
    if not family_by_fact_id:
        return payload
    updated = json.loads(json.dumps(payload))
    for index, answer_request in enumerate(
        updated.get("answer_requests") or (), start=1
    ):
        if not isinstance(answer_request, dict):
            continue
        requested_fact_id = str(
            answer_request.get("requested_fact_id") or f"fact_{index}"
        )
        family = family_by_fact_id.get(requested_fact_id)
        if family:
            expression: dict[str, Any] = {"family": family}
            answer_request["answer_expression"] = expression
    return updated


def _answer_expression_family_by_fact_id(
    fact_plan: dict[str, Any],
) -> dict[str, str]:
    outcome = fact_plan.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("kind") != "fact_plan":
        return {}
    output: dict[str, str] = {}
    for answer in outcome.get("answers") or ():
        if not isinstance(answer, dict):
            continue
        requested_fact_id = str(answer.get("requested_fact_id") or "")
        pattern = str(answer.get("pattern") or "")
        family = _answer_expression_family_for_pattern(pattern)
        if requested_fact_id and family:
            output[requested_fact_id] = family
    return output


def _answer_expression_family_for_pattern(pattern: str) -> str:
    if pattern in {"list_rows", "grouped_rows", "joined_rows"}:
        return "list_rows"
    if pattern == "direct_field_value":
        return "scalar_value"
    if pattern == "aggregate_scalar":
        return "scalar_aggregate"
    if pattern == "aggregate_by_group":
        return "grouped_aggregate"
    if pattern == "ranked_aggregate":
        return "ranked_selection"
    if pattern == "computed_scalar":
        return "computed_scalar"
    if pattern == "set_difference":
        return "set_difference"
    return "scalar_aggregate"


def _answer_subject_payload(subject: str) -> dict[str, object]:
    return {
        "subject_text": subject,
        "instance_interpretation": {"kind": "NORMAL_BUSINESS_INSTANCE"},
    }


def _answer_population_payload(
    *,
    description: str,
    subject_text: str,
) -> dict[str, object]:
    return default_answer_population(
        description=description,
        subject_text=subject_text,
        instance_interpretation=RequestedFactAnswerSubject(
            subject_text=subject_text,
        ).instance_interpretation,
    ).to_question_contract_dict()


def _question_contract_with_prompt_memory(
    payload: dict[str, Any],
    prompt: str,
    *,
    fact_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = json.loads(
        json.dumps(_question_contract_for_fact_plan(payload, fact_plan=fact_plan or {}))
    )
    updated.setdefault(
        "question_input_inventory_check",
        {"all_input_like_phrases_declared": True},
    )
    question_inputs = updated.get("question_inputs")
    if isinstance(question_inputs, list):
        for item in question_inputs:
            if isinstance(item, dict):
                item.setdefault(
                    "inventory_check",
                    {
                        "why_this_is_an_input": (
                            f"{item.get('reference_text') or 'input'} is a declared question input"
                        )
                    },
                )
    answer_requests = updated.get("answer_requests")
    if not isinstance(answer_requests, list):
        return updated
    updated.setdefault("answer_requests_count", len(answer_requests))
    updated["answer_requests"] = [
        _question_contract_answer_request_with_prompt_memory(item, prompt=prompt)
        for item in answer_requests
    ]
    return updated


def _question_contract_answer_request_with_prompt_memory(
    item: Any,
    *,
    prompt: str,
) -> Any:
    if not isinstance(item, dict):
        return item
    updated = dict(item)
    if "answer_subject" in updated:
        updated["answer_subject"] = (
            _question_contract_answer_subject_with_prompt_memory(
                updated.get("answer_subject"),
            )
        )
    updated.setdefault("answer_expression", {"family": "scalar_aggregate"})
    if "answer_population" not in updated:
        subject_text = (
            str(updated["answer_subject"].get("subject_text") or "")
            if isinstance(updated.get("answer_subject"), dict)
            else str(updated.get("answer_fact") or "")
        )
        updated["answer_population"] = _answer_population_payload(
            description=str(updated.get("answer_fact") or subject_text),
            subject_text=subject_text,
        )
    return updated


def _question_contract_answer_subject_with_prompt_memory(
    raw: Any,
) -> Any:
    if isinstance(raw, dict):
        subject_text = str(raw.get("subject_text") or "").strip()
        if subject_text:
            instance = raw.get("instance_interpretation")
            kind = (
                str(instance.get("kind") or "").strip()
                if isinstance(instance, dict)
                else "NORMAL_BUSINESS_INSTANCE"
            )
            return {
                "subject_text": subject_text,
                "instance_interpretation": {"kind": kind},
            }
    return raw


def _current_question_from_prompt(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _activated_memory_ids_from_prompt(prompt: str) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'"memory_id":\s*"([^"]+)"', prompt):
        memory_id = match.group(1)
        if memory_id in seen:
            continue
        seen.add(memory_id)
        ids.append(memory_id)
    return tuple(ids)


def _fact_plan_answer(**answer: Any) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": "fact_1",
                    **answer,
                }
            ],
        }
    }


def _metric_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="metric_read",
                endpoint_name="metric_read",
                resource_names=("metric",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="uuid",
                    ),
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
                candidate_keys=_primary_key(
                    "location", "location_id", "field.location_id"
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _single_metric_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="metric_read",
                endpoint_name="metric_read",
                resource_names=("metric",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.ONE),
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
            ),
        ),
    )


def _precomputed_count_catalog(
    *,
    row_cardinality: RowCardinality = RowCardinality.ONE,
) -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="summary_count_read",
                endpoint_name="summary_count_read",
                resource_names=("summary count",),
                row_paths=(
                    RowPath(
                        id="summary",
                        path="summary",
                        cardinality=row_cardinality,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.summary.total_count",
                        path="summary.total_count",
                        row_path_id="summary",
                        type="integer",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _records_catalog_with_channel_param() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="records_read",
                endpoint_name="records_read",
                resource_names=("record",),
                params=(
                    CatalogParam(
                        ref="records_read.query.channel",
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
                        ref="records_read.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.record_id",
                        path="data.record_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                ),
                candidate_keys=_primary_key("record", "record_id", "field.record_id"),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _records_catalog_with_optional_params() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="records_read",
                endpoint_name="records_read",
                resource_names=("record",),
                params=(
                    CatalogParam(
                        ref="records_read.query.channel",
                        name="channel",
                        source=ParamSource.QUERY,
                        type="choice",
                        choices=("STORE", "ONLINE"),
                    ),
                    CatalogParam(
                        ref="records_read.query.status",
                        name="status",
                        source=ParamSource.QUERY,
                        type="choice",
                        choices=("PLACED", "COMPLETED"),
                    ),
                    CatalogParam(
                        ref="records_read.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.record_id",
                        path="data.record_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                ),
                candidate_keys=_primary_key("record", "record_id", "field.record_id"),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _same_scope_items_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
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
                        ref="sales_read.query.include_items",
                        name="include_items",
                        source=ParamSource.QUERY,
                        type="boolean",
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                    RowPath(
                        id="items",
                        path="data.items",
                        cardinality=RowCardinality.MANY,
                        parent_path="data",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.sale_id",
                        path="data.sale_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.product_name",
                        path="data.product_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.shade_name",
                        path="data.items.shade_name",
                        row_path_id="items",
                        type="string",
                        requirements=(
                            FieldRequirement(
                                param_ref="sales_read.query.include_items",
                                value="true",
                            ),
                        ),
                    ),
                ),
                candidate_keys=_primary_key("sale", "sale_id", "field.sale_id"),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _sales_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="sales_read",
                endpoint_name="sales_read",
                resource_names=("sale",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.sale_id",
                        path="data.sale_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.snapshot_merch_name",
                        path="data.snapshot_merch_name",
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
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _sales_product_shade_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="sales_read",
                endpoint_name="sales_read",
                resource_names=("sale",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="choice",
                    ),
                    CatalogField(
                        ref="field.snapshot_merch_name",
                        path="data.snapshot_merch_name",
                        row_path_id="data",
                        type="choice",
                    ),
                    CatalogField(
                        ref="field.snapshot_shade_name",
                        path="data.snapshot_shade_name",
                        row_path_id="data",
                        type="choice",
                    ),
                    CatalogField(
                        ref="field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        ),
    )


def _variant_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            _variant_read("candidate_read", include_name=True),
            _variant_read("observed_read", include_name=False),
        )
    )


def _variant_read(read_id: str, *, include_name: bool) -> EndpointRead:
    fields = [
        CatalogField(
            ref=f"field.{read_id}.variant_id",
            path="data.variant_id",
            row_path_id="data",
            type="string",
        )
    ]
    if include_name:
        fields.append(
            CatalogField(
                ref=f"field.{read_id}.variant_name",
                path="data.variant_name",
                row_path_id="data",
                type="string",
            )
        )
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        resource_names=("variant",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=tuple(fields),
        candidate_keys=_primary_key(
            "variant", "variant_id", f"field.{read_id}.variant_id"
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _join_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="items_read",
                endpoint_name="items_read",
                resource_names=("item",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.items.item_id",
                        path="data.item_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.items.quantity",
                        path="data.quantity",
                        row_path_id="data",
                        type="number",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
            EndpointRead(
                id="products_read",
                endpoint_name="products_read",
                resource_names=("product",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.products.item_id",
                        path="data.item_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.products.item_name",
                        path="data.item_name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        )
    )


def _join_catalog_with_shared_context_field() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="items_read",
                endpoint_name="items_read",
                resource_names=("item",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.items.item_id",
                        path="data.item_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.items.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.items.quantity",
                        path="data.quantity",
                        row_path_id="data",
                        type="number",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
            EndpointRead(
                id="products_read",
                endpoint_name="products_read",
                resource_names=("product",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.products.item_id",
                        path="data.item_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.products.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="field.products.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                pagination=PaginationMetadata(
                    mode=PaginationMode.NONE,
                    completeness_policy=CompletenessPolicy.COMPLETE,
                ),
            ),
        )
    )


@dataclass
class _CatalogPort:
    catalog: RelationCatalog

    def build_relation_catalog(self) -> RelationCatalog:
        return self.catalog


@dataclass
class _DataAccessPort:
    responses: dict[str, Any]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": self.responses[endpoint_name],
            "truncated": False,
            "pageCount": 1,
        }


@dataclass
class _PlannerPort:
    question_contract: dict[str, Any]
    fact_plan: dict[str, Any]
    conversation_resolution: Any = None
    query_enrichment: dict[str, Any] | None = None
    read_eligibility_retention_specs: (
        tuple[ReadEligibilityRetentionSpec, ...] | None
    ) = None
    source_binding_selection_prompt: str = ""
    source_binding_payload: dict[str, Any] = field(default_factory=dict)

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
        if any(tool.name in CONVERSATION_RESOLUTION_TOOL_NAMES for tool in tool_specs):
            arguments = _conversation_resolution_payload_from_response(
                prompt,
                self.conversation_resolution,
            )
            return _tool_response(
                _conversation_resolution_tool_name_for_payload(arguments),
                arguments,
            )
        else:
            tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_with_prompt_memory(
                self.question_contract,
                prompt,
                fact_plan=self.fact_plan,
            )
        elif tool_name == "submit_query_enrichment":
            arguments = self.query_enrichment or _query_enrichment_payload_from_prompt(
                prompt
            )
        elif tool_name == "submit_grounding":
            arguments = _grounding_payload_from_prompt(prompt)
        elif tool_name == "submit_read_eligibility":
            if self.read_eligibility_retention_specs is not None:
                return read_eligibility_response_from_prompt(
                    prompt,
                    retention_specs=self.read_eligibility_retention_specs,
                )
            return read_eligibility_response_from_fact_plan(prompt, self.fact_plan)
        elif tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            arguments = source_binding_payload_from_fact_plan(
                self.fact_plan,
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                self.fact_plan,
                prompt=prompt,
            )
        elif tool_name == "submit_pattern_fact_plan":
            arguments = bound_fact_plan_payload_from_fact_plan(
                self.fact_plan,
                prompt=prompt,
                provider_schema=tool_specs[0].input_schema if tool_specs else None,
            )
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(arguments)
        return {
            "answer": json.dumps({"tool": tool_name, "arguments": arguments}),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


@dataclass
class _OpaqueSourceHandlePlannerPort(_PlannerPort):
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
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_from_prompt(prompt)
            return _tool_response(
                _conversation_resolution_tool_name_for_payload(arguments), arguments
            )
        if tool_name == "submit_question_contract_outcome":
            return _tool_response(
                tool_name,
                _question_contract_with_prompt_memory(
                    self.question_contract,
                    prompt,
                    fact_plan=self.fact_plan,
                ),
            )
        if tool_name == "submit_query_enrichment":
            return _tool_response(
                tool_name,
                _query_enrichment_payload(("metric",)),
            )
        if tool_name == "submit_read_eligibility":
            return read_eligibility_response_from_fact_plan(prompt, self.fact_plan)
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            candidate = source_candidate_with_fields(
                prompt,
                required=("location_name", "metric_total"),
                forbidden=(),
            )
            candidate_id = str(candidate["source_candidate_id"])
            assert candidate_id.startswith("source_")
            binding_target_id = source_binding_target_id_for_candidate(
                prompt,
                requested_fact_id="fact_1",
                source_candidate_id=candidate_id,
                plan_shape="list_rows",
            )
            self.source_binding_payload = {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                binding_target_id=binding_target_id,
                            ),
                            "fulfillment_decisions": (
                                source_fulfills_fields_for_candidate(
                                    candidate,
                                    field_ids_by_answer_output={
                                        "answer_1": ("location_name",),
                                        "answer_2": ("metric_total",),
                                    },
                                )
                            ),
                            "param_decisions": {},
                        },
                    },
                }
            }
            return _tool_response(
                tool_name,
                source_binding_payload_for_one_call(
                    self.source_binding_payload,
                    prompt=prompt,
                ),
            )
        if tool_name == "submit_source_alignment_reviews":
            return _tool_response(
                tool_name,
                plan_selection_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                ),
            )
        if tool_name == "submit_pattern_fact_plan":
            return _tool_response(
                tool_name,
                bound_fact_plan_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                    provider_schema=tool_specs[0].input_schema if tool_specs else None,
                ),
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


@dataclass
class _AllSalesAnswerEvidencePlannerPort(_PlannerPort):
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
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_from_prompt(prompt)
            return _tool_response(
                _conversation_resolution_tool_name_for_payload(arguments), arguments
            )
        if tool_name == "submit_question_contract_outcome":
            return _tool_response(
                tool_name,
                _question_contract_with_prompt_memory(
                    self.question_contract,
                    prompt,
                    fact_plan=self.fact_plan,
                ),
            )
        if tool_name == "submit_query_enrichment":
            payload = _query_enrichment_payload_from_prompt(prompt)
            return _tool_response(tool_name, payload)
        if tool_name == "submit_read_eligibility":
            return read_eligibility_response_from_prompt(
                prompt,
                retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("staff_name",),
                        group_key_fields=("staff_name",),
                    ),
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("snapshot_merch_name",),
                        group_key_fields=("snapshot_merch_name",),
                    ),
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("snapshot_shade_name",),
                        group_key_fields=("snapshot_shade_name",),
                    ),
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        measured_value_fields=("amount",),
                    ),
                ),
            )
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            self.source_binding_payload = source_binding_payload_from_fact_plan(
                self.fact_plan,
                prompt=prompt,
            )
            return _source_binding_response(
                tool_name,
                self.source_binding_payload,
                prompt=prompt,
            )
        if tool_name == "submit_source_alignment_reviews":
            return _tool_response(
                tool_name,
                plan_selection_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                ),
            )
        if tool_name == "submit_pattern_fact_plan":
            return _tool_response(
                tool_name,
                bound_fact_plan_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                    provider_schema=tool_specs[0].input_schema if tool_specs else None,
                ),
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


@dataclass
class _TwoAnswerOutputPlannerPort(_PlannerPort):
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
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_from_prompt(prompt)
            return _tool_response(
                _conversation_resolution_tool_name_for_payload(arguments), arguments
            )
        if tool_name == "submit_question_contract_outcome":
            return _tool_response(
                tool_name,
                _question_contract_with_prompt_memory(
                    self.question_contract,
                    prompt,
                    fact_plan=self.fact_plan,
                ),
            )
        if tool_name == "submit_query_enrichment":
            return _tool_response(
                tool_name,
                _query_enrichment_payload(("sale",)),
            )
        if tool_name == "submit_read_eligibility":
            return read_eligibility_response_from_prompt(
                prompt,
                retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("snapshot_merch_name",),
                    ),
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("snapshot_shade_name",),
                    ),
                ),
            )
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            candidate = source_candidate_with_fields(
                prompt,
                required=("snapshot_merch_name", "snapshot_shade_name"),
                forbidden=(),
            )
            candidate_id = str(candidate["source_candidate_id"])
            binding_target_id = source_binding_target_id_for_candidate(
                prompt,
                requested_fact_id="fact_1",
                source_candidate_id=candidate_id,
                plan_shape="list_rows",
            )
            self.source_binding_payload = {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                binding_target_id=binding_target_id,
                            ),
                            "fulfillment_decisions": {
                                **source_fulfills_for_candidate(
                                    candidate,
                                    field_ids=("snapshot_merch_name",),
                                    answer_output_ids=("answer_1",),
                                ),
                                **source_fulfills_for_candidate(
                                    candidate,
                                    field_ids=("snapshot_shade_name",),
                                    answer_output_ids=("answer_2",),
                                ),
                            },
                            "param_decisions": {},
                        },
                    },
                }
            }
            self.source_binding_payload = source_binding_payload_for_one_call(
                self.source_binding_payload,
                prompt=prompt,
            )
            return _source_binding_response(
                tool_name,
                self.source_binding_payload,
                prompt=prompt,
            )
        if tool_name == "submit_source_alignment_reviews":
            return _tool_response(
                tool_name,
                plan_selection_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                ),
            )
        if tool_name == "submit_pattern_fact_plan":
            return _tool_response(
                tool_name,
                bound_fact_plan_payload_from_fact_plan(
                    self.fact_plan,
                    prompt=prompt,
                    provider_schema=tool_specs[0].input_schema if tool_specs else None,
                ),
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


@dataclass
class _FactPlanPromptInspectingPlannerPort(_PlannerPort):
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
        if tool_name == "submit_pattern_fact_plan":
            assert "Relation catalog:" not in prompt
            assert "source.param_bindings" not in prompt
            assert "Bound sources:" in prompt
        return super().generate(
            provider=provider,
            prompt=prompt,
            max_thinking_tokens=max_thinking_tokens,
            system_prompt=system_prompt,
            output_mode=output_mode,
            tool_specs=tool_specs,
        )


@dataclass
class _CanonicalSourceBindingPlannerPort(_PlannerPort):
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
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            schema_text = json.dumps(tool_specs[0].input_schema, sort_keys=True)
            assert "source_param_decision_ids" not in schema_text
            assert "param_decision_ids" not in prompt
        return super().generate(
            provider=provider,
            prompt=prompt,
            max_thinking_tokens=max_thinking_tokens,
            system_prompt=system_prompt,
            output_mode=output_mode,
            tool_specs=tool_specs,
        )


@dataclass
class _SameScopeFieldPlannerPort:
    question_contract: dict[str, Any]
    field_id: str
    conversation_resolution: Any = None
    source_binding_payload: dict[str, Any] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)

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
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_from_response(
                prompt,
                self.conversation_resolution,
            )
            return _tool_response(
                _conversation_resolution_tool_name_for_payload(arguments), arguments
            )
        if tool_name == "submit_question_contract_outcome":
            return _tool_response(
                tool_name,
                _question_contract_with_prompt_memory(self.question_contract, prompt),
            )
        if tool_name == "submit_query_enrichment":
            return _tool_response(
                tool_name,
                _query_enrichment_payload(("sale",)),
            )
        if tool_name == "submit_read_eligibility":
            return read_eligibility_response_from_prompt(
                prompt,
                retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        row_path_ids=("items",),
                        answer_value_fields=(self.field_id,),
                    ),
                ),
            )
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            try:
                candidate = source_candidate_with_fields(
                    prompt,
                    kind="same_scope_api_read",
                    required=(self.field_id,),
                    forbidden=(),
                )
            except AssertionError:
                candidate = source_candidate_with_kind(
                    prompt,
                    kind="same_scope_api_read",
                )
            candidate_id = str(candidate["source_candidate_id"])
            param_decisions = {}
            if _source_candidate_param(candidate, "include_items") is not None:
                param_decisions["include_items"] = {
                    "population_intent": "item rows",
                    "match_basis_explanation": "Item rows require include_items true.",
                    "param_decision_id": _param_decision_id_for(
                        candidate,
                        param_id="include_items",
                        decision="bind",
                        value="true",
                    ),
                }
            binding_target_id = source_binding_target_id_for_candidate(
                prompt,
                requested_fact_id="fact_1",
                source_candidate_id=candidate_id,
                plan_shape="list_rows",
            )
            self.source_binding_payload = {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                binding_target_id=binding_target_id,
                            ),
                            "fulfillment_decisions": _same_scope_fulfillment_decisions(
                                candidate,
                                field_id=self.field_id,
                            ),
                            "param_decisions": param_decisions,
                        },
                    },
                }
            }
            return _source_binding_response(
                tool_name,
                self.source_binding_payload,
                prompt=prompt,
            )
        if tool_name == "submit_source_alignment_reviews":
            return _tool_response(
                tool_name,
                _same_scope_plan_selection_payload(prompt, field_id=self.field_id),
            )
        if tool_name == "submit_pattern_fact_plan":
            return _tool_response(
                tool_name,
                _fact_plan_answer(
                    pattern="list_rows",
                    source_binding_id="sb_1",
                    answer_output_ids=("answer_1",),
                    output_fields=({"field_id": self.field_id},),
                ),
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


def _tool_response(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "submit_question_contract_outcome":
        arguments = _question_contract_decision(arguments)
    return {
        "answer": json.dumps({"tool": tool_name, "arguments": arguments}),
        "usage": {
            "inputTokens": 1,
            "outputTokens": 1,
            "thinkingTokens": 0,
            "costUsd": 0,
        },
    }


def _source_binding_response(
    tool_name: str,
    source_binding_payload: dict[str, Any],
    *,
    prompt: str,
) -> dict[str, Any]:
    return _tool_response(
        tool_name,
        source_binding_payload_for_one_call(
            source_binding_payload,
            prompt=prompt,
        ),
    )


def _prompt_json_section(prompt: str, *, label: str) -> dict[str, Any]:
    return prompt_section_payload(prompt, label)


def _conversation_resolution_from_prompt(prompt: str) -> dict[str, Any]:
    question = _current_question_from_prompt(prompt)
    return {
        "kind": "conversation_resolution",
        "current_question_text": question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": "The current question is context-free.",
            "contextualized_question": question,
            "clauses": [
                {
                    "current_clause_text": question,
                    "occurrence": 1,
                    "resolved_text": question,
                    "retained_frame_parts": [],
                    "values": [],
                }
            ],
        },
    }


def _conversation_resolution_payload_using_memories(
    prompt: str,
    *,
    contextualized_question: str,
) -> dict[str, Any]:
    question = _current_question_from_prompt(prompt)
    context_sources = (
        _prompt_json_section_optional(prompt, label="Context sources").get(
            "context_sources"
        )
        or []
    )
    selected_sources = tuple(item for item in context_sources if isinstance(item, dict))
    return _conversation_resolution_clause_payload(
        prompt=prompt,
        current_question=question,
        contextualized_question=contextualized_question,
        actual_text=question,
        selected_sources=selected_sources,
    )


def _prompt_json_section_optional(prompt: str, *, label: str) -> dict[str, Any]:
    marker = f"{label}:\n"
    if marker not in prompt:
        return {}
    return _prompt_json_section(prompt, label=label)


def _memory_kind_for_test_id(memory_id: str) -> str:
    if ".prior_request." in memory_id:
        return "prior_answer_request"
    if ".relation." in memory_id:
        return "row_set"
    if ".entity." in memory_id:
        return "entity_identity"
    if ".outcome." in memory_id:
        return "clarification_response"
    if ".value." in memory_id:
        return "scalar_value"
    return ""


def _conversation_resolution_payload_from_response(
    prompt: str,
    response: Any,
) -> dict[str, Any]:
    if response is None:
        return _conversation_resolution_from_prompt(prompt)
    if callable(response):
        payload = response(prompt)
        if not isinstance(payload, dict):
            raise AssertionError("conversation-resolution builder must return dict")
        return payload
    if isinstance(response, dict):
        return dict(response)
    raise AssertionError("conversation-resolution response must be a dict or callable")


def _current_question_from_prompt(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _source_binding_prompt(planner: Any) -> str:
    for prompt in getattr(planner, "prompts", ()):
        if "Candidate evidence sources:\n" in prompt:
            return prompt
    raise AssertionError("source binding prompt was not captured")


def _first_available_relation(payload: dict[str, Any], read_id: str) -> dict[str, Any]:
    for fact_relations in payload.get("requested_fact_relations") or ():
        for relation in fact_relations.get("available_relations") or ():
            if relation.get("read_id") == read_id:
                return relation
    raise AssertionError(f"prompt missing available relation: {read_id}")


def _first_source_candidate(
    payload: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    for candidate in _all_source_candidates(payload):
        if candidate.get("source_candidate_id") == candidate_id:
            return candidate
    raise AssertionError(f"prompt missing source candidate: {candidate_id}")


def _source_candidate_with_fields(
    payload: dict[str, Any],
    *,
    kind: str | None = None,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    for candidate in _all_source_candidates(payload):
        if kind is not None and candidate.get("kind") != kind:
            continue
        field_ids = {
            str(item.get("field_id") or item.get("id") or "")
            for field_source in (
                _candidate_binding_surface(candidate).get("evidence_items") or (),
                _candidate_binding_surface(candidate).get("fields") or (),
                candidate.get("fields") or (),
            )
            for item in field_source
            if isinstance(item, dict)
        }
        field_ids |= _candidate_fulfillment_field_ids(candidate)
        if set(required) <= field_ids and not (set(forbidden) & field_ids):
            return candidate
    raise AssertionError(
        f"prompt missing {kind or 'source'} candidate with fields {required} "
        f"and without fields {forbidden}"
    )


def _source_candidate_with_kind(
    payload: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    for candidate in _all_source_candidates(payload):
        if candidate.get("kind") == kind:
            return candidate
    raise AssertionError(f"prompt missing {kind} candidate")


def _same_scope_fulfillment_decisions(
    candidate: dict[str, Any],
    *,
    field_id: str,
) -> dict[str, dict[str, Any]]:
    return source_fulfills_for_candidate(candidate, field_ids=(field_id,))


def _first_candidate_fulfillment_choice_set(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    for support_set in (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if isinstance(support_set, dict) and str(
            support_set.get("fulfillment_choice_id") or ""
        ):
            return support_set
    raise AssertionError("candidate missing fulfillment support set")


def _same_scope_plan_selection_payload(prompt: str, *, field_id: str) -> dict[str, Any]:
    del field_id
    return plan_selection_payload_from_fact_plan(
        {
            "outcome": {
                "kind": "fact_plan",
                "answers": [
                    {
                        "requested_fact_id": "fact_1",
                        "pattern": "list_rows",
                        "source": {
                            "kind": "same_scope_api_read",
                            "read_id": "sales_read",
                        },
                    }
                ],
            }
        },
        prompt=prompt,
    )


def _plan_selection_source_strategy_with_member_field(
    group: dict[str, Any],
    *,
    plan_shape: str,
    kind: str,
    field_id: str,
) -> dict[str, Any]:
    for source_strategy in group.get("source_strategies") or ():
        if (
            not isinstance(source_strategy, dict)
            or source_strategy.get("plan_shape") != plan_shape
        ):
            continue
        if any(
            isinstance(member, dict)
            and member.get("kind") == kind
            and field_id
            in (
                {str(item) for item in member.get("field_ids") or ()}
                | {
                    str(field.get("field_id") or "")
                    for row in member.get("response_rows") or ()
                    if isinstance(row, dict)
                    for field in row.get("fields") or ()
                    if isinstance(field, dict)
                }
            )
            for member in source_strategy.get("source_members") or ()
        ):
            return source_strategy
    raise AssertionError(
        f"plan selection prompt missing {kind} source strategy with {field_id}"
    )


def _source_candidate_with_result_grain_fields(
    payload: dict[str, Any],
    *,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> dict[str, Any]:
    for candidate in _all_source_candidates(payload):
        for grain in _candidate_result_grains(candidate):
            if not isinstance(grain, dict):
                continue
            field_ids = {
                str(item.get("field_id") or item.get("id") or "")
                for item in grain.get("evidence_items") or ()
                if isinstance(item, dict)
            }
            if set(required) <= field_ids and not (set(forbidden) & field_ids):
                return candidate
    raise AssertionError(
        f"prompt missing source candidate result grain with fields {required} "
        f"and without fields {forbidden}"
    )


def _assert_no_empty_source_candidates(payload: dict[str, Any]) -> None:
    for fact_sources in payload.get("requested_fact_sources") or ():
        for candidate in _source_options_for_fact_sources(fact_sources):
            if candidate.get("kind") != "new_api_read":
                continue
            if (
                candidate.get("read_contract")
                or candidate.get("fields")
                or candidate.get("response_rows")
                or _candidate_binding_surface(candidate).get("evidence_items")
            ):
                continue
            raise AssertionError("source candidate without evidence was exposed")


def _source_evidence_id(
    prompt: str,
    *,
    source_candidate_id: str,
    field_id: str,
) -> str:
    payload = _prompt_json_section(prompt, label="Candidate evidence sources")
    candidate = _first_source_candidate(payload, source_candidate_id)
    for item in _candidate_binding_surface(candidate).get("evidence_items") or ():
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id.endswith(f".{field_id}") or evidence_id == field_id:
            return evidence_id
    for slot in _candidate_fulfillment_slots(candidate):
        if not isinstance(slot, dict):
            continue
        for key in _FULFILLMENT_EVIDENCE_KEYS:
            for item in slot.get(key) or ():
                if not isinstance(item, dict):
                    continue
                evidence_id = str(item.get("evidence_id") or "")
                candidate_field_id = str(item.get("field_id") or "")
                if (
                    candidate_field_id.endswith(f".{field_id}")
                    or candidate_field_id == field_id
                ):
                    return evidence_id
    for item in candidate.get("fields") or ():
        if not isinstance(item, dict):
            continue
        candidate_field_id = str(item.get("field_id") or item.get("id") or "")
        if (
            candidate_field_id.endswith(f".{field_id}")
            or candidate_field_id == field_id
        ):
            return candidate_field_id
    raise AssertionError(f"missing source evidence field {field_id}")


def _param_decision_id_for(
    candidate: dict[str, Any],
    *,
    param_id: str,
    decision: str,
    value: str = "",
) -> str:
    for param in _candidate_binding_surface(candidate).get("params") or ():
        if not isinstance(param, dict) or param.get("param_id") != param_id:
            continue
        for option in param.get("decision_options") or ():
            if not isinstance(option, dict):
                continue
            if option.get("decision") != decision:
                continue
            if value and option.get("value") != value:
                continue
            decision_id = str(option.get("param_decision_id") or "")
            if decision_id:
                return decision_id
        if value and value in set(param.get("choices") or ()):
            return ".".join(
                (
                    "param_decision",
                    str(candidate.get("source_candidate_id") or "source_1").lower(),
                    param_id.lower(),
                    decision.lower(),
                    value.lower(),
                )
            )
    raise AssertionError(f"missing param decision option for {param_id}={value}")


def _source_candidate_param(
    candidate: dict[str, Any],
    param_id: str,
) -> dict[str, Any] | None:
    for param in _candidate_binding_surface(candidate).get("params") or ():
        if isinstance(param, dict) and param.get("param_id") == param_id:
            return param
    return None


def _choice_label(value: str) -> str:
    return " ".join(part.capitalize() for part in value.lower().split("_"))


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


def _all_source_candidates(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    top_level_candidates = tuple(
        candidate
        for key in (
            "memory_source_candidates",
            "utility_source_candidates",
            "value_source_candidates",
        )
        for candidate in payload.get(key) or ()
        if isinstance(candidate, dict)
    )
    fact_candidates = tuple(
        candidate
        for fact_sources in payload.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
        for candidate in _source_options_for_fact_sources(fact_sources)
    )
    return (*top_level_candidates, *fact_candidates)


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


_FULFILLMENT_EVIDENCE_KEYS = (
    "metric_measure_evidence",
    "row_count_basis_evidence",
    "scope_evidence",
    "group_key_evidence",
)


def _candidate_fulfillment_slots(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        slot
        for support_set in _candidate_binding_surface(candidate).get(
            "fulfillment_support_sets"
        )
        or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
    )


def _candidate_fulfillment_field_ids(candidate: dict[str, Any]) -> set[str]:
    fields = {
        str(item.get("field_id") or "")
        for slot in _candidate_fulfillment_slots(candidate)
        for key in _FULFILLMENT_EVIDENCE_KEYS
        for item in slot.get(key) or ()
        if isinstance(item, dict)
    }
    for item in candidate.get("fields") or ():
        if isinstance(item, dict):
            fields.add(str(item.get("field_id") or ""))
    for item in _candidate_binding_surface(candidate).get("fields") or ():
        if isinstance(item, dict):
            fields.add(str(item.get("field_id") or ""))
    return fields


def _candidate_result_grains(candidate: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    result_grains = candidate.get("result_grains")
    if isinstance(result_grains, list):
        return tuple(item for item in result_grains if isinstance(item, dict))
    read_contract = candidate.get("read_contract")
    if not isinstance(read_contract, dict):
        return ()
    grains: list[dict[str, Any]] = []
    primary = read_contract.get("result_population")
    if isinstance(primary, dict):
        grains.append(primary)
    grains.extend(
        item
        for item in read_contract.get("nested_result_grains") or ()
        if isinstance(item, dict)
    )
    return tuple(grains)


def _answer_population_from_prompt(
    prompt: str,
    *,
    source_candidate_id: str = "source_1",
) -> dict[str, str]:
    intent_text = _current_question_text(prompt)
    return {
        "population_binding_id": _population_binding_id_from_prompt(
            prompt,
            source_candidate_id=source_candidate_id,
        ),
        "intent_text": intent_text,
        "match_basis_explanation": f"{intent_text} defines the source population",
    }


def _population_binding_id_from_prompt(
    prompt: str,
    *,
    source_candidate_id: str,
) -> str:
    payload = _prompt_json_section(prompt, label="Candidate evidence sources")
    candidate = _first_source_candidate(payload, source_candidate_id)
    for binding in (
        _candidate_binding_surface(candidate).get("population_bindings") or ()
    ):
        if not isinstance(binding, dict):
            continue
        binding_id = str(binding.get("population_binding_id") or "")
        if binding_id:
            return binding_id
    raise AssertionError("source_1 missing population binding")


def _current_question_text(prompt: str) -> str:
    marker = "Current question:\n"
    if marker not in prompt:
        return "sales"
    return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip() or "sales"
