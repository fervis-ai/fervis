from __future__ import annotations

import pytest

from tests.lookup.orchestrator._helpers import *  # noqa: F403

from fervis.memory.addresses import (
    EvidenceRef,
    FactAddress,
    RelationSourceKind,
)
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from fervis.model_io.backbone.registry import reset_registry_for_tests
from fervis.model_io.providers.bootstrap import (
    bootstrap_default_providers,
    reset_provider_bootstrap_for_tests,
)
from fervis.lookup.orchestration import pipeline as lookup_pipeline
from fervis.lookup.memory.projection import project_conversation_memory_cards
from fervis.lookup.turn_prompts.context import active_clarification_context


_PROVIDER_PREFS = {"provider": "anthropic", "modelKey": "FAKE"}


@pytest.fixture(autouse=True)
def _provider_registry():
    reset_provider_bootstrap_for_tests()
    reset_registry_for_tests()
    bootstrap_default_providers()
    yield
    reset_provider_bootstrap_for_tests()
    reset_registry_for_tests()


def _memory_attribution_response(
    *,
    question: str,
    conversation_context: dict[str, object],
    selected_memory_id: str,
    contextualized_question: str,
    source_containing: str = "",
    dependency_kind: str = "reference",
    retained_part_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    del dependency_kind
    projection = project_conversation_memory_cards(
        conversation_context,
        current_question=question,
    )
    selected = _selected_prior_source_for_memory_id(
        projection=projection,
        memory_id=selected_memory_id,
        containing=source_containing,
    )
    anchors = tuple(getattr(selected, "meaning_anchors", ()) or ())
    selected_anchors = tuple(
        anchor
        for anchor in anchors
        if str(getattr(anchor, "memory_id", "")) == selected_memory_id
    )
    values: list[dict[str, object]] = [
        {
            "value_id": f"context_value_{index}",
            "resolved_text": str(getattr(anchor, "text")),
            "sources": [
                {
                    "kind": "context_anchor",
                    "source_id": str(getattr(selected, "source_id")),
                    "memory_id": str(getattr(anchor, "memory_id")),
                    "source_text": str(getattr(anchor, "text")),
                }
            ],
        }
        for index, anchor in enumerate(selected_anchors, start=1)
    ]
    retained_frame_parts: list[dict[str, object]] = []
    for part_id in retained_part_ids:
        matches = [
            (frame.frame_id, part.part_id)
            for frame in projection.context_frames
            if str(getattr(selected, "source_id")) in frame.source_ids
            for part in frame.parts
            if part.part_id == part_id
        ]
        if len(matches) != 1:
            raise AssertionError(f"retained frame part is not unique: {part_id}")
        frame_id, matched_part_id = matches[0]
        retained_frame_parts.append(
            {
                "kind": "frame_part",
                "frame_id": frame_id,
                "part_id": matched_part_id,
            }
        )
    return {
        "kind": "conversation_resolution",
        "current_question_text": question,
        "outcome": {
            "kind": "resolved",
            "resolution_basis": (
                "The selected visible memory supplies the context omitted by the "
                "current clause."
            ),
            "contextualized_question": contextualized_question,
            "clauses": [
                {
                    "current_clause_text": question,
                    "occurrence": 1,
                    "resolved_text": contextualized_question,
                    "retained_frame_parts": retained_frame_parts,
                    "values": values,
                }
            ],
            "frame_call": {"kind": "none"},
        },
    }


def _standalone_attribution_response(
    *,
    question: str,
    conversation_context: dict[str, object],
) -> dict[str, object]:
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
            "frame_call": {"kind": "none"},
        },
    }


def _clause_resolution_conversation_response(
    *,
    question: str,
    conversation_context: dict[str, object],
    contextualized_question: str,
    prior_request_memory_id: str,
    entity_memory_id: str,
) -> dict[str, object]:
    payload = _memory_attribution_response(
        question=question,
        conversation_context=conversation_context,
        selected_memory_id=prior_request_memory_id,
        contextualized_question=contextualized_question,
        source_containing="total sales amount",
        retained_part_ids=("output:1",),
    )
    projection = project_conversation_memory_cards(
        conversation_context,
        current_question=question,
    )
    entity_selected = _selected_prior_source_for_memory_id(
        projection=projection,
        memory_id=entity_memory_id,
        containing="Alice Smith",
    )
    entity_anchor = next(
        anchor
        for anchor in entity_selected.meaning_anchors
        if anchor.memory_id == entity_memory_id
    )
    clauses = payload["outcome"]["clauses"]
    assert isinstance(clauses, list)
    values = clauses[0]["values"]
    assert isinstance(values, list)
    values.append(
        {
            "value_id": "entity_value",
            "resolved_text": "Alice Smith",
            "sources": [
                {
                    "kind": "context_anchor",
                    "source_id": entity_selected.source_id,
                    "memory_id": entity_anchor.memory_id,
                    "source_text": entity_anchor.text,
                }
            ],
        }
    )
    return payload


def _clause_resolution_response(
    *,
    question: str,
    conversation_context: dict[str, object],
    contextualized_question: str,
    prior_request_memory_id: str,
) -> dict[str, object]:
    return _clause_resolution_conversation_response(
        question=question,
        conversation_context=conversation_context,
        contextualized_question=contextualized_question,
        prior_request_memory_id=prior_request_memory_id,
        entity_memory_id="turn_staff_sales.entity.grounded_fact_1_entity_1",
    )


def _selected_prior_source_for_memory_id(
    *,
    projection: object,
    memory_id: str,
    containing: str = "",
) -> object:
    sources = tuple(getattr(projection, "context_sources", ()) or ())
    if containing:
        for source in sources:
            if memory_id in tuple(
                getattr(source, "source_memory_ids", ()) or ()
            ) and containing in str(getattr(source, "text") or ""):
                return source
    for source in sources:
        if memory_id in tuple(getattr(source, "source_memory_ids", ()) or ()):
            return source
    if len(sources) == 1:
        return sources[0]
    raise AssertionError(f"selected memory source not found: {memory_id}")
def _memory_context() -> dict[str, object]:
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        source_question="How much money did we make yesterday?",
        addresses=(
            FactAddress.relation(
                address="relation.sales_rows",
                source={"kind": RelationSourceKind.API_READ.value},
                grain_keys=("sale_key",),
                completeness={"status": "complete", "pagination": "all_pages"},
                row_addresses=("row.sale_1",),
            ),
            FactAddress.row(
                address="row.sale_1",
                relation="relation.sales_rows",
                identity={"sale_key": "sale-1"},
                values={"amount": {"type": "decimal", "value": "100.00"}},
            ),
        ),
    )
    return {"factArtifacts": [artifact.to_dict()]}


def _memory_context_with_prior_sales_and_location_identity(
    *,
    include_relation_identity_type: bool = True,
) -> dict[str, object]:
    prior_sales_question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales from yesterday",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1",
                    ),
                ),
            ),
        )
    )
    prior_sales = build_fact_artifact(
        artifact_id="turn_sales",
        outcome=FactOutcome.ANSWERED,
        source_question="How much money did we make yesterday?",
        source_answer="sales row sale-1 had amount 10.00.",
        provenance={"question_contract": prior_sales_question_contract.to_model_dict()},
        addresses=(
            FactAddress.relation(
                address="relation.sales",
                source={
                    "kind": RelationSourceKind.API_READ.value,
                    **(
                        {"identityType": "sale"}
                        if include_relation_identity_type
                        else {}
                    ),
                },
                grain_keys=("sale_id",),
                field_coverage={"sale_amount": "sales.sale_amount"},
                completeness={
                    "status": "complete",
                    "pagination": "all_pages",
                    "rowCount": 1,
                    "scopeFingerprint": json.dumps(
                        {
                            "endpointArgs": {
                                "sales_read.query.location_id": "loc_selected"
                            },
                            "endpointArgProofRefs": {
                                "sales_read.query.location_id": ["known_input:location"]
                            },
                        }
                    ),
                },
                row_addresses=("row.sales.1",),
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
            FactAddress.row(
                address="row.sales.1",
                relation="relation.sales",
                identity={"sale_id": "sale-1"},
                values={"sale_amount": {"type": "decimal", "value": "10.00"}},
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
        ),
    )
    location_identity = build_fact_artifact(
        artifact_id="turn_location",
        outcome=FactOutcome.ANSWERED,
        source_question="Which location is ABC Mall?",
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                reference_text="ABC Mall",
                identity={"location_id": "loc_stale_memory"},
            ),
        ),
    )
    return {"factArtifacts": [prior_sales.to_dict(), location_identity.to_dict()]}


def _memory_context_with_selected_and_unselected_memory() -> dict[str, object]:
    selected_sales = build_fact_artifact(
        artifact_id="turn_selected_sales",
        outcome=FactOutcome.ANSWERED,
        source_question="Which sales happened yesterday?",
        addresses=(
            FactAddress.relation(
                address="relation.sales",
                source={
                    "kind": RelationSourceKind.API_READ.value,
                    "identityType": "sale",
                },
                grain_keys=("sale_id",),
                field_coverage={"sale_amount": "sales.sale_amount"},
                completeness={
                    "status": "complete",
                    "pagination": "all_pages",
                    "rowCount": 1,
                    "scopeFingerprint": json.dumps(
                        {
                            "endpointArgs": {
                                "sales_read.query.location_id": "loc_selected"
                            },
                            "endpointArgProofRefs": {
                                "sales_read.query.location_id": ["known_input:location"]
                            },
                        }
                    ),
                },
                row_addresses=("row.sales.1",),
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
            FactAddress.row(
                address="row.sales.1",
                relation="relation.sales",
                identity={"sale_id": "sale-1"},
                values={"sale_amount": {"type": "decimal", "value": "10.00"}},
                evidence=EvidenceRef(step_ids=("read:sales_read",)),
            ),
        ),
    )
    unselected_sales = build_fact_artifact(
        artifact_id="turn_unselected_sales",
        outcome=FactOutcome.ANSWERED,
        source_question="How much money did Secret Outlet make?",
        addresses=(
            FactAddress.relation(
                address="relation.secret_sales",
                source={
                    "kind": RelationSourceKind.API_READ.value,
                    "identityType": "sale",
                },
                grain_keys=("sale_id",),
                field_coverage={"secret_amount": "secret_sales.secret_amount"},
                completeness={
                    "status": "complete",
                    "pagination": "all_pages",
                    "rowCount": 1,
                    "scopeFingerprint": json.dumps(
                        {
                            "endpointArgs": {
                                "secret_sales_read.query.location_id": "loc_secret"
                            },
                            "endpointArgProofRefs": {
                                "secret_sales_read.query.location_id": [
                                    "known_input:location"
                                ]
                            },
                        }
                    ),
                },
                row_addresses=("row.secret_sales.1",),
                evidence=EvidenceRef(step_ids=("read:secret_sales_read",)),
            ),
            FactAddress.row(
                address="row.secret_sales.1",
                relation="relation.secret_sales",
                identity={"sale_id": "secret-sale-1"},
                values={"secret_amount": {"type": "decimal", "value": "999.00"}},
                evidence=EvidenceRef(step_ids=("read:secret_sales_read",)),
            ),
        ),
    )
    return {
        "factArtifacts": [
            selected_sales.to_dict(),
            unselected_sales.to_dict(),
        ]
    }


def _memory_context_with_selected_and_unselected_scalar_values() -> dict[str, object]:
    selected_threshold = build_fact_artifact(
        artifact_id="turn_selected_threshold",
        outcome=FactOutcome.ANSWERED,
        source_question="What amount threshold should I use?",
        source_answer="Use 50.",
        addresses=(
            FactAddress.value(
                address="value.threshold",
                value={"type": "number", "value": "50"},
                display="50",
                evidence=EvidenceRef(step_ids=("selected_threshold",)),
            ),
        ),
    )
    unselected_threshold = build_fact_artifact(
        artifact_id="turn_unselected_threshold",
        outcome=FactOutcome.ANSWERED,
        source_question="What old amount threshold should I ignore?",
        source_answer="Use 999.",
        addresses=(
            FactAddress.value(
                address="value.threshold",
                value={"type": "number", "value": "999"},
                display="999",
                evidence=EvidenceRef(step_ids=("unselected_threshold",)),
            ),
        ),
    )
    return {
        "factArtifacts": [
            selected_threshold.to_dict(),
            unselected_threshold.to_dict(),
        ]
    }


def _memory_context_with_staff_identities() -> dict[str, object]:
    alice = build_fact_artifact(
        artifact_id="turn_alice",
        outcome=FactOutcome.ANSWERED,
        source_question="Who is Alice Smith?",
        addresses=(
            FactAddress.entity(
                address="entity.staff.alice",
                resource="staff",
                reference_text="Alice Smith",
                identity={"staff_id": "staff_alice"},
            ),
        ),
    )
    jane = build_fact_artifact(
        artifact_id="turn_jane",
        outcome=FactOutcome.ANSWERED,
        source_question="Who is Jane Doe?",
        addresses=(
            FactAddress.entity(
                address="entity.staff.jane",
                resource="staff",
                reference_text="Jane Doe",
                identity={"staff_id": "staff_jane"},
            ),
        ),
    )
    return {"factArtifacts": [alice.to_dict(), jane.to_dict()]}


def _memory_context_with_prior_staff_sales_request() -> dict[str, object]:
    artifact = build_fact_artifact(
        artifact_id="turn_staff_sales",
        outcome=FactOutcome.ANSWERED,
        source_question="How much did Alice Smith sell today?",
        source_answer="Alice Smith sold 100 today.",
        provenance={
            "question_contract": {
                "question_inputs": [
                    {
                        "id": "fact_1_entity_1",
                        "kind": "literal_text",
                        "source": "question_context",
                        "text": "Alice Smith",
                        "role": "reference_value",
                        "resolved_value_text": "Alice Smith",
                        "value_meaning_hint": "staff member",
                    },
                    {
                        "id": "fact_1_time_1",
                        "kind": "literal_text",
                        "source": "question_context",
                        "text": "today",
                        "role": "time_value",
                        "resolved_value_text": "today",
                    },
                ],
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "sales by Alice Smith today",
                        "answer_expression": {"family": "scalar_aggregate"},
                        "answer_subject": _answer_subject_payload("Alice Smith"),
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "total sales amount",
                            }
                        ],
                        "used_question_inputs": [
                            "fact_1_entity_1",
                            "fact_1_time_1",
                        ],
                    }
                ]
            }
        },
        addresses=(
            FactAddress.entity(
                address="entity.grounded_fact_1_entity_1",
                resource="staff",
                reference_text="Alice Smith",
                identity={"staff_id": "staff_alice"},
                evidence=EvidenceRef(
                    step_ids=(
                        "known_input:fact_1_entity_1",
                        "source_read:staff_list_read:row_1",
                    )
                ),
            ),
            FactAddress.value(
                address="value.answer_output_1",
                value={"type": "decimal", "value": "100.00"},
                display="100.00",
                derivation={"answer_output_ids": ["answer_output_1"]},
            ),
        ),
    )
    return {"factArtifacts": [artifact.to_dict()]}


def _memory_context_with_prior_numeric_slots() -> dict[str, object]:
    artifact = build_fact_artifact(
        artifact_id="turn_ranked_sales",
        outcome=FactOutcome.ANSWERED,
        source_question="Show the top 5 staff with more than 20 sales today.",
        source_answer="Alice Smith was in the top 5.",
        provenance={
            "question_contract": {
                "question_inputs": [
                    {
                        "id": "fact_1_limit_1",
                        "kind": "literal_text",
                        "source": "question_context",
                        "text": "top 5",
                        "role": "result_limit",
                        "resolved_value_text": "5",
                    },
                    {
                        "id": "fact_1_time_1",
                        "kind": "literal_text",
                        "source": "question_context",
                        "text": "today",
                        "role": "time_value",
                        "resolved_value_text": "today",
                    },
                ],
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "top staff with more than 20 sales today",
                        "answer_subject": _answer_subject_payload("staff"),
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "staff sales ranking",
                            }
                        ],
                        "used_question_inputs": [
                            "fact_1_limit_1",
                            "fact_1_time_1",
                        ],
                    }
                ]
            }
        },
        addresses=(
            FactAddress.value(
                address="value.limit_top_5",
                value={"type": "integer", "value": 5},
                display="top 5",
                evidence=EvidenceRef(step_ids=("known_input:fact_1_limit_1",)),
            ),
            FactAddress.value(
                address="value.sales_threshold_20",
                value={"type": "integer", "value": "20"},
                display="20",
                evidence=EvidenceRef(step_ids=("known_input:fact_1_number_1",)),
            ),
        ),
    )
    return {"factArtifacts": [artifact.to_dict()]}


def _memory_context_with_active_clarification() -> dict[str, object]:
    clarification = build_fact_artifact(
        artifact_id="prior_clarification",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        source_question="How much sales did we make yesterday?",
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal=FactOutcome.NEEDS_CLARIFICATION.value,
                clarification_questions=("Which store do you mean?",),
            ),
        ),
    )
    return {"factArtifacts": [clarification.to_dict()]}


def _sales_read() -> EndpointRead:
    return EndpointRead(
        id="sales_read",
        endpoint_name="sales_read",
        resource_names=("sales",),
        params=(
            CatalogParam(
                ref="sales_read.query.location_id",
                name="location_id",
                source=ParamSource.QUERY,
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                ),
            ),
        ),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="field.sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="sale",
                    identity_field="sale_id",
                    primary_key=True,
                ),
            ),
            CatalogField(
                ref="field.sale_amount",
                path="data.sale_amount",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _sales_total_by_date_read() -> EndpointRead:
    return EndpointRead(
        id="sales_total_by_date",
        endpoint_name="sales_total_by_date",
        resource_names=("sales",),
        params=(
            CatalogParam(
                ref="sales_total_by_date.query.start_date",
                name="start_date",
                source=ParamSource.QUERY,
                type="date",
            ),
            CatalogParam(
                ref="sales_total_by_date.query.end_date",
                name="end_date",
                source=ParamSource.QUERY,
                type="date",
            ),
        ),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
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


def _secret_sales_read() -> EndpointRead:
    return EndpointRead(
        id="secret_sales_read",
        endpoint_name="secret_sales_read",
        resource_names=("secret sales",),
        params=(
            CatalogParam(
                ref="secret_sales_read.query.location_id",
                name="location_id",
                source=ParamSource.QUERY,
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                ),
            ),
        ),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="field.secret_sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="sale",
                    identity_field="sale_id",
                    primary_key=True,
                ),
            ),
            CatalogField(
                ref="field.secret_amount",
                path="data.secret_amount",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _staff_sales_read() -> EndpointRead:
    return EndpointRead(
        id="staff_sales_read",
        endpoint_name="staff_sales_read",
        resource_names=("staff", "staff sales"),
        params=(
            CatalogParam(
                ref="staff_sales_read.query.staff_id",
                name="staff_id",
                source=ParamSource.QUERY,
                type="string",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                ),
            ),
            CatalogParam(
                ref="staff_sales_read.query.business_date",
                name="business_date",
                source=ParamSource.QUERY,
                type="date",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                ),
            ),
            CatalogField(
                ref="field.amount",
                path="data.amount",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _staff_list_read() -> EndpointRead:
    return EndpointRead(
        id="staff_list_read",
        endpoint_name="staff_list_read",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="staff_list_read.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
            CatalogField(
                ref="field.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _sales_metric_read(read_id: str) -> EndpointRead:
    return EndpointRead(
        id=read_id,
        endpoint_name=read_id,
        resource_names=("sales",),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
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


def _inventory_read() -> EndpointRead:
    return EndpointRead(
        id="inventory_read",
        endpoint_name="inventory_read",
        resource_names=("inventory",),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="field.inventory_count",
                path="data.inventory_count",
                row_path_id="data",
                type="number",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _orders_above_amount_read() -> EndpointRead:
    return EndpointRead(
        id="orders_above_amount_read",
        endpoint_name="orders_above_amount_read",
        resource_names=("orders",),
        params=(
            CatalogParam(
                ref="orders_above_amount_read.query.minimum_amount",
                name="minimum_amount",
                source=ParamSource.QUERY,
                type="number",
            ),
        ),
        row_paths=(
            RowPath(id="root", path="", cardinality=RowCardinality.ONE),
            RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
        ),
        fields=(
            CatalogField(
                ref="field.order_id",
                path="data.order_id",
                row_path_id="data",
                type="string",
            ),
        ),
        pagination=PaginationMetadata(
            mode=PaginationMode.NONE,
            completeness_policy=CompletenessPolicy.COMPLETE,
        ),
    )


def _source_options_by_fact(prompt: str) -> dict[str, list[dict[str, Any]]]:
    payload = _prompt_json_section(prompt, "Candidate evidence sources")
    output: dict[str, list[dict[str, Any]]] = {}
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        fact_id = str(fact_sources.get("requested_fact_id") or "")
        options: list[dict[str, Any]] = []
        for context in fact_sources.get("source_contexts") or ():
            if isinstance(context, dict):
                options.extend(
                    item
                    for item in context.get("source_options") or ()
                    if isinstance(item, dict)
                )
        output[fact_id] = options
    return output


def _source_contexts_by_fact(prompt: str) -> dict[str, list[dict[str, Any]]]:
    payload = _prompt_json_section(prompt, "Candidate evidence sources")
    output: dict[str, list[dict[str, Any]]] = {}
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        fact_id = str(fact_sources.get("requested_fact_id") or "")
        output[fact_id] = [
            context
            for context in fact_sources.get("source_contexts") or ()
            if isinstance(context, dict)
        ]
    return output


def _catalog_request_has_active_memory_value(request: Any) -> bool:
    return any(
        any(
            str(ref).startswith("memory:")
            for ref in getattr(value, "proof_refs", ()) or ()
        )
        for value in getattr(request, "available_values", ()) or ()
    )


def _candidate_with_read_and_field(
    candidates: list[dict[str, Any]],
    *,
    read_id: str,
    field_id: str,
) -> dict[str, Any]:
    for candidate in candidates:
        if _candidate_read_id(candidate) != read_id:
            continue
        if _candidate_has_field(candidate, field_id):
            return candidate
    raise AssertionError(f"candidate {read_id}.{field_id} not found")


def _candidate_with_read(
    candidates: list[dict[str, Any]],
    *,
    read_id: str,
) -> dict[str, Any]:
    for candidate in candidates:
        if _candidate_read_id(candidate) == read_id:
            return candidate
    raise AssertionError(f"candidate {read_id} not found")


def _candidate_with_field(
    candidates: list[dict[str, Any]],
    *,
    field_id: str,
) -> dict[str, Any]:
    for candidate in candidates:
        if _candidate_has_field(candidate, field_id):
            return candidate
    raise AssertionError(f"candidate with field {field_id} not found")


def _candidate_has_field(candidate: dict[str, Any], field_id: str) -> bool:
    for item in _candidate_evidence_items(candidate):
        if isinstance(item, dict) and item.get("field_id") == field_id:
            return True
    for item in candidate.get("fields") or ():
        if isinstance(item, dict) and item.get("field_id") == field_id:
            return True
    for item in _candidate_binding_surface(candidate).get("fields") or ():
        if isinstance(item, dict) and item.get("field_id") == field_id:
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


def _source_fulfillment(
    candidate: dict[str, Any],
    *,
    field_id: str,
    answer_output_id: str = "answer_1",
) -> dict[str, dict[str, Any]]:
    evidence_id = _candidate_evidence_id(candidate, field_id=field_id)
    choice_id = _candidate_fulfillment_choice_id(
        candidate,
        answer_output_id=answer_output_id,
        evidence_id=evidence_id,
    )
    return {
        answer_output_id: {
            "match_basis_explanation": (
                f"{evidence_id} provides the requested {answer_output_id} value."
            ),
            "fulfillment_choice_id": choice_id,
        }
    }


def _required_output_fields_from_prompt(
    prompt: str,
    *,
    source_binding_id: str,
) -> list[dict[str, str]]:
    evidence_payload = _prompt_json_section(prompt, "Required fulfillment evidence")
    field_ids = tuple(
        dict.fromkeys(
            str(item.get("field_id") or "")
            for requirement in evidence_payload.get("required_fulfillment_evidence")
            or ()
            if isinstance(requirement, dict)
            and str(requirement.get("source_binding_id") or "") == source_binding_id
            for item in requirement.get("must_use_evidence") or ()
            if isinstance(item, dict) and str(item.get("field_id") or "")
        )
    )
    if not field_ids:
        field_ids = _bound_source_field_ids_from_prompt(
            prompt,
            source_binding_id=source_binding_id,
        )
    return [{"field_id": field_id} for field_id in field_ids]


def _computed_scalar_from_bound_value_payload(
    *,
    requested_fact_id: str,
    answer_output_ids: tuple[str, ...],
    source_binding_id: str,
) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "fact_plan",
            "answers": [
                {
                    "requested_fact_id": requested_fact_id,
                    "answer_output_ids": list(answer_output_ids),
                    "pattern": "computed_scalar",
                    "scalar_inputs": [
                        {
                            "input_id": "value",
                            "source_binding_id": source_binding_id,
                        }
                    ],
                    "expression": [{"input_id": "value"}],
                    "output": {"scalar_id": "answer", "label": "answer"},
                }
            ],
        }
    }


def _bound_source_field_ids_from_prompt(
    prompt: str,
    *,
    source_binding_id: str,
) -> tuple[str, ...]:
    bound_sources = _prompt_json_section(prompt, "Bound sources")
    return tuple(
        dict.fromkeys(
            str(field.get("field_id") or "")
            for source in bound_sources.get("bound_sources") or ()
            if isinstance(source, dict)
            and str(source.get("source_binding_id") or "") == source_binding_id
            for field in source.get("fields") or ()
            if isinstance(field, dict)
            and str(field.get("evidence_id") or "")
            and str(field.get("field_id") or "")
        )
    )


def _candidate_evidence_id(candidate: dict[str, Any], *, field_id: str) -> str:
    for item in _candidate_evidence_items(candidate):
        if isinstance(item, dict) and item.get("field_id") == field_id:
            return str(item["evidence_id"])
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
                        return str(item["evidence_id"])
    return field_id


def _candidate_evidence_items(candidate: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in (
            *(_candidate_binding_surface(candidate).get("evidence_items") or ()),
            *(candidate.get("evidence_items") or ()),
        )
        if isinstance(item, dict)
    )


def _candidate_fulfillment_choice_id(
    candidate: dict[str, Any],
    *,
    answer_output_id: str,
    evidence_id: str,
) -> str:
    for support_set in (
        _candidate_binding_surface(candidate).get("fulfillment_support_sets") or ()
    ):
        if not isinstance(support_set, dict):
            continue
        if str(support_set.get("answer_output_id") or "") != answer_output_id:
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            if isinstance(slot, dict) and evidence_id in _slot_evidence_ids(slot):
                return str(support_set.get("fulfillment_choice_id") or "")
    raise AssertionError(
        f"fulfillment support set not found for {answer_output_id}:{evidence_id}"
    )


def _slot_evidence_ids(slot: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "scope_evidence",
            "group_key_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict)
        for evidence_id in (str(item.get("evidence_id") or ""),)
        if evidence_id
    )


def test_pipeline_passes_raw_question_and_typed_resolution_to_question_contract():
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _memory_attribution_response(
                question="How about the day before?",
                conversation_context=_memory_context(),
                selected_memory_id="turn_1.relation.sales_rows",
                contextualized_question="How much money did we make the day before yesterday?",
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="money made the day before yesterday",
                answer_subject="money",
                parts=("total money",),
            ),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="metric_read",
                output_fields=({"field_id": "metric_total", "label": "answer_1"},),
            ),
        }
    )

    run_lookup_question(
        LookupRequest(
            question="How about the day before?",
            run_id="run_contextualized_question",
            conversation_context=_memory_context(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_metric_read("metric_read"))),
            data_access_port=_DataAccessPort(
                {"metric_read": {"data": [{"metric_total": "12"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    question_contract_prompt = planner.prompts[1]
    assert "Current question:\nHow about the day before?" in question_contract_prompt
    assert "Conversation resolution context:" in question_contract_prompt
    assert '"resolved_values"' in question_contract_prompt
    assert (
        "How much money did we make the day before yesterday?"
        not in question_contract_prompt
    )


def test_pipeline_does_not_pass_memory_cards_to_question_contract_prompt():
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _memory_attribution_response(
                question="How much did those sales make?",
                conversation_context=_memory_context(),
                selected_memory_id="turn_1.relation.sales_rows",
                contextualized_question="How much money did the prior sales make?",
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="money made by prior sales",
                answer_subject="money",
                parts=("total money",),
            ),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="metric_read",
                output_fields=({"field_id": "metric_total", "label": "answer_1"},),
            ),
        }
    )

    run_lookup_question(
        LookupRequest(
            question="How much did those sales make?",
            run_id="run_no_cards_in_question_contract",
            conversation_context=_memory_context(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_metric_read("metric_read"))),
            data_access_port=_DataAccessPort(
                {"metric_read": {"data": [{"metric_total": "12"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    question_contract_prompt = planner.prompts[1]
    assert "Available Memory Cards" not in question_contract_prompt
    assert "prior_reference_candidates" not in question_contract_prompt


def test_standalone_resolution_is_backend_pass_through_without_model_rewrite():
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _standalone_attribution_response(
                question="How much money did we make yesterday?",
                conversation_context=_memory_context_with_staff_identities(),
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="money made yesterday",
                answer_subject="money",
                answer_expression_family="list_rows",
                parts=("total money",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sales",)),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="sales_total_read",
                output_fields=({"field_id": "metric_total", "label": "answer_1"},),
            ),
        }
    )
    result = run_lookup_question(
        LookupRequest(
            question="How much money did we make yesterday?",
            run_id="run_standalone_pass_through",
            conversation_context=_memory_context_with_staff_identities(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(_sales_metric_read("sales_total_read"))
            ),
            data_access_port=_DataAccessPort(
                {"sales_total_read": {"data": [{"metric_total": "12"}]}}
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.tool_names)
    question_contract_prompt = planner.prompts[
        planner.tool_names.index("submit_answer_request_contract")
    ]
    assert (
        "Current question:\nHow much money did we make yesterday?"
        in question_contract_prompt
    )
    assert "Integrated question:" not in question_contract_prompt
    assert "Alice Smith" not in question_contract_prompt
    assert "Jane Doe" not in question_contract_prompt


def test_standalone_referential_question_does_not_activate_or_expose_memory():
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _standalone_attribution_response(
                question="How much did she sell yesterday?",
                conversation_context=_memory_context_with_staff_identities(),
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="her sales yesterday",
                answer_subject="she",
                parts=("sales amount",),
            ),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="staff_sales_read",
                output_fields=({"field_id": "amount", "label": "answer_1"},),
            ),
        }
    )

    run_lookup_question(
        LookupRequest(
            question="How much did she sell yesterday?",
            run_id="run_standalone_referential_no_memory",
            conversation_context=_memory_context_with_staff_identities(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_staff_sales_read())),
            data_access_port=_DataAccessPort(
                {
                    "staff_sales_read": {
                        "data": [{"staff_id": "unknown", "amount": "12"}]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    question_contract_prompt = planner.prompts[
        planner.tool_names.index("submit_answer_request_contract")
    ]
    assert (
        "Current question:\nHow much did she sell yesterday?"
        in question_contract_prompt
    )
    assert "Integrated question:" not in question_contract_prompt
    if "submit_source_binding" in planner.tool_names:
        source_binding_prompt = planner.prompts[
            planner.tool_names.index("submit_source_binding")
        ]
        assert "turn_alice" not in source_binding_prompt
        assert "staff_alice" not in source_binding_prompt
        assert "Alice Smith" not in source_binding_prompt


def test_active_clarification_requires_clause_resolution_not_standalone():
    standalone_planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: {
                "kind": "conversation_resolution",
            },
            "submit_answer_request_contract": _question_contract_response(
                subject="ABC Mall",
                parts=("ABC Mall",),
            ),
        }
    )

    standalone_result = run_lookup_question(
        LookupRequest(
            question="ABC Mall",
            run_id="run_active_clarification_standalone_wrong",
            conversation_context=_memory_context_with_active_clarification(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_metric_read("metric_read"))),
            data_access_port=_DataAccessPort({}),
            planner_model_port=standalone_planner,
        ),
    )

    assert standalone_result.status == "FAILED"
    assert standalone_planner.tool_names == [CONVERSATION_RESOLUTION_TOOL_NAME]

    resolved_planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _memory_attribution_response(
                question="ABC Mall",
                conversation_context=_memory_context_with_active_clarification(),
                selected_memory_id="prior_clarification.outcome.needs_clarification",
                contextualized_question="How much sales did we make at ABC Mall yesterday?",
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="sales at ABC Mall yesterday",
                answer_subject="sales",
                answer_expression_family="list_rows",
                parts=("sales amount",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sales",)),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="sales_total_read",
                output_fields=({"field_id": "metric_total", "label": "answer_1"},),
            ),
        }
    )

    resolved_result = run_lookup_question(
        LookupRequest(
            question="ABC Mall",
            run_id="run_active_clarification_clause_resolution",
            conversation_context=_memory_context_with_active_clarification(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(_sales_metric_read("sales_total_read"))
            ),
            data_access_port=_DataAccessPort(
                {"sales_total_read": {"data": [{"metric_total": "12"}]}}
            ),
            planner_model_port=resolved_planner,
        ),
    )

    assert resolved_result.status == "COMPLETED", (
        resolved_result,
        resolved_planner.tool_names,
        resolved_planner.prompts,
    )
    question_contract_prompt = resolved_planner.prompts[
        resolved_planner.tool_names.index("submit_answer_request_contract")
    ]
    assert "Current question:\nABC Mall" in question_contract_prompt
    assert "Conversation resolution context:" in question_contract_prompt
    assert "Prior question:" not in question_contract_prompt
    assert "Clarification answer:" not in question_contract_prompt


def test_active_clarification_memory_card_includes_clarification_question():
    projection = project_conversation_memory_cards(
        _memory_context_with_active_clarification(),
        current_question="ABC Mall",
    )

    card = next(
        card for card in projection.cards if card.kind == "clarification_answer"
    )
    assert (
        card.display
        == "How much sales did we make yesterday? Clarification needed: Which store do you mean?"
    )
    assert card.details == {
        "clarification_question": "Which store do you mean?",
        "question_being_clarified": "How much sales did we make yesterday?",
    }


def test_activated_memory_does_not_hide_current_question_source_candidates(monkeypatch):
    planner = _TwoFactActiveMemoryPlannerPort()
    catalog_requests: list[Any] = []
    original_select_relation_catalog = lookup_pipeline.select_relation_catalog

    def capture_catalog_request(request: Any) -> Any:
        catalog_requests.append(request)
        return original_select_relation_catalog(request)

    monkeypatch.setattr(
        lookup_pipeline,
        "select_relation_catalog",
        capture_catalog_request,
    )

    result = run_lookup_question(
        LookupRequest(
            question="For those sales, show the amount; also show current inventory.",
            run_id="run_active_memory_fact_scope",
            conversation_context=_memory_context_with_prior_sales_and_location_identity(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(_sales_read(), _inventory_read())
            ),
            data_access_port=_DataAccessPort(
                {
                    "sales_read": {
                        "data": [{"sale_id": "sale-1", "sale_amount": "10.00"}]
                    },
                    "inventory_read": {"data": [{"inventory_count": 7}]},
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.tool_names)
    for tool_name in (
        "submit_answer_request_contract",
        "submit_query_enrichment",
        "submit_source_binding",
    ):
        prompt = planner.prompts[planner.tool_names.index(tool_name)]
        assert (
            "Current question:\nFor those sales, show the amount; also show current inventory."
            in prompt
        )
        if tool_name == "submit_answer_request_contract":
            assert "Conversation resolution context:" in prompt
        assert "For the prior sales, show sale amount." not in prompt
    for tool_name in ("submit_source_alignment_reviews", "submit_pattern_fact_plan"):
        prompt = planner.prompts[planner.tool_names.index(tool_name)]
        assert (
            "Current question:\nFor those sales, show the amount; also show current inventory."
            in prompt
        )
        assert "Conversation resolution annotations:" not in prompt
        assert "For the prior sales, show sale amount." not in prompt
    source_binding_prompt = planner.prompts[
        planner.tool_names.index("submit_source_binding")
    ]
    candidates_by_fact = _source_options_by_fact(source_binding_prompt)
    contexts_by_fact = _source_contexts_by_fact(source_binding_prompt)
    assert all(
        context.get("kind") != "active_memory"
        for contexts in contexts_by_fact.values()
        for context in contexts
    )
    assert any(
        _candidate_read_id(candidate) == "inventory_read"
        for candidate in candidates_by_fact.get("fact_2", [])
    )
    assert catalog_requests
    assert all(not request.active_memory_signals for request in catalog_requests)
    assert all(
        not _catalog_request_has_active_memory_value(request)
        for request in catalog_requests
    )


def test_source_binding_prompt_excludes_unactivated_memory_context():
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _memory_attribution_response(
                question="Show the amount for those sales.",
                conversation_context=_memory_context_with_selected_and_unselected_memory(),
                selected_memory_id="turn_selected_sales.relation.sales",
                contextualized_question="Show sale amount for the prior sales.",
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="the prior sales",
                answer_subject="sales",
                answer_expression_family="list_rows",
                parts=("sale amount",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sales",)),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="sales_read",
                output_fields=({"field_id": "sale_amount", "label": "answer_1"},),
            ),
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="Show the amount for those sales.",
            run_id="run_unactivated_memory_exclusion",
            conversation_context=_memory_context_with_selected_and_unselected_memory(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(_sales_read(), _secret_sales_read())
            ),
            data_access_port=_DataAccessPort(
                {
                    "sales_read": {
                        "data": [{"sale_id": "sale-1", "sale_amount": "10.00"}]
                    },
                    "secret_sales_read": {
                        "data": [
                            {"sale_id": "secret-sale-1", "secret_amount": "999.00"}
                        ]
                    },
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.tool_names, planner.prompts)
    source_binding_prompt = planner.prompts[
        planner.tool_names.index("submit_source_binding")
    ]
    assert "turn_selected_sales.relation.sales" in source_binding_prompt
    assert "turn_unselected_sales" not in source_binding_prompt
    assert "relation.secret_sales" not in source_binding_prompt
    plan_prompt = planner.prompts[planner.tool_names.index("submit_pattern_fact_plan")]
    assert '"value": "10"' in plan_prompt
    assert "999.00" not in plan_prompt


def test_source_binding_prompt_uses_only_current_run_grounded_values():
    planner = _ToolNamePlannerPort(
        read_eligibility_retention_specs=(
            ReadEligibilityRetentionSpec(
                requested_fact_id="fact_1",
                read_id="orders_above_amount_read",
                group_key_fields=("order_id",),
            ),
        ),
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _memory_attribution_response(
                question="Show orders above that amount.",
                conversation_context=_memory_context_with_selected_and_unselected_scalar_values(),
                selected_memory_id="turn_selected_threshold.value.threshold",
                contextualized_question="Show orders above 50.",
                source_containing="50",
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="orders above 50",
                answer_subject="orders",
                answer_expression_family="list_rows",
                parts=("order ids",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("orders",)),
            "submit_source_alignment_reviews": {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "pattern": "list_rows",
                            "source": {
                                "kind": "read",
                                "read_id": "orders_above_amount_read",
                            },
                        }
                    ],
                }
            },
            "submit_source_binding": {
                "outcome": {
                    "kind": "impossible",
                    "blocked_facts": [
                        {
                            "requested_fact_id": "fact_1",
                            "basis": "catalog_access",
                            "evidence_refs": ["catalog_selection:fact_1"],
                            "explanation": "stop after source binding prompt capture",
                        }
                    ],
                }
            },
        },
    )

    run_lookup_question(
        LookupRequest(
            question="Show orders above that amount.",
            run_id="run_active_scalar_param_binding",
            conversation_context=_memory_context_with_selected_and_unselected_scalar_values(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_orders_above_amount_read())),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    source_binding_prompt = planner.prompts[
        planner.tool_names.index("submit_source_binding")
    ]
    grounded_values = _prompt_json_section(source_binding_prompt, "Grounded values")
    assert grounded_values == {"values": []}


def test_runtime_expands_selected_memory_through_activation_chokepoint(monkeypatch):
    calls: list[tuple[str, ...]] = []
    real_expand = lookup_pipeline.expand_activated_memory_cards

    def tracking_expand_activated_memory_cards(**kwargs):
        calls.append(tuple(kwargs["used_memory_ids"]))
        return real_expand(**kwargs)

    monkeypatch.setattr(
        lookup_pipeline,
        "expand_activated_memory_cards",
        tracking_expand_activated_memory_cards,
    )
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _memory_attribution_response(
                question="Show the amount for those sales.",
                conversation_context=_memory_context_with_selected_and_unselected_memory(),
                selected_memory_id="turn_selected_sales.relation.sales",
                contextualized_question="Show sale amount for the prior sales.",
            ),
            "submit_answer_request_contract": _question_contract_response(
                subject="the prior sales",
                answer_subject="sales",
                parts=("sale amount",),
            ),
            "submit_query_enrichment": _query_enrichment_payload(("sales",)),
            "submit_pattern_fact_plan": _pattern_fact_plan_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                read_id="sales_read",
                output_fields=({"field_id": "sale_amount", "label": "answer_1"},),
            ),
        }
    )

    run_lookup_question(
        LookupRequest(
            question="Show the amount for those sales.",
            run_id="run_activation_chokepoint",
            conversation_context=_memory_context_with_selected_and_unselected_memory(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_sales_read())),
            data_access_port=_DataAccessPort(
                {
                    "sales_read": {
                        "data": [{"sale_id": "sale-1", "sale_amount": "10.00"}]
                    }
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert calls == [("turn_selected_sales.relation.sales",)]


def test_selected_prior_request_outputs_reach_question_contract():
    context = _memory_context_with_prior_staff_sales_request()
    prior_memory_id = "turn_staff_sales.prior_request.fact_1"
    entity_memory_id = "turn_staff_sales.entity.grounded_fact_1_entity_1"
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: (
                _clause_resolution_conversation_response(
                    question="How much did Alice Smith sell yesterday?",
                    conversation_context=context,
                    contextualized_question="What is the total sales amount for Alice Smith yesterday?",
                    prior_request_memory_id=prior_memory_id,
                    entity_memory_id=entity_memory_id,
                )
            ),
                "submit_answer_request_contract": _question_contract_response(
                    subject="total sales amount for Alice Smith yesterday",
                    answer_subject="Alice Smith",
                    answer_expression_family="computed_scalar",
                    parts=("total sales amount",),
                    question_inputs=(
                        {
                            "input_ref": "input_staff",
                            "kind": "literal_text",
                            "source": "conversation_resolution",
                            "value_source_text": "Alice Smith",
                            "resolved_input_ref": "conversation.entity_value",
                            "role": "reference_value",
                            "value_meaning_hint": "staff identity",
                            "resolved_value_text": "Alice Smith",
                        },
                        {
                            "input_ref": "input_period",
                            "kind": "literal_text",
                            "source": "question_context",
                            "value_source_text": "yesterday",
                            "role": "time_value",
                            "resolved_value_text": "yesterday",
                        },
                    ),
                ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("staff sales",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "input_staff",
                        "catalog_search_terms": [
                            {"basis": "staff member", "term": "staff"},
                        ],
                    }
                ],
            ),
            "submit_pattern_fact_plan": _computed_scalar_from_bound_value_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                source_binding_id="sb_1",
            ),
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="How much did Alice Smith sell yesterday?",
            run_id="run_clause_resolution_prior_output",
            conversation_context=context,
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-15",
                timezone="Africa/London",
            ),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(_staff_sales_read(), _staff_list_read())
            ),
            data_access_port=_DataAccessPort(
                {
                    "staff_list_read": {
                        "data": [
                            {
                                "staff_id": "staff_alice",
                                "name": "Alice Smith",
                            }
                        ]
                    },
                    "staff_sales_read": {
                        "data": [{"staff_id": "staff_alice", "amount": "12"}]
                    },
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.tool_names)
    question_contract_prompt = planner.prompts[
        planner.tool_names.index("submit_answer_request_contract")
    ]
    assert prior_memory_id not in question_contract_prompt
    assert '"resolved_values"' in question_contract_prompt
    assert '"answer_output"' in question_contract_prompt
    assert "total sales amount" in question_contract_prompt


def test_clause_resolution_prior_answer_frame_reaches_question_contract():
    context = _memory_context_with_prior_staff_sales_request()
    prior_memory_id = "turn_staff_sales.prior_request.fact_1"
    contextualized_question = (
        "What is the total sales amount for products Alice Smith sold yesterday?"
    )
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: _clause_resolution_response(
                question="And how much did she make yesterday?",
                conversation_context=context,
                contextualized_question=contextualized_question,
                prior_request_memory_id=prior_memory_id,
            ),
                "submit_answer_request_contract": _question_contract_response(
                    subject=(
                        "total sales amount for products Alice Smith sold yesterday"
                    ),
                    answer_subject="she",
                    answer_expression_family="computed_scalar",
                    parts=("total sales amount",),
                    question_inputs=(
                        {
                            "input_ref": "input_staff",
                            "kind": "literal_text",
                            "source": "conversation_resolution",
                            "value_source_text": "Alice Smith",
                            "resolved_input_ref": "conversation.entity_value",
                            "role": "reference_value",
                            "value_meaning_hint": "staff identity",
                            "resolved_value_text": "Alice Smith",
                        },
                        {
                            "input_ref": "input_period",
                            "kind": "literal_text",
                            "source": "question_context",
                            "value_source_text": "yesterday",
                            "role": "time_value",
                            "resolved_value_text": "yesterday",
                        },
                    ),
                ),
            "submit_query_enrichment": _query_enrichment_payload(
                ("staff sales",),
                entity_target_catalog_search_terms=[
                    {
                        "target_id": "input_staff",
                        "catalog_search_terms": [
                            {"basis": "staff member", "term": "staff"},
                        ],
                    }
                ],
            ),
            "submit_pattern_fact_plan": _computed_scalar_from_bound_value_payload(
                requested_fact_id="fact_1",
                answer_output_ids=("answer_1",),
                source_binding_id="sb_1",
            ),
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="And how much did she make yesterday?",
            run_id="run_clause_resolution_prior_answer_frame",
            conversation_context=context,
            runtime_values=RuntimeValueContext(
                runtime_date="2026-05-15",
                timezone="Africa/London",
            ),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(
                _catalog(_staff_sales_read(), _staff_list_read())
            ),
            data_access_port=_DataAccessPort(
                {
                    "staff_list_read": {
                        "data": [
                            {
                                "staff_id": "staff_alice",
                                "name": "Alice Smith",
                            }
                        ]
                    },
                    "staff_sales_read": {
                        "data": [{"staff_id": "staff_alice", "amount": "12"}]
                    },
                }
            ),
            planner_model_port=planner,
        ),
    )

    question_contract_prompt = planner.prompts[
        planner.tool_names.index("submit_answer_request_contract")
    ]
    assert (
        "Current question:\nAnd how much did she make yesterday?"
        in question_contract_prompt
    )
    assert "Conversation resolution context:" in question_contract_prompt
    assert '"resolved_values"' in question_contract_prompt
    assert '"answer_output"' in question_contract_prompt
    assert "total sales amount" in question_contract_prompt
    assert result.status == "COMPLETED", (result, planner.tool_names, planner.prompts)


def test_question_scoped_active_memory_is_available_as_explicit_candidate_context():
    planner = _TwoSalesFactActiveMemoryPlannerPort()

    result = run_lookup_question(
        LookupRequest(
            question="For those sales, show the amount; also show sale id.",
            run_id="run_active_memory_token_overlap_scope",
            conversation_context=_memory_context_with_prior_sales_and_location_identity(),
            provider_preferences=_PROVIDER_PREFS,
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_sales_read())),
            data_access_port=_DataAccessPort(
                {
                    "sales_read": {
                        "data": [{"sale_id": "sale-1", "sale_amount": "10.00"}]
                    },
                }
            ),
            planner_model_port=planner,
        ),
    )

    assert result.status == "COMPLETED", (result, planner.tool_names, planner.prompts)
    source_binding_prompt = planner.prompts[
        planner.tool_names.index("submit_source_binding")
    ]
    contexts_by_fact = _source_contexts_by_fact(source_binding_prompt)
    assert all(
        context.get("kind") != "active_memory"
        for contexts in contexts_by_fact.values()
        for context in contexts
    )
    assert any(
        option.get("memory_relation_id") == "turn_sales.relation.sales"
        for contexts in contexts_by_fact.values()
        for context in contexts
        if context.get("kind") == "memory_sources"
        for option in context.get("source_options") or ()
        if isinstance(option, dict)
    )


def test_active_clarification_context_uses_chronological_artifacts():
    answered = build_fact_artifact(
        artifact_id="turn_answered",
        outcome=FactOutcome.ANSWERED,
        source_question="What were sales yesterday?",
        addresses=(
            FactAddress.value(
                address="value.total",
                value={"type": "number", "value": "10.00"},
            ),
        ),
    )
    clarification = build_fact_artifact(
        artifact_id="turn_clarification",
        outcome=FactOutcome.NEEDS_CLARIFICATION,
        source_question="How much money did we make yesterday?",
        addresses=(
            FactAddress.outcome(
                address="outcome.needs_clarification",
                terminal="needs_clarification",
                clarification_questions=("Which location?",),
            ),
        ),
    )

    context = active_clarification_context(
        {"factArtifacts": [answered.to_dict(), clarification.to_dict()]},
        current_question="ABC Mall",
    )

    assert context is not None
    assert context.original_question == "How much money did we make yesterday?"
    assert context.exchanges[0].questions == ("Which location?",)
    assert context.exchanges[0].answer == "ABC Mall"


@dataclass
class _TwoFactActiveMemoryPlannerPort:
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
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
        self.calls += 1
        self.prompts.append(prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        self.tool_names.append(tool_name)
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            return _tool_response(
                CONVERSATION_RESOLUTION_TOOL_NAME,
                _memory_attribution_response(
                    question=(
                        "For those sales, show the amount; also show current inventory."
                    ),
                    conversation_context=(
                        _memory_context_with_prior_sales_and_location_identity()
                    ),
                    selected_memory_id="turn_sales.relation.sales",
                    contextualized_question=(
                        "For the prior sales, show sale amount. "
                        "Show current inventory."
                    ),
                    source_containing="sales",
                ),
            )
        if tool_name == "submit_answer_request_contract":
            return _tool_response(
                tool_name,
                {
                    "kind": "question_contract",
                    "answer_requests_count": 2,
                    "question_inputs": [],
                    "answer_requests": [
                        {
                            "answer_fact": "the prior sales",
                            "answer_expression": {"family": "list_rows"},
                            "answer_subject": _answer_subject_payload("sales"),
                            "answer_population": _answer_population_payload_from_text(
                                description="the prior sales",
                                subject_text="sales",
                            ),
                            "answer_outputs": [{"description": "sale amount"}],
                            "used_question_inputs": [],
                        },
                        {
                            "answer_fact": "current inventory",
                            "answer_expression": {"family": "list_rows"},
                            "answer_subject": _answer_subject_payload("inventory"),
                            "answer_population": _answer_population_payload_from_text(
                                description="current inventory",
                                subject_text="inventory",
                            ),
                            "answer_outputs": [{"description": "inventory count"}],
                            "used_question_inputs": [],
                        },
                    ],
                    "question_input_inventory_check": {
                        "all_input_like_phrases_declared": True,
                    },
                },
            )
        if tool_name == "submit_query_enrichment":
            return _tool_response(
                tool_name,
                {
                    "requested_fact_resource_name_matches": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_resource_lineage": [
                                {
                                    "answer_output_id": "answer_1",
                                    "support_role": "MEASURED_VALUE",
                                    "source_text": "sales",
                                    "matching_resource_names": ["sales"],
                                }
                            ],
                        },
                        {
                            "requested_fact_id": "fact_2",
                            "answer_output_resource_lineage": [
                                {
                                    "answer_output_id": "answer_1",
                                    "support_role": "MEASURED_VALUE",
                                    "source_text": "inventory",
                                    "matching_resource_names": ["inventory"],
                                }
                            ],
                        },
                    ],
                    "entity_target_catalog_search_terms": [],
                },
            )
        if tool_name == "submit_read_eligibility":
            return read_eligibility_response_from_prompt(
                prompt,
                retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("sale_amount",),
                    ),
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_2",
                        read_id="inventory_read",
                        answer_value_fields=("inventory_count",),
                    ),
                ),
            )
        if tool_name == "submit_source_binding":
            prior_sales = source_candidate_with_fields(
                prompt,
                requested_fact_id="fact_1",
                required=("sale_amount",),
            )
            inventory = source_candidate_with_fields(
                prompt,
                requested_fact_id="fact_2",
                read_id="inventory_read",
                required=("inventory_count",),
            )
            self.source_binding_payload = {
                "outcome": {
                    "kind": "source_bindings",
                    "source_invocations": [
                        {
                            "binding_target_id": source_binding_target_id_for_candidate(
                                prompt,
                                requested_fact_id="fact_1",
                                source_candidate_id=str(
                                    prior_sales["source_candidate_id"]
                                ),
                                plan_shape="list_rows",
                            ),
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                source_candidate_id=str(
                                    prior_sales["source_candidate_id"]
                                ),
                            ),
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                prior_sales,
                                field_ids=("sale_amount",),
                            ),
                            "param_decisions": {},
                        },
                        {
                            "binding_target_id": source_binding_target_id_for_candidate(
                                prompt,
                                requested_fact_id="fact_2",
                                source_candidate_id=str(
                                    inventory["source_candidate_id"]
                                ),
                                plan_shape="list_rows",
                            ),
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                source_candidate_id=str(
                                    inventory["source_candidate_id"]
                                ),
                            ),
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                inventory,
                                field_ids=("inventory_count",),
                            ),
                            "param_decisions": {},
                        },
                    ],
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
                    {
                        "outcome": {
                            "kind": "fact_plan",
                            "answers": [
                                {
                                    "requested_fact_id": "fact_1",
                                    "pattern": "list_rows",
                                    "source": {
                                        "kind": "memory_relation",
                                        "memory_relation_id": (
                                            "turn_sales.relation.sales"
                                        ),
                                    },
                                },
                                {
                                    "requested_fact_id": "fact_2",
                                    "pattern": "list_rows",
                                    "source": {
                                        "kind": "read",
                                        "read_id": "inventory_read",
                                    },
                                },
                            ],
                        }
                    },
                    prompt=prompt,
                ),
            )
        if tool_name == "submit_pattern_fact_plan":
            return _tool_response(
                tool_name,
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "fact_1",
                                "answer_output_ids": ["answer_1"],
                                "pattern": "list_rows",
                                "source_binding_id": "sb_1",
                                "output_fields": [{"field_id": "sale_amount"}],
                            },
                            {
                                "requested_fact_id": "fact_2",
                                "answer_output_ids": ["answer_1"],
                                "pattern": "list_rows",
                                "source_binding_id": "sb_2",
                                "output_fields": [{"field_id": "inventory_count"}],
                            },
                        ],
                    }
                },
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


@dataclass
class _TwoSalesFactActiveMemoryPlannerPort:
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
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
        self.calls += 1
        self.prompts.append(prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        self.tool_names.append(tool_name)
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            return _tool_response(
                CONVERSATION_RESOLUTION_TOOL_NAME,
                _memory_attribution_response(
                    question=(
                        "For those sales, show the amount; also show sale id."
                    ),
                    conversation_context=(
                        _memory_context_with_prior_sales_and_location_identity()
                    ),
                    selected_memory_id="turn_sales.relation.sales",
                    contextualized_question=(
                        "For the prior sales, show sale amount and sale id."
                    ),
                    source_containing="sales",
                ),
            )
        if tool_name == "submit_answer_request_contract":
            return _tool_response(
                tool_name,
                {
                    "kind": "question_contract",
                    "answer_requests_count": 2,
                    "question_inputs": [],
                    "answer_requests": [
                        {
                            "answer_fact": "the prior sales",
                            "answer_expression": {"family": "list_rows"},
                            "answer_subject": _answer_subject_payload("sales"),
                            "answer_population": _answer_population_payload_from_text(
                                description="the prior sales",
                                subject_text="sales",
                            ),
                            "answer_outputs": [{"description": "sale amount"}],
                            "used_question_inputs": [],
                        },
                        {
                            "answer_fact": "sale id",
                            "answer_expression": {"family": "list_rows"},
                            "answer_subject": _answer_subject_payload("sale id"),
                            "answer_population": _answer_population_payload_from_text(
                                description="sale id",
                                subject_text="sale id",
                            ),
                            "answer_outputs": [{"description": "sale id"}],
                            "used_question_inputs": [],
                        },
                    ],
                    "question_input_inventory_check": {
                        "all_input_like_phrases_declared": True,
                    },
                },
            )
        if tool_name == "submit_query_enrichment":
            return _tool_response(
                tool_name,
                {
                    "requested_fact_resource_name_matches": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_resource_lineage": [
                                {
                                    "answer_output_id": "answer_1",
                                    "support_role": "MEASURED_VALUE",
                                    "source_text": "sales",
                                    "matching_resource_names": ["sales"],
                                }
                            ],
                        },
                        {
                            "requested_fact_id": "fact_2",
                            "answer_output_resource_lineage": [
                                {
                                    "answer_output_id": "answer_1",
                                    "support_role": "MEASURED_VALUE",
                                    "source_text": "sales",
                                    "matching_resource_names": ["sales"],
                                }
                            ],
                        },
                    ],
                    "entity_target_catalog_search_terms": [],
                },
            )
        if tool_name == "submit_read_eligibility":
            return read_eligibility_response_from_prompt(
                prompt,
                retention_specs=(
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_1",
                        read_id="sales_read",
                        answer_value_fields=("sale_amount",),
                    ),
                    ReadEligibilityRetentionSpec(
                        requested_fact_id="fact_2",
                        read_id="sales_read",
                        answer_value_fields=("sale_id",),
                    ),
                ),
            )
        if tool_name == "submit_source_binding":
            prior_sales = source_candidate_with_fields(
                prompt,
                requested_fact_id="fact_1",
                required=("sale_amount",),
            )
            sale_id = source_candidate_with_fields(
                prompt,
                requested_fact_id="fact_2",
                required=("sale_id",),
            )
            self.source_binding_payload = {
                "outcome": {
                    "kind": "source_bindings",
                    "source_invocations": [
                        {
                            "binding_target_id": source_binding_target_id_for_candidate(
                                prompt,
                                requested_fact_id="fact_1",
                                source_candidate_id=str(
                                    prior_sales["source_candidate_id"]
                                ),
                                plan_shape="list_rows",
                            ),
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                source_candidate_id=str(
                                    prior_sales["source_candidate_id"]
                                ),
                            ),
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                prior_sales,
                                field_ids=("sale_amount",),
                            ),
                            "param_decisions": {},
                        },
                        {
                            "binding_target_id": source_binding_target_id_for_candidate(
                                prompt,
                                requested_fact_id="fact_2",
                                source_candidate_id=str(sale_id["source_candidate_id"]),
                                plan_shape="list_rows",
                            ),
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                source_candidate_id=str(sale_id["source_candidate_id"]),
                            ),
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                sale_id,
                                field_ids=("sale_id",),
                            ),
                            "param_decisions": {},
                        },
                    ],
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
                    {
                        "outcome": {
                            "kind": "fact_plan",
                            "answers": [
                                {
                                    "requested_fact_id": "fact_1",
                                    "pattern": "list_rows",
                                    "source": {
                                        "kind": "memory_relation",
                                        "memory_relation_id": (
                                            "turn_sales.relation.sales"
                                        ),
                                    },
                                },
                                {
                                    "requested_fact_id": "fact_2",
                                    "pattern": "list_rows",
                                    "source": {
                                        "kind": "memory_relation",
                                        "memory_relation_id": (
                                            "turn_sales.relation.sales"
                                        ),
                                    },
                                },
                            ],
                        }
                    },
                    prompt=prompt,
                ),
            )
        if tool_name == "submit_pattern_fact_plan":
            return _tool_response(
                tool_name,
                {
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
                                    source_binding_id="sb_1",
                                ),
                            },
                            {
                                "requested_fact_id": "fact_2",
                                "answer_output_ids": ["answer_1"],
                                "pattern": "list_rows",
                                "source_binding_id": "sb_2",
                                "output_fields": _required_output_fields_from_prompt(
                                    prompt,
                                    source_binding_id="sb_2",
                                ),
                            },
                        ],
                    }
                },
            )
        raise AssertionError(f"unexpected tool: {tool_name}")


def _tool_response(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
