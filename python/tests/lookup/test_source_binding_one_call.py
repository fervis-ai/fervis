import json
from dataclasses import replace

import pytest

from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    EndpointRead,
    IdentityMetadata,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFactBasis,
    PlanImpossible,
)
from fervis.lookup.fact_plan.row_sources import api_row_source_id
from fervis.lookup.fact_plan.values import (
    FactValue,
    TimeComponent,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    NormalInstanceExcludedStateRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactLiteralInput,
)
from fervis.lookup.read_eligibility.candidate_identity import (
    read_candidate_signature,
)
from fervis.lookup.read_eligibility import (
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)
from fervis.lookup.source_binding import (
    AnswerPopulation,
    BoundSource,
    SourceEvidenceItem,
    SourceFulfillment,
    SourceBindingRequest,
    SourceBindingTurnPrompt,
    generate_source_binding,
)
from fervis.lookup.plan_selection import (
    SelectedSourceStrategy,
    PlanSelectionSet,
    SourceStrategyMember,
)
from fervis.lookup.operation_families.source_binding_registry import (
    source_binding_metric_evidence_ids_by_requested_fact,
)
from fervis.lookup.source_binding.candidates.bound_payload import (
    _bound_sources_prompt_payload,
)
from fervis.lookup.source_binding.candidates import (
    raw_payload as raw_payload_module,
)
from fervis.lookup.source_binding.candidates.fulfillment_slots import (
    _candidate_with_fulfillment_slots,
)
from fervis.lookup.source_binding.candidates.evidence import (
    _candidate_with_evidence_items,
)
from fervis.lookup.source_binding.candidates import source_candidate_registry
from fervis.lookup.source_binding.candidates.registry import (
    source_candidate_discovery_payload,
)
from fervis.lookup.source_binding.parser.candidate_access import (
    candidate_value_is_used_by_bound_source,
)
from fervis.lookup.source_binding.parser import parse_source_binding
from tests.lookup.source_binding_helpers import source_fulfills_for_candidate


def _candidate_with_evidence_and_fulfillment_slots(
    candidate: dict[str, object],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> dict[str, object]:
    return _candidate_with_fulfillment_slots(
        _candidate_with_evidence_items(candidate),
        requested_facts=requested_facts,
    )


def _selected_plan(
    *,
    requested_fact_id: str = "fact_1",
    source_candidate_ids: tuple[str, ...] = ("source_1",),
    plan_shape: str = "direct_field_value",
    answer_output_ids: tuple[str, ...] = ("answer_1",),
) -> PlanSelectionSet:
    return PlanSelectionSet(
        plan_selections=(
            SelectedSourceStrategy(
                plan_selection_id=f"plan.{requested_fact_id}",
                requested_fact_id=requested_fact_id,
                source_strategy_id=f"source_strategy.{requested_fact_id}.{plan_shape}.1",
                plan_shape=plan_shape,
                required_answer_output_ids=answer_output_ids,
                source_members=tuple(
                    SourceStrategyMember(source_candidate_id=source_candidate_id)
                    for source_candidate_id in source_candidate_ids
                ),
                basis="Selected by test fixture.",
            ),
        )
    )


def _selected_plans(
    selections: tuple[tuple[str, str], ...],
    *,
    plan_shape: str = "direct_field_value",
    answer_output_ids: tuple[str, ...] = ("answer_1",),
) -> PlanSelectionSet:
    return PlanSelectionSet(
        plan_selections=tuple(
            SelectedSourceStrategy(
                plan_selection_id=f"plan.{requested_fact_id}",
                requested_fact_id=requested_fact_id,
                source_strategy_id=f"source_strategy.{requested_fact_id}.{plan_shape}.1",
                plan_shape=plan_shape,
                required_answer_output_ids=answer_output_ids,
                source_members=(
                    SourceStrategyMember(source_candidate_id=source_candidate_id),
                ),
                basis="Selected by test fixture.",
            )
            for requested_fact_id, source_candidate_id in selections
        )
    )


def test_source_binding_does_not_promote_utility_sources_as_fact_sources_after_read_eligibility(
    monkeypatch: pytest.MonkeyPatch,
):
    def relation_payload(*args, **kwargs):
        return {
            "requested_fact_relations": [
                {
                    "requested_fact_id": "fact_1",
                    "query_terms": [],
                    "available_relations": [],
                }
            ],
            "utility_relations": [
                {
                    "calendar_id": "calendar_days",
                    "kind": "generated_calendar",
                    "fields": [],
                    "params": [],
                }
            ],
        }

    monkeypatch.setattr(
        raw_payload_module,
        "available_relation_catalog_payload",
        relation_payload,
    )
    request = _request_with_optional_params(
        read_eligibility=ReadEligibilityResult(read_assessments=())
    )
    request = replace(
        request,
        catalog_selection=CatalogSelectionResult(
            relation_catalog=RelationCatalog(reads=()),
            requested_fact_selections=(
                RequestedFactCatalogSelection(
                    requested_fact_id="fact_1",
                    query_terms=("sales", "today"),
                    rankings=(),
                    selected_read_ids=(),
                    unselected_positive_read_ids=("sales",),
                ),
            ),
            selected_read_ids=(),
        ),
    )

    payload = raw_payload_module._raw_source_binding_candidate_payload(request)

    assert payload["requested_fact_sources"] == [
        {"requested_fact_id": "fact_1", "source_contexts": []}
    ]
    assert "utility_source_candidates" not in payload


def test_source_binding_registry_candidates_use_model_visible_support_sets():
    request = _request_with_optional_params()
    registry = source_candidate_registry(request)
    prompt_candidate = _only_source_candidate(registry.prompt_payload)
    parser_candidate = registry.candidates_by_id[
        prompt_candidate["source_candidate_id"]
    ]
    prompt_support_sets = _binding_surface(prompt_candidate)["fulfillment_support_sets"]
    parser_support_sets = parser_candidate.payload["fulfillment_support_sets"]

    assert set(registry.candidates_by_id) == set(registry.prompt_candidate_ids)
    assert all("fulfillment_choice_id" in item for item in prompt_support_sets)
    assert all("fulfillment_support_set_id" not in item for item in prompt_support_sets)
    parser_visible_support_sets = [
        item for item in parser_support_sets if item.get("fulfillment_choice_id")
    ]
    assert [item["fulfillment_choice_id"] for item in parser_visible_support_sets] == [
        item["fulfillment_choice_id"] for item in prompt_support_sets
    ]
    assert all(
        "fulfillment_support_set_id" in item for item in parser_visible_support_sets
    )


def test_source_binding_targets_are_compact_role_targets_not_private_plan_variants():
    request = replace(
        _request_with_optional_params(),
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.direct_field_value.1",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.1",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(SourceStrategyMember(source_candidate_id="source_1"),),
                    basis="First private strategy.",
                ),
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.direct_field_value.2",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.2",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(SourceStrategyMember(source_candidate_id="source_1"),),
                    basis="Equivalent private strategy.",
                ),
            )
        ),
    )

    targets = SourceBindingTurnPrompt(request).binding_targets_payload()[
        "binding_targets"
    ]

    assert targets == [
        {
            "binding_target_id": "target.fact_1.direct_field_value.source_1.source",
            "requested_fact_id": "fact_1",
            "plan_shape": "direct_field_value",
            "source_candidate_id": "source_1",
            "requirement_id": "source",
            "answer_output_ids": ["answer_1"],
            "required_answer_output_ids": ["answer_1"],
        }
    ]


def test_source_binding_schema_uses_one_compact_invocation_shape():
    request = replace(
        _request_with_optional_params(),
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.direct_field_value.1",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.1",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(SourceStrategyMember(source_candidate_id="source_1"),),
                    basis="First private strategy.",
                ),
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.direct_field_value.2",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.2",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(SourceStrategyMember(source_candidate_id="source_1"),),
                    basis="Equivalent private strategy.",
                ),
            )
        ),
    )
    tool_schema = SourceBindingTurnPrompt(request).tool_contract().tool_specs[
        0
    ].input_schema
    invocation_items = _source_invocation_items_schema(tool_schema)

    assert "oneOf" not in invocation_items
    assert invocation_items["properties"]["binding_target_id"] == {
        "enum": ["target.fact_1.direct_field_value.source_1.source"]
    }


def test_read_eligibility_relevant_fields_limit_fulfillment_support_sets():
    initial_request = _request_with_optional_params(
        include_extra_evidence_field=True,
        include_secondary_metric_field=True,
    )
    initial_payload = source_candidate_discovery_payload(initial_request)
    initial_candidate = _source_candidate(
        initial_payload,
        requested_fact_id="fact_1",
        read_id="sales",
    )
    candidate_signature = read_candidate_signature(
        initial_candidate,
        requested_fact_id="fact_1",
    )
    retained_request = _request_with_optional_params(
        include_extra_evidence_field=True,
        include_secondary_metric_field=True,
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _retained_read_assessment(
                    source_candidate_id=str(initial_candidate["source_candidate_id"]),
                    source_candidate_signature=candidate_signature,
                    requested_fact_id="fact_1",
                    read_id="sales",
                    relevant_field_refs=("sales.field.amount",),
                    retention_basis=(
                        "Only the sale amount is needed for this aggregate answer."
                    ),
                ),
            )
        ),
    )

    payload = source_candidate_discovery_payload(retained_request)
    candidate = _source_candidate(
        payload,
        requested_fact_id="fact_1",
        read_id="sales",
    )

    assert _candidate_field_refs(candidate) >= {
        "sales.field.amount",
        "sales.field.status",
        "sales.field.unrelated",
        "sales.field.secondary_amount",
    }
    assert _support_set_field_refs(candidate) == {"sales.field.amount"}


def test_bound_source_prompt_payload_carries_only_answer_evidence_roles():
    payload = _bound_sources_prompt_payload(
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=AnswerPopulation(
                    population_binding_id="population_1",
                    intent_text="sales rows",
                    match_basis_explanation="The source exposes sales rows.",
                ),
                source=None,
                value_id="value_1",
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="The source exposes the answer value.",
                        group_key_evidence_ids=("source_1.root.name",),
                    ),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.root.name",
                        field_id="name",
                        type="string",
                    ),
                ),
            ),
        ),
    )

    fulfillment = payload["bound_sources"][0]["fulfills"][0]
    assert set(fulfillment) == {
        "requested_fact_id",
        "answer_output_id",
        "match_basis_explanation",
        "metric_measure_evidence_ids",
        "row_count_basis_evidence_ids",
        "scope_evidence_ids",
        "group_key_evidence_ids",
    }
    assert fulfillment["group_key_evidence_ids"] == ["source_1.root.name"]


def test_source_linked_value_usage_checks_metric_evidence():
    candidate = type(
        "Candidate",
        (),
        {"payload": {"source_field_id": "amount"}},
    )()
    bound = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=AnswerPopulation(
            population_binding_id="pop.source_1.candidate_population",
            intent_text="sales total",
            match_basis_explanation="sales total defines the answer population.",
        ),
        value_id="selected_source",
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="source_1.data.amount",
                field_id="amount",
                type="number",
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                metric_measure_evidence_ids=("source_1.data.amount",),
                match_basis_explanation="amount is the measured quantity.",
            ),
        ),
    )

    assert candidate_value_is_used_by_bound_source(candidate, bound)


def test_ranked_aggregate_source_binding_keeps_selected_group_key_lineage():
    base = _request_with_optional_params(include_extra_evidence_field=True)
    fact = replace(
        base.requested_facts[0],
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
    )
    request = replace(
        base,
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        plan_selection=_selected_plan(plan_shape="ranked_aggregate"),
    )
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_source_candidate(prompt.source_invocation_candidate_payload())
    outcome = _source_binding_outcome(
        candidate,
        binding_target_id=_only_binding_target_id(prompt),
        param_decisions={
            "start_date": _first_param_decision(candidate, "start_date"),
            "end_date": _first_param_decision(candidate, "end_date"),
        },
        finite_choice_param_reviews={
            "status": _choice_reviews(
                counts=("OPEN",),
                does_not_count=("CLOSED",),
            )
        },
        field_ids=("unrelated",),
    )
    metric_evidence_ids = source_binding_metric_evidence_ids_by_requested_fact(
        request
    )["fact_1"]
    outcome["metric_fit_bases"] = {
        "fact_1": {
            evidence_id: {
                "metric_meaning": f"{evidence_id} is a candidate metric.",
                "fit_basis": f"{evidence_id} fits the ranked answer.",
            }
            for evidence_id in metric_evidence_ids
        }
    }
    outcome["fit_basis_interpretations"] = {
        "fact_1": {
            evidence_id: {"interpretation": "FITS_REQUESTED_ANSWER"}
            for evidence_id in metric_evidence_ids
        }
    }

    result = parse_source_binding({"outcome": outcome}, request=request)

    fulfillment = result.outcome.bound_sources[0].fulfillments[0]
    assert fulfillment.group_key_evidence_ids == ("source_1.root.unrelated",)


def test_generate_source_binding_runs_one_model_turn():
    request = _request_with_optional_params()
    prompt = SourceBindingTurnPrompt(request)
    candidate = _only_source_candidate(prompt.source_invocation_candidate_payload())
    model_port = _SourceBindingModelPort(
        arguments={
            "outcome": _source_binding_outcome(
                candidate,
                binding_target_id=_only_binding_target_id(prompt),
                param_decisions={
                    "start_date": _first_param_decision(candidate, "start_date"),
                    "end_date": _first_param_decision(candidate, "end_date"),
                },
                finite_choice_param_reviews={
                    "status": _choice_reviews(
                        counts=("OPEN",),
                        does_not_count=("CLOSED",),
                    )
                },
            ),
        },
    )

    result = generate_source_binding(
        request=request,
        model_port=model_port,
        provider="stub",
        model_key="test",
        max_thinking_tokens=64,
    )

    assert model_port.tool_names == ["submit_source_binding"]
    assert result.artifact.selected_tool_name == "submit_source_binding"
    assert [subturn.artifact.selected_tool_name for subturn in result.subturns] == [
        "submit_source_binding"
    ]


def test_generate_source_binding_returns_backend_impossible_without_answer_candidates():
    base_request = _request_with_optional_params()
    request = replace(
        base_request,
        relation_catalog=RelationCatalog(),
        catalog_selection=CatalogSelectionResult(
            relation_catalog=RelationCatalog(),
            requested_fact_selections=(
                RequestedFactCatalogSelection(
                    requested_fact_id="fact_1",
                    query_terms=("sales", "today"),
                    rankings=(),
                    selected_read_ids=(),
                ),
            ),
            selected_read_ids=(),
        ),
        read_eligibility=ReadEligibilityResult(read_assessments=()),
        available_values=(),
    )
    model_port = _ExplodingSourceBindingModelPort()

    result = generate_source_binding(
        request=request,
        model_port=model_port,
        provider="stub",
        model_key="test",
        max_thinking_tokens=64,
    )

    assert isinstance(result.result.outcome, PlanImpossible)
    assert result.subturns == ()
    assert model_port.calls == []
    blocked = result.result.outcome.blocked_facts[0]
    assert blocked.requested_fact_id == "fact_1"
    assert blocked.basis == BlockedFactBasis.CATALOG_ACCESS
    assert blocked.evidence_refs == ("catalog_selection:fact_1",)


def _request_with_optional_params(
    *,
    read_eligibility: ReadEligibilityResult | None = None,
    answer_output_ids: tuple[str, ...] = ("answer_1",),
    default_choice_param: bool = False,
    response_shape_choice_param: bool = False,
    include_object_container_field: bool = False,
    include_extra_evidence_field: bool = False,
    include_secondary_metric_field: bool = False,
    include_identity_evidence_field: bool = False,
    include_many_data_row_path: bool = False,
    include_paid_at_evidence_field: bool = False,
    include_boolean_response_field: bool = False,
    include_unsupported_read: bool = False,
    include_grounded_time: bool = False,
    grounded_time_field_id: str = "",
) -> SourceBindingRequest:
    root_row_source_id = api_row_source_id("sales", "root")
    answer_outputs = tuple(
        RequestedFactAnswerOutput(
            id=answer_output_id,
            description=f"Sales output {answer_output_id}.",
        )
        for answer_output_id in answer_output_ids
    )
    fact = RequestedFact(
        id="fact_1",
        description="Sales that happened today.",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=answer_outputs,
        known_inputs=(
            RequestedFactLiteralInput(
                id="time_1",
                source=KnownInputSource.QUESTION_CONTEXT,
                text="today",
                role=LiteralInputRole.TIME_VALUE,
                resolved_value_text="today",
            ),
        ),
    )
    params = [
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
        CatalogParam(
            ref="sales.query.status",
            name="status",
            source=ParamSource.QUERY,
            type="choice",
            choices=("OPEN", "CLOSED"),
            choice_labels={"OPEN": "Open", "CLOSED": "Closed"},
        ),
    ]
    row_path_id = "data" if include_many_data_row_path else "root"

    def field_path(name: str) -> str:
        return f"data.{name}" if include_many_data_row_path else name

    fields = [
        *(
            (
                CatalogField(
                    ref="sales.field.sale_id",
                    path=field_path("sale_id"),
                    row_path_id=row_path_id,
                    type="uuid",
                    identity=IdentityMetadata(
                        entity_ref="sale",
                        identity_field="sale_id",
                        primary_key=True,
                    ),
                ),
            )
            if include_identity_evidence_field
            else ()
        ),
        CatalogField(
            ref="sales.field.amount",
            path=field_path("amount"),
            row_path_id=row_path_id,
            type="decimal",
        ),
        CatalogField(
            ref="sales.field.status",
            path=field_path("status"),
            row_path_id=row_path_id,
            type="choice",
            choices=("OPEN", "CLOSED"),
        ),
    ]
    if include_extra_evidence_field:
        fields.append(
            CatalogField(
                ref="sales.field.unrelated",
                path=field_path("unrelated"),
                row_path_id=row_path_id,
                type="string",
            )
        )
    if include_secondary_metric_field:
        fields.append(
            CatalogField(
                ref="sales.field.secondary_amount",
                path=field_path("secondary_amount"),
                row_path_id=row_path_id,
                type="decimal",
            )
        )
    if include_paid_at_evidence_field:
        fields.append(
            CatalogField(
                ref="sales.field.paid_at",
                path=field_path("paid_at"),
                row_path_id=row_path_id,
                type="datetime",
            )
        )
    if include_boolean_response_field:
        fields.append(
            CatalogField(
                ref="sales.field.is_active",
                path=field_path("is_active"),
                row_path_id=row_path_id,
                type="boolean",
            )
        )
    if include_object_container_field:
        fields.append(
            CatalogField(
                ref="sales.field.metadata",
                path=field_path("metadata"),
                row_path_id=row_path_id,
                type="object",
            )
        )
    if default_choice_param:
        params.append(
            CatalogParam(
                ref="sales.query.granularity",
                name="granularity",
                source=ParamSource.QUERY,
                type="choice",
                choices=("day", "month"),
                choice_labels={"day": "Day", "month": "Month"},
                default="day",
            )
        )
    if response_shape_choice_param:
        params.append(
            CatalogParam(
                ref="sales.query.ordering",
                name="ordering",
                source=ParamSource.QUERY,
                type="choice",
                choices=("created_at", "-created_at"),
                choice_labels={
                    "created_at": "Created At",
                    "-created_at": "Created At",
                },
                semantics="response_shape",
            )
        )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                row_paths=(
                    (
                        RowPath(
                            id="data",
                            path="data",
                            cardinality=RowCardinality.MANY,
                        ),
                    )
                    if include_many_data_row_path
                    else ()
                ),
                params=tuple(params),
                fields=tuple(fields),
            ),
            *(
                (
                    EndpointRead(
                        id="refunds",
                        endpoint_name="list_refund_list",
                        resource_names=("refund",),
                        fields=(
                            CatalogField(
                                ref="refunds.field.refund_id",
                                path="refund_id",
                                type="uuid",
                            ),
                        ),
                    ),
                )
                if include_unsupported_read
                else ()
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales", "today"),
                rankings=(
                    CatalogSelectionRanking(read_id="sales", score=10),
                    *(
                        (CatalogSelectionRanking(read_id="refunds", score=1),)
                        if include_unsupported_read
                        else ()
                    ),
                ),
                selected_read_ids=(
                    "sales",
                    *(("refunds",) if include_unsupported_read else ()),
                ),
            ),
        ),
        selected_read_ids=(
            "sales",
            *(("refunds",) if include_unsupported_read else ()),
        ),
    )
    if read_eligibility is None:
        read_eligibility = ReadEligibilityResult(
            read_assessments=(
                _retained_read_assessment(
                    source_candidate_id="source_1",
                    source_candidate_signature="source_1",
                    requested_fact_id="fact_1",
                    read_id="sales",
                ),
            )
        )
    read_eligibility = _read_eligibility_with_candidate_signatures(
        read_eligibility,
        requested_facts=(fact,),
        catalog_selection=catalog_selection,
    )
    return SourceBindingRequest(
        question="How many sales happened today?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(),
        available_values=(
            FactValue.time(
                id="time_1",
                expression="today",
                resolved_start="2026-05-22",
                resolved_end="2026-05-22",
                proof_refs=("known_input:time_1",),
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
        available_value_uses=(
            (
                GroundedInputUse(
                    id="grounded_start",
                    value_id="time_1",
                    row_source_id=root_row_source_id,
                    param_id="start_date",
                    field_id=grounded_time_field_id,
                    value_component=TimeComponent.START,
                ),
                GroundedInputUse(
                    id="grounded_end",
                    value_id="time_1",
                    row_source_id=root_row_source_id,
                    param_id="end_date",
                    field_id=grounded_time_field_id,
                    value_component=TimeComponent.END,
                ),
            )
            if include_grounded_time
            else ()
        ),
        read_eligibility=read_eligibility,
    )


def _request_with_identity_field_filter() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="Locations in London.",
        answer_subject=RequestedFactAnswerSubject(subject_text="locations"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
        known_inputs=(
            RequestedFactLiteralInput(
                id="area_1",
                source=KnownInputSource.QUESTION_CONTEXT,
                text="London",
                role=LiteralInputRole.REFERENCE_VALUE,
                resolved_value_text="London",
                value_meaning_hint="area",
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="locations",
                endpoint_name="list_locations",
                resource_names=("location",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="locations.field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="location",
                            identity_field="location_id",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                    CatalogField(
                        ref="locations.field.area_id",
                        path="data.area_id",
                        row_path_id="data",
                        type="uuid",
                        identity=IdentityMetadata(
                            entity_ref="area",
                            identity_field="area_id",
                            primary_key=True,
                            stable=True,
                        ),
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("location",),
                rankings=(CatalogSelectionRanking(read_id="locations", score=10),),
                selected_read_ids=("locations",),
            ),
        ),
        selected_read_ids=("locations",),
    )
    read_eligibility = _read_eligibility_with_candidate_signatures(
        ReadEligibilityResult(
            read_assessments=(
                _retained_read_assessment(
                    source_candidate_id="source_1",
                    source_candidate_signature="source_1",
                    requested_fact_id="fact_1",
                    read_id="locations",
                ),
            )
        ),
        requested_facts=(fact,),
        catalog_selection=catalog_selection,
    )
    return SourceBindingRequest(
        question="How many locations are in London?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(),
        available_values=(
            FactValue.identity(
                id="nairobi_area",
                identity_type="area",
                identity_field="area_id",
                value="area_nairobi",
                display_value="London",
                matched_field_ref="field.data.name",
                matched_field_path="data.name",
                proof_refs=("known_input:area_1",),
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
        read_eligibility=read_eligibility,
    )


def _request_with_metric_support(
    *,
    include_secondary_metric_field: bool = False,
) -> SourceBindingRequest:
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            _retained_read_assessment(
                source_candidate_id="source_1",
                source_candidate_signature="source_1",
                requested_fact_id="fact_1",
                read_id="sales",
            ),
        )
    )
    return _request_with_optional_params(
        read_eligibility=read_eligibility,
        include_secondary_metric_field=include_secondary_metric_field,
    )


def _request_with_reused_answer_output_metric_support() -> SourceBindingRequest:
    facts = (
        RequestedFact(
            id="fact_sales",
            description="total sales amount",
            answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
            answer_outputs=(
                RequestedFactAnswerOutput(
                    id="answer_1",
                    description="sales amount",
                ),
            ),
        ),
        RequestedFact(
            id="fact_payments",
            description="total payment amount",
            answer_subject=RequestedFactAnswerSubject(subject_text="payments"),
            answer_outputs=(
                RequestedFactAnswerOutput(
                    id="answer_1",
                    description="payment amount",
                ),
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
            EndpointRead(
                id="payments",
                endpoint_name="list_payment_list",
                resource_names=("payments",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="payments.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_sales",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
            RequestedFactCatalogSelection(
                requested_fact_id="fact_payments",
                query_terms=("payments",),
                rankings=(CatalogSelectionRanking(read_id="payments", score=10),),
                selected_read_ids=("payments",),
            ),
        ),
        selected_read_ids=("sales", "payments"),
    )
    scopes = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Compare sales and payment totals.",
            question_contract=QuestionContract(requested_facts=facts),
            requested_facts=facts,
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes
    scopes_by_fact_read = {
        (scope.requested_fact_id, scope.read_id): scope for scope in scopes
    }
    sales_scope = scopes_by_fact_read[("fact_sales", "sales")]
    payments_scope = scopes_by_fact_read[("fact_payments", "payments")]
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            _retained_read_assessment(
                source_candidate_id=sales_scope.source_candidate_id,
                source_candidate_signature=sales_scope.source_candidate_signature,
                requested_fact_id="fact_sales",
                read_id="sales",
            ),
            _retained_read_assessment(
                source_candidate_id=payments_scope.source_candidate_id,
                source_candidate_signature=payments_scope.source_candidate_signature,
                requested_fact_id="fact_payments",
                read_id="payments",
            ),
        )
    )
    return SourceBindingRequest(
        question="Compare sales and payment totals.",
        question_contract=QuestionContract(requested_facts=facts),
        requested_facts=facts,
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plans(
            (
                ("fact_sales", sales_scope.source_candidate_id),
                ("fact_payments", payments_scope.source_candidate_id),
            )
        ),
        read_eligibility=read_eligibility,
    )


def _request_with_filtered_response_shape_variant() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales count by status",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="sales count",
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="irrelevant_read",
                endpoint_name="irrelevant_read",
                resource_names=("irrelevant",),
                fields=(
                    CatalogField(
                        ref="irrelevant.field.id",
                        path="id",
                        type="string",
                    ),
                ),
            ),
            EndpointRead(
                id="sales_summary",
                endpoint_name="sales_summary",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                params=(
                    CatalogParam(
                        ref="sales_summary.query.group_by",
                        name="group_by",
                        source=ParamSource.QUERY,
                        type="choice",
                        required=True,
                        choices=("location", "status"),
                        semantics="response_shape",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales_summary.field.label",
                        path="data.label",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="sales_summary.field.count",
                        path="data.count",
                        row_path_id="data",
                        type="integer",
                    ),
                ),
            ),
        )
    )
    original_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(
                    CatalogSelectionRanking(read_id="irrelevant_read", score=10),
                    CatalogSelectionRanking(read_id="sales_summary", score=9),
                ),
                selected_read_ids=("irrelevant_read", "sales_summary"),
            ),
        ),
        selected_read_ids=("irrelevant_read", "sales_summary"),
    )
    original_surface = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales by status?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=original_selection,
            conversation_context={},
            available_values=(),
        )
    )
    original_cards = original_surface.card_payload
    status_card = next(
        card
        for group in original_cards["requested_fact_read_candidates"]
        for card in group["read_candidates"]
        if card["read_id"] == "sales_summary"
        and card["bound_params"][0]["value"] == "status"
    )
    status_signature = next(
        scope.source_candidate_signature
        for scope in original_surface.candidate_scopes
        if scope.source_candidate_id == status_card["source_candidate_id"]
    )
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            _retained_read_assessment(
                source_candidate_id=status_card["source_candidate_id"],
                source_candidate_signature=status_signature,
                requested_fact_id="fact_1",
                read_id="sales_summary",
            ),
        )
    )
    filtered_catalog = RelationCatalog(reads=(catalog.read("sales_summary"),))
    filtered_selection = CatalogSelectionResult(
        relation_catalog=filtered_catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales_summary", score=9),),
                selected_read_ids=("sales_summary",),
                unselected_positive_read_ids=("irrelevant_read",),
            ),
        ),
        selected_read_ids=("sales_summary",),
    )
    return SourceBindingRequest(
        question="How many sales by status?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=filtered_catalog,
        catalog_selection=filtered_selection,
        plan_selection=_selected_plan(
            source_candidate_ids=(status_card["source_candidate_id"],)
        ),
        read_eligibility=read_eligibility,
    )


def _request_with_source_default_param_after_read_eligibility() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales summary",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="sales summary",
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales_summary",
                endpoint_name="sales_summary",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                params=(
                    CatalogParam(
                        ref="sales_summary.query.group_by",
                        name="group_by",
                        source=ParamSource.QUERY,
                        type="choice",
                        required=True,
                        choices=("date", "location"),
                        default="date",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales_summary.field.label",
                        path="data.label",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="sales_summary.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales",),
                rankings=(CatalogSelectionRanking(read_id="sales_summary", score=10),),
                selected_read_ids=("sales_summary",),
            ),
        ),
        selected_read_ids=("sales_summary",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Show sales summary.",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
            available_values=(),
        )
    ).candidate_scopes[0]
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            _retained_read_assessment(
                source_candidate_id=scope.source_candidate_id,
                source_candidate_signature=scope.source_candidate_signature,
                requested_fact_id="fact_1",
                read_id="sales_summary",
            ),
        )
    )
    return SourceBindingRequest(
        question="Show sales summary.",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(
            source_candidate_ids=(scope.source_candidate_id,)
        ),
        read_eligibility=read_eligibility,
    )


def _read_eligibility_with_candidate_signatures(
    read_eligibility: ReadEligibilityResult,
    *,
    requested_facts: tuple[RequestedFact, ...],
    catalog_selection: CatalogSelectionResult,
) -> ReadEligibilityResult:
    scopes_by_candidate_id = {
        scope.source_candidate_id: scope
        for scope in read_eligibility_candidate_surface(
            ReadEligibilityRequest(
                question="How many sales happened today?",
                question_contract=QuestionContract(requested_facts=requested_facts),
                requested_facts=requested_facts,
                catalog_selection=catalog_selection,
                conversation_context={},
                available_values=(),
            )
        ).candidate_scopes
    }

    def candidate_signature(assessment: ReadAssessment) -> str:
        scope = scopes_by_candidate_id.get(assessment.source_candidate_id)
        return (
            scope.source_candidate_signature
            if scope is not None
            else assessment.source_candidate_signature
        )

    def relevant_field_refs(assessment: ReadAssessment) -> tuple[str, ...]:
        if assessment.relevant_field_refs or not assessment.is_retained:
            return assessment.relevant_field_refs
        scope = scopes_by_candidate_id.get(assessment.source_candidate_id)
        if scope is None:
            return ()
        return tuple(scope.field_refs_by_evidence_token.values())

    return ReadEligibilityResult(
        read_assessments=tuple(
            ReadAssessment(
                source_candidate_id=assessment.source_candidate_id,
                source_candidate_signature=candidate_signature(assessment),
                requested_fact_id=assessment.requested_fact_id,
                read_id=assessment.read_id,
                relevant_row_path_ids=assessment.relevant_row_path_ids,
                relevant_field_refs=relevant_field_refs(assessment),
                retention_basis=assessment.retention_basis,
                retention_decision=assessment.retention_decision,
            )
            for assessment in read_eligibility.read_assessments
        )
    )


def _source_binding_outcome(
    candidate: dict[str, object],
    *,
    binding_target_id: str,
    param_decisions: dict[str, dict[str, object]],
    finite_choice_param_reviews: dict[str, dict[str, object]],
    row_predicate_reviews: dict[str, dict[str, object]] | None = None,
    field_ids: tuple[str, ...] = ("amount",),
) -> dict[str, object]:
    fulfillment_decisions = source_fulfills_for_candidate(
        candidate,
        field_ids=field_ids,
    )
    metric_fit_contract = _metric_fit_contract_for_candidate(
        candidate,
        requested_fact_id="fact_1",
        fulfillment_decisions=fulfillment_decisions,
    )
    return {
        "kind": "source_bindings",
        **metric_fit_contract,
        "source_invocations": [
            {
                "binding_target_id": binding_target_id,
                "answer_population": {
                    "population_binding_id": _binding_surface(candidate)[
                        "population_bindings"
                    ][0]["population_binding_id"],
                    "intent_text": "sales that happened today",
                    "match_basis_explanation": "The requested fact asks for today's sales.",
                },
                "fulfillment_decisions": fulfillment_decisions,
                "param_decisions": {
                    param_id: _param_decision(param_id, decision)
                    for param_id, decision in param_decisions.items()
                },
                "row_predicate_reviews": dict(row_predicate_reviews or {}),
                "finite_choice_param_reviews": finite_choice_param_reviews,
            }
        ],
    }


def _metric_fit_contract_for_candidate(
    candidate: dict[str, object],
    *,
    requested_fact_id: str,
    fulfillment_decisions: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    support_sets = _binding_surface(candidate).get("fulfillment_support_sets") or ()
    bases: dict[str, dict[str, dict[str, str]]] = {}
    interpretations: dict[str, dict[str, dict[str, str]]] = {}
    selected_choice_ids = {
        str(decision.get("fulfillment_choice_id") or "")
        for decision in fulfillment_decisions.values()
        if isinstance(decision, dict)
    }
    for support_set in support_sets:
        if not isinstance(support_set, dict):
            continue
        choice_id = str(support_set.get("fulfillment_choice_id") or "")
        if choice_id not in selected_choice_ids:
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            for item in slot.get("metric_measure_evidence") or ():
                if not isinstance(item, dict):
                    continue
                evidence_id = str(item.get("evidence_id") or "")
                if not evidence_id:
                    continue
                bases.setdefault(requested_fact_id, {})[evidence_id] = {
                    "metric_meaning": f"{evidence_id} is the selected metric evidence.",
                    "fit_basis": (
                        f"{evidence_id} is treated as fitting the requested answer "
                        "in this test fixture."
                    ),
                }
                interpretations.setdefault(requested_fact_id, {})[evidence_id] = {
                    "interpretation": "FITS_REQUESTED_ANSWER",
                }
    return {
        "metric_fit_bases": bases,
        "fit_basis_interpretations": interpretations,
    }


def _set_metric_fit_interpretations(
    outcome: dict[str, object],
    *,
    decision: str,
    evidence_id: str | None = None,
) -> None:
    metric_fit_bases = outcome.setdefault("metric_fit_bases", {})
    if not isinstance(metric_fit_bases, dict):
        raise AssertionError("metric_fit_bases must be an object")
    basis_fact = metric_fit_bases.setdefault("fact_1", {})
    if not isinstance(basis_fact, dict):
        raise AssertionError("metric_fit_bases.fact_1 must be an object")
    evidence_ids = (evidence_id,) if evidence_id is not None else tuple(basis_fact)
    if not evidence_ids:
        evidence_ids = ("source_1.root.amount",)
    interpretations = outcome.setdefault("fit_basis_interpretations", {})
    if not isinstance(interpretations, dict):
        raise AssertionError("fit_basis_interpretations must be an object")
    interpretation_fact = interpretations.setdefault("fact_1", {})
    if not isinstance(interpretation_fact, dict):
        raise AssertionError("fit_basis_interpretations.fact_1 must be an object")
    for item_evidence_id in evidence_ids:
        basis_fact[item_evidence_id] = {
            "metric_meaning": "amount is the sales amount field.",
            "fit_basis": "The sales amount field is compared to the requested sales amount.",
        }
        interpretation_fact[item_evidence_id] = {"interpretation": decision}


def _choice_reviews(
    *,
    counts: tuple[str, ...] = (),
    does_not_count: tuple[str, ...] = (),
    test_ids: tuple[str, ...] = ("subject_identity", "normal_instance_guard"),
    no_decision_test_ids: tuple[str, ...] = (),
    include_normal_guard_result: bool = True,
) -> dict[str, object]:
    effects = {
        **{value: "SATISFIES_TEST" for value in counts},
        **{value: "CONFLICTS_WITH_TEST" for value in does_not_count},
    }
    test_id_set = set(test_ids)
    return {
        "controlled_population_role_id": "role_1",
        "role_selection_basis": "status controls sales rows being counted.",
        "population_test_basis": _population_test_basis(),
        "choice_reviews": [
            {
                "choice_option_id": value,
                "choice_domain_meaning": f"{value.lower()} sales",
                "choice_inclusion_basis": f"{value.lower()} is reviewed for inclusion.",
                "choice_inclusion": (
                    "EXCLUDE" if effects[value] == "CONFLICTS_WITH_TEST" else "INCLUDE"
                ),
                "population_test_results": _choice_population_test_results_by_id(
                    value=value,
                    test_effect=effects[value],
                    no_decision_test_ids=no_decision_test_ids,
                    include_normal_guard_result=include_normal_guard_result,
                    test_id_set=test_id_set,
                ),
            }
            for value in ("OPEN", "CLOSED")
            if value in effects
        ],
    }


def _choice_population_test_results_by_id(
    *,
    value: str,
    test_effect: str,
    no_decision_test_ids: tuple[str, ...],
    include_normal_guard_result: bool,
    test_id_set: set[str],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    no_decision = set(no_decision_test_ids)
    for test_id, item in _choice_population_test_results(
        value=value,
        test_effect=test_effect,
        include_normal_guard_result=include_normal_guard_result,
    ):
        if test_id not in test_id_set:
            continue
        result = dict(item)
        if test_id in no_decision:
            result["population_consequence"] = (
                f"The {value.lower()} choice does not decide this test."
            )
            result["test_effect"] = "DOES_NOT_DECIDE_TEST"
        output[test_id] = result
    return output


def _choice_population_test_results(
    *,
    value: str,
    test_effect: str,
    include_normal_guard_result: bool = True,
) -> tuple[tuple[str, dict[str, object]], ...]:
    effect_text = "satisfies" if test_effect == "SATISFIES_TEST" else "conflicts with"
    normal_guard: dict[str, object] = {
        "role_match_basis": (
            f"The {value.lower()} choice {effect_text} the normal instance test."
        ),
        "explicit_user_override_evidence": [],
        "explicit_user_override_applies": False,
        "population_consequence": (
            f"The {value.lower()} choice {effect_text} the normal instance test."
        ),
        "disposition": {
            "matched_excluded_role": (
                "NONE"
                if test_effect == "SATISFIES_TEST"
                else NormalInstanceExcludedStateRole.CANCELED_OR_VOIDED.value
            ),
            "test_effect": test_effect,
        },
    }
    if include_normal_guard_result:
        guard_fields = _normal_instance_guard_fields(
            choice=value,
            matching_roles=(
                ()
                if test_effect == "SATISFIES_TEST"
                else (NormalInstanceExcludedStateRole.CANCELED_OR_VOIDED,)
            ),
        )
        normal_guard.update(guard_fields)
    return (
        (
            "subject_identity",
            {
                "test_basis": (
                    f"The {value.lower()} choice {effect_text} the requested answer "
                    "population test."
                ),
                "population_consequence": (
                    f"The {value.lower()} choice {effect_text} the requested answer "
                    "population test."
                ),
                "test_effect": test_effect,
            },
        ),
        ("normal_instance_guard", normal_guard),
    )


def _population_test_basis() -> dict[str, dict[str, str]]:
    return {
        "subject_identity": {
            "test_question": "Does the row/value represent sales?",
            "role_scoped_test_question": (
                "For sales rows being counted, does this status value represent sales?"
            ),
        },
        "normal_instance_guard": {
            "test_question": (
                "Is this an ordinary business instance of sales as normally "
                "understood in business operations and reporting?"
            ),
            "role_scoped_test_question": (
                "For sales rows being counted, is this status value an ordinary "
                "business instance of sales?"
            ),
        },
    }


def _normal_instance_guard_fields(
    *,
    choice: str,
    matching_roles: tuple[NormalInstanceExcludedStateRole, ...] = (),
    unknown_role_match: bool = False,
    override_evidence: tuple[dict[str, str], ...] = (),
    explicit_user_override_applies: bool = False,
) -> dict[str, object]:
    label = choice.capitalize()
    if len(matching_roles) > 1:
        raise ValueError("test helper supports one matched role")
    if unknown_role_match and matching_roles:
        raise ValueError("test helper cannot set unknown and matched role")
    matched_role = (
        "UNKNOWN"
        if unknown_role_match
        else (matching_roles[0].value if matching_roles else "NONE")
    )
    return {
        "role_match_basis": f"{label} was compared to the excluded normal-instance roles.",
        "explicit_user_override_evidence": list(override_evidence),
        "explicit_user_override_applies": explicit_user_override_applies,
        "disposition": {
            "matched_excluded_role": matched_role,
            "test_effect": (
                "UNKNOWN_TEST_EFFECT"
                if unknown_role_match
                else ("CONFLICTS_WITH_TEST" if matching_roles else "SATISFIES_TEST")
            ),
        },
    }


def _param_decision(param_id: str, decision: dict[str, object]) -> dict[str, object]:
    return {
        "population_intent": "sales that happened today",
        "match_basis_explanation": f"{param_id} matches the selected source invocation.",
        "param_decision_id": decision["param_decision_id"],
    }


def _first_param_decision(
    candidate: dict[str, object],
    param_id: str,
) -> dict[str, object]:
    param = _param(candidate, param_id)
    return param["decision_options"][0]


def _param(candidate: dict[str, object], param_id: str) -> dict[str, object]:
    for param in _binding_surface(candidate).get("params", ()):
        if isinstance(param, dict) and param.get("param_id") == param_id:
            return param
    raise AssertionError(f"missing param {param_id}")


def _only_source_candidate(payload: dict[str, object]) -> dict[str, object]:
    candidates = _source_options(payload)
    for candidate in candidates:
        params = {
            str(item.get("param_id") or "")
            for item in _binding_surface(candidate).get("params", ())
            if isinstance(item, dict)
        }
        if "status" in params:
            return candidate
    raise AssertionError(
        f"expected source option with status param, got {candidates!r}"
    )


def _source_candidate(
    payload: dict[str, object],
    *,
    requested_fact_id: str,
    read_id: str,
) -> dict[str, object]:
    for source in payload["requested_fact_sources"]:
        if not isinstance(source, dict):
            continue
        if source.get("requested_fact_id") != requested_fact_id:
            continue
        for context in source.get("source_contexts", ()):
            if not isinstance(context, dict):
                continue
            for candidate in context.get("source_options", ()):
                if isinstance(candidate, dict) and candidate.get("read_id") == read_id:
                    return candidate
    raise AssertionError(f"missing source candidate: {requested_fact_id}/{read_id}")


def _source_invocation_for_metric_candidate(
    candidate: dict[str, object],
    *,
    requested_fact_id: str,
    binding_target_id: str,
) -> dict[str, object]:
    return {
        "binding_target_id": binding_target_id,
        "answer_population": {
            "population_binding_id": _binding_surface(candidate)["population_bindings"][
                0
            ]["population_binding_id"],
            "intent_text": f"{requested_fact_id} rows",
            "match_basis_explanation": f"{requested_fact_id} uses this source.",
        },
        "fulfillment_decisions": source_fulfills_for_candidate(
            candidate,
            field_ids=("amount",),
        ),
        "param_decisions": {},
        "row_predicate_reviews": {},
        "finite_choice_param_reviews": {},
    }


def _only_binding_target_id(prompt: SourceBindingTurnPrompt) -> str:
    targets = prompt.transport_context_payload()["binding_targets"]
    assert isinstance(targets, list)
    assert len(targets) == 1
    return str(targets[0]["binding_target_id"])


def _source_options(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        candidate
        for source in payload["requested_fact_sources"]
        if isinstance(source, dict)
        for context in source.get("source_contexts", ())
        if isinstance(context, dict)
        for candidate in context.get("source_options", ())
        if isinstance(candidate, dict)
    ]


def _candidate_field_refs(candidate: dict[str, object]) -> set[str]:
    return {
        str(item.get("field_ref") or "")
        for grain in candidate.get("result_grains") or ()
        if isinstance(grain, dict)
        for item in grain.get("evidence_items") or ()
        if isinstance(item, dict) and str(item.get("field_ref") or "")
    }


def _support_set_field_refs(candidate: dict[str, object]) -> set[str]:
    return {
        str(item.get("field_ref") or "")
        for support_set in candidate.get("fulfillment_support_sets") or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for key in (
            "group_key_evidence",
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "scope_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict) and str(item.get("field_ref") or "")
    }


def _binding_surface(candidate: dict[str, object]) -> dict[str, object]:
    surface = candidate.get("binding_surface")
    if isinstance(surface, dict):
        return surface
    if candidate.get("kind") in {"new_api_read", "same_scope_api_read"}:
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
    raise AssertionError("source candidate missing binding surface")


def _retained_read_assessment(
    *,
    source_candidate_id: str = "source_1",
    source_candidate_signature: str = "source_1",
    requested_fact_id: str = "fact_1",
    read_id: str = "sales",
    relevant_row_path_ids: tuple[str, ...] = ("root",),
    relevant_field_refs: tuple[str, ...] = (),
    retention_basis: str = "The read exposes evidence that may be useful later.",
) -> ReadAssessment:
    return ReadAssessment(
        source_candidate_id=source_candidate_id,
        source_candidate_signature=source_candidate_signature,
        requested_fact_id=requested_fact_id,
        read_id=read_id,
        relevant_row_path_ids=relevant_row_path_ids,
        relevant_field_refs=relevant_field_refs,
        retention_basis=retention_basis,
        retention_decision="RETAIN",
    )


def _source_invocation_items_schema(schema: dict[str, object]) -> dict[str, object]:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            source_invocations = properties.get("source_invocations")
            if isinstance(source_invocations, dict):
                items = source_invocations.get("items")
                if isinstance(items, dict):
                    return items
        for value in schema.values():
            if isinstance(value, dict):
                try:
                    return _source_invocation_items_schema(value)
                except AssertionError:
                    pass
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        try:
                            return _source_invocation_items_schema(item)
                        except AssertionError:
                            pass
    raise AssertionError("missing source_invocations items schema")


def _first_fulfillment_decisions_schema(schema: dict[str, object]) -> dict[str, object]:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            fulfillment = properties.get("fulfillment_decisions")
            if isinstance(fulfillment, dict):
                return fulfillment
        for value in schema.values():
            if isinstance(value, dict):
                try:
                    return _first_fulfillment_decisions_schema(value)
                except AssertionError:
                    pass
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        try:
                            return _first_fulfillment_decisions_schema(item)
                        except AssertionError:
                            pass
    raise AssertionError("missing fulfillment_decisions schema")


class _SourceBindingModelPort:
    def __init__(
        self,
        *,
        arguments: dict[str, object],
    ) -> None:
        self.arguments = arguments
        self.calls: list[dict[str, object]] = []

    @property
    def tool_names(self) -> list[str]:
        return [
            str(call["tool_specs"][0].name)
            for call in self.calls
            if call.get("tool_specs")
        ]

    def generate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        tool_specs = kwargs.get("tool_specs")
        if not isinstance(tool_specs, tuple) or not tool_specs:
            raise AssertionError("expected tool specs")
        tool_name = tool_specs[0].name
        if tool_name != "submit_source_binding":
            raise AssertionError(f"unexpected source binding tool: {tool_name}")
        return {
            "answer": json.dumps({"tool": tool_name, "arguments": self.arguments}),
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }


class _ExplodingSourceBindingModelPort:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        raise AssertionError("source binding model should not be called")
