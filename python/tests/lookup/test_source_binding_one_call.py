import json
from dataclasses import replace

import pytest

from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityReference,
    EntityReferenceComponent,
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
from fervis.lookup.answer_program.values import (
    FactValue,
    TimeComponent,
)
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    KnownInputSource,
    LiteralInputRole,
    NormalInstanceExcludedStateRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    ResultSelectionKind,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactLiteralInput,
)
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
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
    CandidateKeyEvidence,
    EntityEvidenceComponent,
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
    source_binding_metric_fit_surface_payload,
)
from fervis.lookup.source_binding.candidates.bound_payload import (
    _bound_sources_prompt_payload,
)
from fervis.lookup.source_binding.candidates import (
    raw_payload as raw_payload_module,
)
from fervis.lookup.source_binding.candidates.fulfillment_slots import (
    FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE,
    _candidate_with_fulfillment_slots,
)
from fervis.lookup.source_binding.candidates.evidence import (
    _candidate_with_evidence_items,
)
from fervis.lookup.source_binding.candidates import (
    SourceCandidate,
    source_candidate_registry,
)
from fervis.lookup.source_binding.candidates.registry import (
    source_candidate_discovery_payload,
)
from fervis.lookup.source_binding.parser.candidate_access import (
    candidate_value_is_used_by_bound_source,
)
from fervis.lookup.source_binding.parser import parse_source_binding
from fervis.lookup.turn_prompts.projections import source_binding_candidates_xml
from tests.lookup.source_binding_helpers import (
    source_binding_request,
    source_fulfills_for_candidate,
    source_fulfills_keys_for_candidate,
)


def _candidate_with_evidence_and_fulfillment_slots(
    candidate: dict[str, object],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> dict[str, object]:
    return _candidate_with_fulfillment_slots(
        _candidate_with_evidence_items(candidate),
        requested_facts=requested_facts,
    )


def test_fulfillment_role_dispatch_covers_answer_output_support_roles():
    assert set(FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE) == set(
        ANSWER_OUTPUT_SUPPORT_ROLE_VALUES
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


def test_source_candidate_discovery_offers_source_with_grounded_named_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    base_request = _request_with_optional_params()
    location_input = RequestedFactLiteralInput(
        id="location_1",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="Nairobi",
        resolved_value_text="Nairobi",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    fact = replace(
        base_request.requested_facts[0],
        answer_population=RequestedFactAnswerPopulation(
            population_label="stores in Nairobi",
            counted_unit="stores",
            membership_tests=(
                RequestedFactAnswerPopulationMembershipTest(
                    id="subject_identity",
                    kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Does this row represent a store?",
                ),
                RequestedFactAnswerPopulationMembershipTest(
                    id="location_constraint",
                    kind=(
                        AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT
                    ),
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Is the store in the requested location?",
                    owned_question_input_refs=("location_1",),
                ),
            ),
        ),
        known_inputs=(location_input,),
        input_refs=("location_1",),
    )
    location_value = FactValue.named(
        id="grounded_location_1",
        known_input_id="location_1",
        text="Nairobi",
        matched_field_ref="field.location.area_name",
        applies_to_requested_fact_ids=("fact_1",),
    )
    request = replace(
        base_request,
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        available_values=(location_value,),
        available_value_uses=(),
        read_eligibility=None,
    )
    candidates = (
        {
            "read_id": "list_store_list",
            "row_source_id": "api:list_store_list:root",
            "row_path_id": "root",
            "read_row_source_count": 1,
            "fields": [{"field_id": "store_id", "type": "uuid"}],
            "result_grains": [],
        },
        {
            "read_id": "list_location_list",
            "row_source_id": "api:list_location_list:root",
            "row_path_id": "root",
            "read_row_source_count": 1,
            "fields": [{"field_id": "location_id", "type": "uuid"}],
            "result_grains": [],
            "applied_filters": [
                {
                    "kind": "named",
                    "known_input_id": "location_1",
                    "value_id": "grounded_location_1",
                    "field_ids": ["area_name"],
                    "matched_field_ref": "field.location.area_name",
                    "display_value": "Nairobi",
                }
            ],
        },
    )
    monkeypatch.setattr(
        raw_payload_module,
        "available_relation_catalog_payload",
        lambda *args, **kwargs: {
            "requested_fact_relations": [
                {
                    "requested_fact_id": "fact_1",
                    "query_terms": ["stores", "Nairobi"],
                    "available_relations": list(candidates),
                }
            ]
        },
    )

    payload = raw_payload_module._raw_source_binding_candidate_payload(request)

    assert [candidate["read_id"] for candidate in _source_options(payload)] == [
        "list_location_list"
    ]


def test_source_binding_registry_candidates_use_model_visible_support_sets():
    request = _request_with_optional_params()
    registry = source_candidate_registry(request)
    prompt_candidate = _only_source_candidate(registry.prompt_payload)
    parser_candidate = registry.candidates_by_id[
        prompt_candidate["source_candidate_id"]
    ]
    prompt_support_sets = _binding_surface(prompt_candidate)["fulfillment_support_sets"]
    parser_support_sets = parser_candidate.fulfillment_support_sets

    assert set(registry.candidates_by_id) == set(registry.prompt_candidate_ids)
    assert all("fulfillment_choice_id" in item for item in prompt_support_sets)
    assert all("fulfillment_support_set_id" not in item for item in prompt_support_sets)
    parser_visible_support_sets = [
        item for item in parser_support_sets if item.fulfillment_choice_id
    ]
    assert [item.fulfillment_choice_id for item in parser_visible_support_sets] == [
        item["fulfillment_choice_id"] for item in prompt_support_sets
    ]
    assert all(item.fulfillment_support_set_id for item in parser_visible_support_sets)


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
                    source_members=(
                        SourceStrategyMember(source_candidate_id="source_1"),
                    ),
                    basis="First private strategy.",
                ),
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.direct_field_value.2",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.2",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        SourceStrategyMember(source_candidate_id="source_1"),
                    ),
                    basis="Equivalent private strategy.",
                ),
            )
        ),
    )

    targets = _binding_targets(SourceBindingTurnPrompt(request))

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
                    source_members=(
                        SourceStrategyMember(source_candidate_id="source_1"),
                    ),
                    basis="First private strategy.",
                ),
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.direct_field_value.2",
                    requested_fact_id="fact_1",
                    source_strategy_id="source_strategy.fact_1.direct_field_value.2",
                    plan_shape="direct_field_value",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        SourceStrategyMember(source_candidate_id="source_1"),
                    ),
                    basis="Equivalent private strategy.",
                ),
            )
        ),
    )
    tool_schema = (
        SourceBindingTurnPrompt(request).tool_contract().tool_specs[0].input_schema
    )
    invocation_items = _source_invocation_schema(tool_schema)

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
                        entity_evidence=CandidateKeyEvidence(
                            evidence_id="source_1.root.key.staff_key",
                            key_id="staff_key",
                            entity_kind="staff",
                            components=(
                                EntityEvidenceComponent(
                                    component_id="staff_id",
                                    field_evidence_id="source_1.root.name",
                                    field_id="name",
                                ),
                            ),
                            row_source_id="read.staff.root",
                            row_path_id="root",
                        ),
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
        "value_evidence_ids",
        "row_count_basis_evidence_ids",
        "entity_evidence",
    }
    assert fulfillment["entity_evidence"] == {
        "type": "candidate_key",
        "key_id": "staff_key",
        "entity_kind": "staff",
        "components": [
            {
                "component_id": "staff_id",
                "field_evidence_id": "source_1.root.name",
                "field_id": "name",
            }
        ],
    }


def test_source_linked_value_usage_checks_metric_evidence():
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="value",
        source_field_id="amount",
    )
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
    base = _request_with_optional_params(
        include_extra_evidence_field=True,
        include_identity_evidence_field=True,
        include_many_data_row_path=True,
    )
    fact = replace(
        base.requested_facts[0],
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
            selection_kind=ResultSelectionKind.LIMITED_RESULTS,
            limit_input_ref="limit",
        ),
    )
    request = replace(
        base,
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        plan_selection=PlanSelectionSet(
            plan_selections=(
                replace(
                    _selected_plan(plan_shape="ranked_aggregate").plan_selections[0],
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id="source_1",
                            requirement_ids=("operation",),
                        ),
                    ),
                ),
            )
        ),
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
        key_ids_by_answer_output={"answer_1": "sale_key"},
    )
    metric_evidence_ids = source_binding_metric_evidence_ids_by_requested_fact(request)[
        "fact_1"
    ]
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
    assert fulfillment.entity_evidence is not None
    assert fulfillment.entity_evidence.key_id == "sale_key"
    assert tuple(
        component.field_id for component in fulfillment.entity_evidence.components
    ) == ("sale_id",)


def test_ranked_aggregate_source_binding_choices_use_canonical_group_identity():
    request = _ranked_staff_compensation_request()
    expected_choice_id = "source_1.data.reference.compensation_staff"
    candidate = source_candidate_registry(request).candidates_by_id["source_1"]
    selected_support_set = next(
        support_set
        for support_set in candidate.fulfillment_support_sets
        if support_set.fulfillment_choice_id == expected_choice_id
    )
    selected_plan = request.plan_selection.plan_selections[0]
    selected_member = replace(
        selected_plan.source_members[0],
        fulfillment_support_set_ids=(
            selected_support_set.fulfillment_support_set_id,
        ),
    )
    request = replace(
        request,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                replace(selected_plan, source_members=(selected_member,)),
            )
        ),
    )
    prompt = SourceBindingTurnPrompt(request)
    candidate_payload = prompt.source_invocation_candidate_payload()
    candidate = _source_candidate(
        candidate_payload,
        requested_fact_id="fact_1",
        read_id="shift_compensation",
    )
    support_sets = _binding_surface(candidate).get("fulfillment_support_sets") or ()
    choice_ids = tuple(
        support_set.get("fulfillment_choice_id")
        for support_set in support_sets
        if isinstance(support_set, dict)
        and support_set.get("answer_output_id") == "answer_1"
    )
    candidate_prompt = source_binding_candidates_xml(candidate_payload)

    assert choice_ids == (expected_choice_id,)
    assert f'<choice id="{expected_choice_id}"' in candidate_prompt
    assert "data.staff_name" in candidate_prompt
    assert '<choice id="source_1.data.staff_name"' not in candidate_prompt
    assert '<choice id="source_1.data.location_name"' not in candidate_prompt


def test_ranked_aggregate_source_binding_exposes_declared_entity_evidence():
    request = _ranked_store_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    candidate_payload = prompt.source_invocation_candidate_payload()
    candidate = _source_candidate(
        candidate_payload,
        requested_fact_id="fact_1",
        read_id="sales",
    )
    support_sets = _binding_surface(candidate).get("fulfillment_support_sets") or ()
    choice_ids = tuple(
        support_set.get("fulfillment_choice_id")
        for support_set in support_sets
        if isinstance(support_set, dict)
        and support_set.get("answer_output_id") == "answer_1"
    )
    candidate_prompt = source_binding_candidates_xml(candidate_payload)

    assert "source_1.data.reference.sale_location" in choice_ids
    assert "source_1.data.key.sale_key" in choice_ids
    assert '<choice id="source_1.data.reference.sale_location"' in candidate_prompt
    assert '<choice id="source_1.data.key.sale_key"' in candidate_prompt
    assert '<choice id="source_1.data.location_name"' not in candidate_prompt


def test_row_population_metric_fit_surface_is_backend_owned_count_basis():
    base = _request_with_optional_params(
        read_eligibility=ReadEligibilityResult(
            read_assessments=(
                _retained_read_assessment(
                    relevant_row_path_ids=("data",),
                    relevant_field_refs=("sales.field.status",),
                ),
            )
        ),
        include_many_data_row_path=True,
        include_secondary_metric_field=True,
    )
    fact = replace(
        base.requested_facts[0],
        description="sales count",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="sales count",
                role="ROW_COUNT",
            ),
        ),
    )
    request = replace(
        base,
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        plan_selection=_selected_plan(plan_shape="aggregate_scalar"),
    )

    surface = source_binding_metric_fit_surface_payload(request)
    requested_fact_surfaces = surface["requested_fact_metric_fit_surface"]
    metric_candidates = requested_fact_surfaces[0]["metric_candidates"]
    metric_evidence_ids = tuple(
        candidate["metric_evidence_id"] for candidate in metric_candidates
    )

    expected_row_population_id = f"row_population.{api_row_source_id('sales', 'data')}"
    assert metric_evidence_ids == (expected_row_population_id,)
    assert source_binding_metric_evidence_ids_by_requested_fact(request) == {
        "fact_1": (expected_row_population_id,),
    }


def test_row_population_metric_fit_keeps_read_scoped_summary_metric():
    request = _request_with_scoped_summary_count_metric()

    surface = source_binding_metric_fit_surface_payload(request)
    metric_candidates = surface["requested_fact_metric_fit_surface"][0][
        "metric_candidates"
    ]
    metric_evidence_ids = tuple(
        candidate["metric_evidence_id"] for candidate in metric_candidates
    )

    expected_row_population_id = (
        f"row_population.{api_row_source_id('sales_summary', 'data')}"
    )
    assert metric_evidence_ids == (
        expected_row_population_id,
        "source_1.summary.total_count",
    )
    assert "source_1.data.item_count" not in metric_evidence_ids


def test_row_population_metric_fit_ids_do_not_collapse_across_candidates():
    base = _request_with_optional_params(include_many_data_row_path=True)
    read_eligibility = ReadEligibilityResult(
        read_assessments=(
            _retained_read_assessment(
                source_candidate_id="source_1",
                source_candidate_signature="source_1",
                requested_fact_id="fact_1",
                read_id="sales",
                relevant_row_path_ids=("data",),
            ),
            _retained_read_assessment(
                source_candidate_id="source_2",
                source_candidate_signature="source_2",
                requested_fact_id="fact_1",
                read_id="returns",
                relevant_row_path_ids=("data",),
            ),
        )
    )
    returns_read = EndpointRead(
        id="returns",
        endpoint_name="list_return_list",
        resource_names=("return",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="returns.field.return_id",
                path="data.return_id",
                row_path_id="data",
                type="uuid",
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            *base.relation_catalog.reads,
            returns_read,
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("records",),
                rankings=(
                    CatalogSelectionRanking(read_id="sales", score=10),
                    CatalogSelectionRanking(read_id="returns", score=9),
                ),
                selected_read_ids=("sales", "returns"),
            ),
        ),
        selected_read_ids=("sales", "returns"),
    )
    row_count_fact = replace(
        base.requested_facts[0],
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="record count",
                role="ROW_COUNT",
            ),
        ),
    )
    request = source_binding_request(
        question=base.question,
        question_contract=QuestionContract(requested_facts=(row_count_fact,)),
        requested_facts=(row_count_fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(
            source_candidate_ids=("source_1", "source_2"),
            plan_shape="aggregate_scalar",
        ),
        read_eligibility=_read_eligibility_with_candidate_signatures(
            read_eligibility,
            requested_facts=base.requested_facts,
            catalog_selection=catalog_selection,
        ),
        available_values=base.available_values,
        available_value_uses=base.available_value_uses,
        conversation_context=base.conversation_context,
        host=base.host,
    )

    surface = source_binding_metric_fit_surface_payload(request)
    metric_candidates = surface["requested_fact_metric_fit_surface"][0][
        "metric_candidates"
    ]
    row_population_candidates = tuple(
        candidate
        for candidate in metric_candidates
        if candidate["field_type"] == "row_population"
    )
    metric_evidence_ids = tuple(
        candidate["metric_evidence_id"] for candidate in row_population_candidates
    )

    assert len(metric_evidence_ids) == 2
    assert len(set(metric_evidence_ids)) == 2


def test_measured_value_metric_fit_surface_uses_measured_fields_not_row_population():
    base = _request_with_optional_params(
        include_many_data_row_path=True,
        include_secondary_metric_field=True,
    )
    fact = replace(
        base.requested_facts[0],
        description="sales total",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="sales total",
                role="MEASURED_VALUE",
            ),
        ),
    )
    request = replace(
        base,
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        plan_selection=_selected_plan(plan_shape="aggregate_scalar"),
    )

    metric_evidence_ids = source_binding_metric_evidence_ids_by_requested_fact(request)

    assert metric_evidence_ids == {
        "fact_1": ("source_1.data.amount", "source_1.data.secondary_amount"),
    }


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
    request = source_binding_request(
        question=base_request.question,
        question_contract=base_request.question_contract,
        requested_facts=base_request.requested_facts,
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
        plan_selection=base_request.plan_selection,
        conversation_context=base_request.conversation_context,
        host=base_request.host,
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
            role="ANSWER_VALUE",
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
                candidate_keys=(
                    CandidateKey(
                        id="sale_key",
                        entity_kind="sale",
                        components=(
                            CandidateKeyComponent(
                                id="sale_id",
                                field_ref="sales.field.sale_id",
                            ),
                        ),
                        primary=True,
                    ),
                )
                if include_identity_evidence_field
                else (),
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
    return source_binding_request(
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
                granularity="day",
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
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),),
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
                    ),
                    CatalogField(
                        ref="locations.field.area_id",
                        path="data.area_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="location_key",
                        entity_kind="location",
                        components=(
                            CandidateKeyComponent(
                                id="location_id",
                                field_ref="locations.field.location_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
                entity_references=(
                    EntityReference(
                        id="location_area",
                        target_entity_kind="area",
                        target_key_id="area_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="area_id",
                                local_field_ref="locations.field.area_id",
                            ),
                        ),
                    ),
                ),
            ),
            EndpointRead(
                id="areas",
                endpoint_name="list_areas",
                resource_names=("area",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="areas.field.area_id",
                        path="data.area_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="area_key",
                        entity_kind="area",
                        components=(
                            CandidateKeyComponent(
                                id="area_id",
                                field_ref="areas.field.area_id",
                            ),
                        ),
                        primary=True,
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
    return source_binding_request(
        question="How many locations are in London?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(),
        available_values=(
            FactValue.identity(
                id="nairobi_area",
                key=entity_key_value(
                    "area", "primary_key", {"area_id": "area_nairobi"}
                ),
                display_value="London",
                matched_field_ref="field.data.name",
                matched_field_path="data.name",
                proof_refs=("known_input:area_1",),
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
        read_eligibility=read_eligibility,
    )


def _ranked_staff_compensation_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="staff member with the highest compensation this month",
        answer_subject=RequestedFactAnswerSubject(subject_text="staff"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                role="ANSWER_VALUE",
                description="staff member who earned the most compensation",
            ),
        ),
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
            selection_kind=ResultSelectionKind.LIMITED_RESULTS,
            limit_input_ref="limit",
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="shift_compensation",
                endpoint_name="list_shift_compensation_list",
                resource_names=("shift compensation",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="shift_compensation.field.shift_compensation_id",
                        path="data.shift_compensation_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="shift_compensation.field.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="shift_compensation.field.staff_name",
                        path="data.staff_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="shift_compensation.field.location_name",
                        path="data.location_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="shift_compensation.field.calculated_pay",
                        path="data.calculated_pay",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
                entity_references=(
                    EntityReference(
                        id="compensation_staff",
                        target_entity_kind="staff",
                        target_key_id="staff_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="staff_id",
                                local_field_ref="shift_compensation.field.staff_id",
                            ),
                        ),
                        context_field_refs=("shift_compensation.field.staff_name",),
                    ),
                ),
            ),
            EndpointRead(
                id="staff",
                endpoint_name="list_staff",
                resource_names=("staff",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="staff.field.staff_id",
                        path="data.staff_id",
                        row_path_id="data",
                        type="uuid",
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
                query_terms=("staff", "compensation"),
                rankings=(
                    CatalogSelectionRanking(read_id="shift_compensation", score=10),
                ),
                selected_read_ids=("shift_compensation",),
            ),
        ),
        selected_read_ids=("shift_compensation",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Which staff earned the most compensation this month?",
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
                read_id="shift_compensation",
                relevant_row_path_ids=("data",),
                relevant_field_refs=(
                    "shift_compensation.field.staff_id",
                    "shift_compensation.field.staff_name",
                    "shift_compensation.field.location_name",
                    "shift_compensation.field.calculated_pay",
                ),
            ),
        )
    )
    return source_binding_request(
        question="Which staff earned the most compensation this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(plan_shape="ranked_aggregate"),
        read_eligibility=read_eligibility,
    )


def _ranked_store_sales_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="store with the highest sales this month",
        answer_subject=RequestedFactAnswerSubject(subject_text="store"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="store selected by highest sales",
                role="ANSWER_VALUE",
            ),
        ),
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
            selection_kind=ResultSelectionKind.LIMITED_RESULTS,
            limit_input_ref="limit",
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales",
                endpoint_name="list_sale_list",
                resource_names=("sale",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="sales.field.sale_id",
                        path="data.sale_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="sales.field.location_id",
                        path="data.location_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="sales.field.location_name",
                        path="data.location_name",
                        row_path_id="data",
                        type="string",
                    ),
                    CatalogField(
                        ref="sales.field.amount",
                        path="data.amount",
                        row_path_id="data",
                        type="decimal",
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="sale_key",
                        entity_kind="sale",
                        components=(
                            CandidateKeyComponent(
                                id="sale_id",
                                field_ref="sales.field.sale_id",
                            ),
                        ),
                        primary=True,
                    ),
                ),
                entity_references=(
                    EntityReference(
                        id="sale_location",
                        target_entity_kind="location",
                        target_key_id="location_key",
                        components=(
                            EntityReferenceComponent(
                                target_component_id="location_id",
                                local_field_ref="sales.field.location_id",
                            ),
                        ),
                        context_field_refs=("sales.field.location_name",),
                    ),
                ),
            ),
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
                    ),
                ),
                candidate_keys=(
                    CandidateKey(
                        id="location_key",
                        entity_kind="location",
                        components=(
                            CandidateKeyComponent(
                                id="location_id",
                                field_ref="locations.field.location_id",
                            ),
                        ),
                        primary=True,
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
                query_terms=("store", "sales"),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Which store has the highest sales this month?",
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
                read_id="sales",
                relevant_row_path_ids=("data",),
                relevant_field_refs=(
                    "sales.field.location_id",
                    "sales.field.location_name",
                    "sales.field.amount",
                ),
            ),
        )
    )
    return source_binding_request(
        question="Which store has the highest sales this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(plan_shape="ranked_aggregate"),
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


def _request_with_scoped_summary_count_metric() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="sales count",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="sales count",
                role="ROW_COUNT",
            ),
        ),
    )
    catalog = RelationCatalog(
        reads=(
            EndpointRead(
                id="sales_summary",
                endpoint_name="list_sales_summary",
                resource_names=("sales",),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                    RowPath(
                        id="summary",
                        path="summary",
                        cardinality=RowCardinality.ONE,
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="sales_summary.field.sale_id",
                        path="data.sale_id",
                        row_path_id="data",
                        type="uuid",
                    ),
                    CatalogField(
                        ref="sales_summary.field.item_count",
                        path="data.item_count",
                        row_path_id="data",
                        type="integer",
                    ),
                    CatalogField(
                        ref="sales_summary.field.total_count",
                        path="summary.total_count",
                        row_path_id="summary",
                        type="integer",
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
            question="How many sales happened this month?",
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
                relevant_row_path_ids=("data", "summary"),
                relevant_field_refs=("sales_summary.field.total_count",),
            ),
        )
    )
    return source_binding_request(
        question="How many sales happened this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=_selected_plan(
            source_candidate_ids=(scope.source_candidate_id,),
            plan_shape="aggregate_scalar",
        ),
        read_eligibility=read_eligibility,
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
                    role="ANSWER_VALUE",
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
                    role="ANSWER_VALUE",
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
    return source_binding_request(
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
                role="ANSWER_VALUE",
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
    return source_binding_request(
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
                role="ANSWER_VALUE",
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
    return source_binding_request(
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
    key_ids_by_answer_output: dict[str, str] | None = None,
) -> dict[str, object]:
    if key_ids_by_answer_output is None:
        fulfillment_decisions = source_fulfills_for_candidate(
            candidate,
            field_ids=field_ids,
        )
    else:
        fulfillment_decisions = source_fulfills_keys_for_candidate(
            candidate,
            key_ids_by_answer_output=key_ids_by_answer_output,
        )
    metric_fit_contract = _metric_fit_contract_for_candidate(
        candidate,
        requested_fact_id="fact_1",
        fulfillment_decisions=fulfillment_decisions,
    )
    return {
        "kind": "source_bindings",
        **metric_fit_contract,
        "bindings_for_fact_1": {
            "plan_shape": binding_target_id.split(".")[2],
            binding_target_id.rsplit(".", 1)[-1]: {
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
            },
        },
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
    targets = _binding_targets(prompt)
    assert len(targets) == 1
    return str(targets[0]["binding_target_id"])


def _binding_targets(prompt: SourceBindingTurnPrompt) -> list[dict[str, object]]:
    facts = prompt.binding_plan_families_payload()["bindings_by_requested_fact"]
    assert isinstance(facts, dict)
    return [
        target
        for fact in facts.values()
        if isinstance(fact, dict)
        for shape in (fact.get("plan_shapes") or {}).values()
        if isinstance(shape, dict)
        for targets in (shape.get("role_targets") or {}).values()
        for target in targets
        if isinstance(target, dict)
    ]


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
            "entity_evidence",
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "value_evidence",
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


def _source_invocation_schema(schema: dict[str, object]) -> dict[str, object]:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            target_id = properties.get("binding_target_id")
            if isinstance(target_id, dict):
                return schema
        for value in schema.values():
            if isinstance(value, dict):
                try:
                    return _source_invocation_schema(value)
                except AssertionError:
                    pass
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        try:
                            return _source_invocation_schema(item)
                        except AssertionError:
                            pass
    raise AssertionError("missing source-binding invocation schema")


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
