from __future__ import annotations

from dataclasses import fields, replace

from types import SimpleNamespace

from typing import Any

import pytest

from jsonschema import validate

from jsonschema.exceptions import ValidationError

from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityKeyComponentTarget,
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

from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    GroupKeyDomainKind,
    KnownInputSource,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactGroupKey,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactLiteralInput,
)

from fervis.lookup.fact_plan.row_sources import api_row_source_id

from fervis.lookup.answer_program.values import FactValue, TimeComponent

from fervis.lookup.canonical_data import entity_key_value

from fervis.lookup.fact_planning.pattern_plan import compile_pattern_answer_program

from fervis.lookup.fact_planning.provider_contract import parse_pattern_answer

from fervis.lookup.provider_contract import ProviderObject

from fervis.lookup.answer_program import compiler_input_context

from fervis.lookup.grounding.model import GroundedInputUse

from fervis.lookup.question_inputs import LiteralInputRole

from fervis.lookup.read_eligibility import (
    RetainedReadAssessment,
    ReadEligibilityRequest,
    ResolvedRetainedReadSet,
)

from fervis.lookup.read_eligibility.surface import read_eligibility_candidate_surface

from fervis.lookup.turn_prompts import build_turn_prompt_context

from fervis.lookup.turn_prompts.projections import source_binding_candidates_xml

from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)

from fervis.lookup.source_binding import (
    BoundSource,
    SourceBindingRequest,
    SourceBindingTurnPrompt,
)

from fervis.lookup.source_binding.prompt import _grain_safe_fulfillment_supports

from fervis.lookup.source_binding.candidates import SourceCandidate

from fervis.lookup.source_binding.candidates.contracts import (
    parse_evidence_item,
    parse_fulfillment_support_set,
)

from fervis.lookup.source_binding.parser.fulfillment import parse_source_fulfillments

from fervis.lookup.source_binding.provider_contract import FulfillmentDecisionOutput

from fervis.lookup.source_binding.schema import build_source_binding_schema

from fervis.lookup.plan_selection.family_specs import SourceMemberConstraint

from fervis.lookup.source_binding.plan_targets import (
    SourceBindingPlanFamily,
    SourceBindingTarget,
    SourceBindingTargetCompatibility,
    source_binding_fact_field_id,
    source_binding_target_index_for_plan_selection,
)

from fervis.lookup.source_binding.parser import parse_source_binding

from fervis.lookup.source_binding.model import SourceBindingPlan

from fervis.lookup.orchestration.pipeline import (
    _bound_plan_selection_from_plan_selection,
)

from fervis.lookup.operation_families.source_binding_registry import (
    source_binding_metric_evidence_ids_by_requested_fact,
)

from tests.lookup.source_binding_helpers import (
    source_binding_request,
    satisfying_source_population_test_results,
    source_binding_target_id_for_candidate,
    source_fulfills_by_row_population_for_candidate,
    source_fulfills_fields_for_candidate,
    source_fulfills_keys_for_candidate,
)


def _set_difference_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="Staff who have not made a sale this month.",
        answer_subject=RequestedFactAnswerSubject(subject_text="staff"),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1", description="staff", role="ANSWER_VALUE"
            ),
        ),
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SET_DIFFERENCE,
        ),
    )
    catalog = RelationCatalog(
        reads=(
            _staff_read(),
            _sales_read(),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("staff", "sales"),
                rankings=(
                    CatalogSelectionRanking(read_id="staff", score=10),
                    CatalogSelectionRanking(read_id="sales", score=9),
                ),
                selected_read_ids=("staff", "sales"),
            ),
        ),
        selected_read_ids=("staff", "sales"),
    )
    scopes = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="Which staff have not made a sale this month?",
            question_contract=QuestionContract(requested_facts=(fact,)),
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
        )
    ).candidate_scopes
    scopes_by_read = {scope.read_id: scope for scope in scopes}
    read_eligibility = ResolvedRetainedReadSet(
        retained_reads=tuple(
            RetainedReadAssessment(
                source_candidate_id=scopes_by_read[read_id].source_candidate_id,
                source_candidate_signature=(
                    scopes_by_read[read_id].source_candidate_signature
                ),
                requested_fact_id="fact_1",
                read_id=read_id,
                relevant_row_path_ids=("data",),
                relevant_field_refs=tuple(
                    scopes_by_read[read_id].field_refs_by_evidence_token.values()
                ),
                retention_basis=f"{read_id} is needed for the set-difference answer.",
            )
            for read_id in ("staff", "sales")
        )
    )
    staff_candidate_id = scopes_by_read["staff"].source_candidate_id
    sales_candidate_id = scopes_by_read["sales"].source_candidate_id
    return source_binding_request(
        question="Which staff have not made a sale this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                _set_difference_plan(
                    plan_id="plan.fact_1.a",
                    candidate_source_id=staff_candidate_id,
                    observed_source_id=sales_candidate_id,
                ),
                _set_difference_plan(
                    plan_id="plan.fact_1.b",
                    candidate_source_id=sales_candidate_id,
                    observed_source_id=staff_candidate_id,
                ),
            )
        ),
        read_eligibility=read_eligibility,
    )


def _closed_key_grouped_staff_sales_request() -> SourceBindingRequest:
    staff_1 = RequestedFactLiteralInput(
        id="staff_id_1",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="51515151-0000-0000-0002-000000000001",
        resolved_value_text="51515151-0000-0000-0002-000000000001",
        field_label_text="staff_id",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    staff_2 = RequestedFactLiteralInput(
        id="staff_id_2",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="51515151-0000-0000-0002-000000000002",
        resolved_value_text="51515151-0000-0000-0002-000000000002",
        field_label_text="staff_id",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    fact = RequestedFact(
        id="fact_1",
        description="sales count for each specified staff member today",
        answer_subject=RequestedFactAnswerSubject(subject_text="sales"),
        answer_population=RequestedFactAnswerPopulation(
            population_label="sales by specified staff member",
            counted_unit="sales",
            membership_tests=(
                RequestedFactAnswerPopulationMembershipTest(
                    id="subject_identity",
                    kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Does the row represent a sale?",
                ),
                RequestedFactAnswerPopulationMembershipTest(
                    id="normal_instance_guard",
                    kind=AnswerPopulationMembershipTestKind.NORMAL_INSTANCE_GUARD,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Is this an ordinary business instance of sales?",
                ),
            ),
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_count",
                description="sales count",
                role="ROW_COUNT",
            ),
        ),
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            group_key=RequestedFactGroupKey(
                id="answer_staff",
                description="specified staff member",
                domain=GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS,
                question_input_refs=("staff_id_1", "staff_id_2"),
            ),
        ),
        known_inputs=(staff_1, staff_2),
        input_refs=("staff_id_1", "staff_id_2"),
    )
    question_contract = QuestionContract(requested_facts=(fact,))
    available_values = (
        FactValue.identity(
            id="staff_identity_1",
            known_input_id="staff_id_1",
            key=entity_key_value(
                "staff",
                "staff_key",
                {"staff_id": "51515151-0000-0000-0002-000000000001"},
            ),
            display_value="51515151-0000-0000-0002-000000000001",
            proof_refs=("known_input:staff_id_1",),
            applies_to_requested_fact_ids=("fact_1",),
        ),
        FactValue.identity(
            id="staff_identity_2",
            known_input_id="staff_id_2",
            key=entity_key_value(
                "staff",
                "staff_key",
                {"staff_id": "51515151-0000-0000-0002-000000000002"},
            ),
            display_value="51515151-0000-0000-0002-000000000002",
            proof_refs=("known_input:staff_id_2",),
            applies_to_requested_fact_ids=("fact_1",),
        ),
    )
    catalog = RelationCatalog(reads=(_staff_sales_read(), _staff_read()))
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales", "staff"),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales did the specified staff members sell each today?",
            question_contract=question_contract,
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
        )
    ).candidate_scopes[0]
    read_eligibility = ResolvedRetainedReadSet(
        retained_reads=(
            RetainedReadAssessment(
                source_candidate_id=scope.source_candidate_id,
                source_candidate_signature=scope.source_candidate_signature,
                requested_fact_id="fact_1",
                read_id="sales",
                relevant_row_path_ids=("data",),
                relevant_field_refs=tuple(
                    dict.fromkeys(
                        (
                            *scope.field_refs_by_evidence_token.values(),
                            "sales.field.staff_name",
                        )
                    )
                ),
                retention_basis="The sales read can count sales by staff_id.",
            ),
        )
    )
    return source_binding_request(
        question="How many sales did the specified staff members sell each today?",
        question_contract=question_contract,
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.grouped_staff_sales_count",
                    requested_fact_id="fact_1",
                    source_strategy_id=(
                        "source_strategy.fact_1.grouped_staff_sales_count"
                    ),
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_staff", "answer_count"),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                            requirement_ids=("operation",),
                        ),
                    ),
                    basis="Fixture-selected grouped aggregate strategy.",
                ),
            )
        ),
        available_values=available_values,
        available_value_uses=(
            GroundedInputUse(
                id="grounded_staff_1",
                value_id="staff_identity_1",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
                requested_fact_id="fact_1",
            ),
            GroundedInputUse(
                id="grounded_staff_2",
                value_id="staff_identity_2",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
                requested_fact_id="fact_1",
            ),
        ),
        read_eligibility=read_eligibility,
    )


def _closed_key_grouped_staff_sales_today_request() -> SourceBindingRequest:
    base = _closed_key_grouped_staff_sales_request()
    today = RequestedFactLiteralInput(
        id="today",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="today",
        resolved_value_text="today",
        field_label_text="time period",
        value_meaning_hint="current day",
        role=LiteralInputRole.TIME_VALUE,
    )
    fact = replace(
        base.requested_facts[0],
        answer_population=replace(
            base.requested_facts[0].answer_population,
            membership_tests=tuple(
                test
                for test in base.requested_facts[0].answer_population.membership_tests
                if test.id != "normal_instance_guard"
            ),
        ),
        known_inputs=(*base.requested_facts[0].known_inputs, today),
        input_refs=("staff_id_1", "staff_id_2", "today"),
    )
    question_contract = QuestionContract(requested_facts=(fact,))
    today_value = FactValue.time(
        id="today_value",
        known_input_id="today",
        expression="today",
        resolved_start="2026-07-06",
        resolved_end="2026-07-06",
        granularity="day",
        proof_refs=("known_input:today",),
        applies_to_requested_fact_ids=("fact_1",),
    )
    available_values = (*base.available_values, today_value)
    catalog = RelationCatalog(reads=(_staff_sales_today_read(), _staff_read()))
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=("sales", "staff"),
                rankings=(CatalogSelectionRanking(read_id="sales", score=10),),
                selected_read_ids=("sales",),
            ),
        ),
        selected_read_ids=("sales",),
    )
    scope = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question="How many sales did the specified staff members sell each today?",
            question_contract=question_contract,
            requested_facts=(fact,),
            catalog_selection=catalog_selection,
            conversation_context={},
        )
    ).candidate_scopes[0]
    read_eligibility = ResolvedRetainedReadSet(
        retained_reads=(
            RetainedReadAssessment(
                source_candidate_id=scope.source_candidate_id,
                source_candidate_signature=scope.source_candidate_signature,
                requested_fact_id="fact_1",
                read_id="sales",
                relevant_row_path_ids=("data",),
                relevant_field_refs=tuple(scope.field_refs_by_evidence_token.values()),
                retention_basis="The sales read can count sales by staff_id and day.",
            ),
        )
    )
    return source_binding_request(
        question=base.question,
        question_contract=question_contract,
        requested_facts=(fact,),
        relation_catalog=catalog,
        catalog_selection=catalog_selection,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                SelectedSourceStrategy(
                    plan_selection_id="plan.fact_1.grouped_staff_sales_count",
                    requested_fact_id="fact_1",
                    source_strategy_id=(
                        "source_strategy.fact_1.grouped_staff_sales_count"
                    ),
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_staff", "answer_count"),
                    source_members=(
                        SourceStrategyMember(
                            source_candidate_id=scope.source_candidate_id,
                            requirement_ids=("operation",),
                        ),
                    ),
                    basis="Fixture-selected grouped aggregate strategy.",
                ),
            )
        ),
        available_values=available_values,
        available_value_uses=(
            GroundedInputUse(
                id="grounded_staff_1",
                value_id="staff_identity_1",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
                requested_fact_id="fact_1",
            ),
            GroundedInputUse(
                id="grounded_staff_2",
                value_id="staff_identity_2",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="staff_id",
                requested_fact_id="fact_1",
            ),
            GroundedInputUse(
                id="grounded_today_start",
                value_id="today_value",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="start_date",
                requested_fact_id="fact_1",
                value_component=TimeComponent.START,
            ),
            GroundedInputUse(
                id="grounded_today_end",
                value_id="today_value",
                row_source_id=api_row_source_id("sales", "data"),
                param_id="end_date",
                requested_fact_id="fact_1",
                value_component=TimeComponent.END,
            ),
        ),
        read_eligibility=read_eligibility,
        same_scope_relation_catalog=base.same_scope_relation_catalog,
        memory_inputs=base.memory_inputs,
        active_memory_ids=base.active_memory_ids,
        conversation_context=base.conversation_context,
        conversation_resolution=base.conversation_resolution,
        host=base.host,
    )


def _closed_key_model_output_with_single_staff_param(
    *,
    binding_target: dict[str, Any],
    candidate: dict[str, Any],
    row_population_evidence_id: str,
) -> dict[str, Any]:
    return {
        "outcome": {
            "kind": "source_bindings",
            "metric_fit_bases": {
                "fact_1": {
                    row_population_evidence_id: {
                        "metric_meaning": "count of sales rows",
                        "fit_basis": "The requested sales count is row cardinality.",
                    }
                }
            },
            "fit_basis_interpretations": {
                "fact_1": {
                    row_population_evidence_id: {
                        "interpretation": "FITS_REQUESTED_ANSWER",
                    }
                }
            },
            **_test_fact_binding(
                requested_fact_id="fact_1",
                plan_shape="aggregate_by_group",
                requirement_id="operation",
                invocation={
                    "binding_target_id": binding_target["binding_target_id"],
                    "answer_population": {
                        "population_binding_id": _population_binding_id(candidate),
                        "intent_text": "sales by specified staff member today",
                        "match_basis_explanation": (
                            "Use the sales row population for the grouped count."
                        ),
                        "population_test_results": (
                            satisfying_source_population_test_results(binding_target)
                        ),
                    },
                    "fulfillment_decisions": {
                        **source_fulfills_fields_for_candidate(
                            candidate,
                            field_ids_by_answer_output={
                                "answer_staff": ("staff_id",),
                            },
                        ),
                        **source_fulfills_by_row_population_for_candidate(
                            candidate,
                            answer_output_ids=("answer_count",),
                            row_path_id="data",
                        ),
                    },
                    "param_decisions": {},
                    "resolved_input_applications": [
                        _resolved_input_application(
                            target_id="start_date",
                            value_id="today_value",
                            value_component="instant",
                        ),
                        _resolved_input_application(
                            target_id="end_date",
                            value_id="today_value",
                            value_component="instant",
                        ),
                    ],
                    "row_predicate_reviews": {},
                    "finite_choice_param_reviews": {},
                },
            ),
        },
    }


def _single_param_decision(
    candidate: dict[str, Any],
    *,
    param_id: str,
    value: str,
    value_component: str = "",
) -> dict[str, object]:
    param = next(
        param for param in candidate["params"] if param["param_id"] == param_id
    )
    option = next(
        option
        for option in param["decision_options"]
        if option.get("decision") == "bind"
        and option.get("value") == value
        and str(option.get("value_component") or "") == value_component
    )
    return {
        "param_decision_id": option["param_decision_id"],
        "match_basis_explanation": f"Bind {param_id} from the grounded question input.",
        "population_intent": f"Filter by {param_id}.",
    }


def _resolved_input_application(
    *,
    target_id: str,
    value_id: str,
    value_component: str,
) -> dict[str, str]:
    return {
        "target_kind": "request_parameter",
        "target_id": target_id,
        "value_id": value_id,
        "value_component": value_component,
        "match_basis_explanation": (
            f"Apply {value_id} to the shown {target_id} request parameter."
        ),
        "population_test_results": {},
    }


def _param_proofs_by_invocation(
    bound_source: BoundSource,
    param_id: str,
) -> tuple[tuple[object, tuple[str, ...]], ...]:
    return tuple(
        (binding.value, binding.proof_refs)
        for invocation in bound_source.source_invocations
        for binding in invocation.param_bindings
        if binding.param_id == param_id
    )


def _staff_sales_read() -> EndpointRead:
    return EndpointRead(
        id="sales",
        endpoint_name="get_staff_sales",
        resource_names=("sale",),
        params=(
            CatalogParam(
                ref="sales.query.staff_id",
                name="staff_id",
                source=ParamSource.QUERY,
                type="uuid",
                required=True,
                entity_target=EntityKeyComponentTarget(
                    entity_kind="staff",
                    key_id="staff_key",
                    component_id="staff_id",
                ),
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="sales.field.sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="sales.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="sales.field.staff_name",
                path="data.staff_name",
                row_path_id="data",
                type="string",
            ),
        ),
        entity_references=(
            EntityReference(
                id="sale_staff",
                target_entity_kind="staff",
                target_key_id="staff_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id="staff_id",
                        local_field_ref="sales.field.staff_id",
                    ),
                ),
                context_field_refs=("sales.field.staff_name",),
            ),
        ),
    )


def _staff_sales_today_read() -> EndpointRead:
    base = _staff_sales_read()
    return replace(
        base,
        params=(
            *base.params,
            CatalogParam(
                ref="sales.query.start_date",
                name="start_date",
                source=ParamSource.QUERY,
                type="date",
                required=True,
            ),
            CatalogParam(
                ref="sales.query.end_date",
                name="end_date",
                source=ParamSource.QUERY,
                type="date",
                required=True,
            ),
        ),
    )


def _set_difference_plan(
    *,
    plan_id: str,
    candidate_source_id: str,
    observed_source_id: str,
) -> SelectedSourceStrategy:
    support_set_ids_by_candidate = {
        "source_1": (
            "support.source_1.answer_1.slot.source_1.answer_1.entity."
            "source_1.data.key.staff_key"
        ),
        "source_2": (
            "support.source_2.answer_1.slot.source_2.answer_1.entity."
            "source_2.data.reference.sale_staff"
        ),
    }
    candidate_support_set_id = support_set_ids_by_candidate[candidate_source_id]
    return SelectedSourceStrategy(
        plan_selection_id=plan_id,
        requested_fact_id="fact_1",
        source_strategy_id=f"source_strategy.{plan_id}",
        plan_shape="set_difference",
        required_answer_output_ids=("answer_1",),
        source_members=(
            SourceStrategyMember(
                source_candidate_id=candidate_source_id,
                requirement_ids=("candidate_set",),
                fulfillment_support_set_ids=(candidate_support_set_id,),
            ),
            SourceStrategyMember(
                source_candidate_id=observed_source_id,
                requirement_ids=("observed_set",),
            ),
        ),
        basis="Fixture-selected set-difference strategy.",
    )


def _staff_read() -> EndpointRead:
    return EndpointRead(
        id="staff",
        endpoint_name="list_staff",
        resource_names=("staff",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="staff.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="staff.field.staff_name",
                path="data.staff_name",
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
            ),
        ),
    )


def _sales_read() -> EndpointRead:
    return EndpointRead(
        id="sales",
        endpoint_name="list_sales",
        resource_names=("sale",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="sales.field.sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="sales.field.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
        ),
        entity_references=(
            EntityReference(
                id="sale_staff",
                target_entity_kind="staff",
                target_key_id="staff_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id="staff_id",
                        local_field_ref="sales.field.staff_id",
                    ),
                ),
            ),
        ),
    )


def _binding_targets(prompt: SourceBindingTurnPrompt) -> tuple[dict[str, Any], ...]:
    families = prompt.transport_context_payload()["binding_plan_families"]
    return tuple(
        target
        for fact in families["bindings_by_requested_fact"].values()
        for shape in fact["plan_shapes"].values()
        for targets in shape["role_targets"].values()
        for target in targets
    )


def _only_binding_target(prompt: SourceBindingTurnPrompt) -> dict[str, Any]:
    targets = _binding_targets(prompt)
    assert len(targets) == 1
    return targets[0]


def _only_metric_evidence_id(request: SourceBindingRequest) -> str:
    evidence_ids_by_fact = source_binding_metric_evidence_ids_by_requested_fact(request)
    evidence_ids = evidence_ids_by_fact["fact_1"]
    assert len(evidence_ids) == 1
    return evidence_ids[0]


def _target_for(
    targets: tuple[dict[str, Any], ...],
    source_candidate_id: str,
    requirement_id: str,
) -> dict[str, Any]:
    matches = [
        target
        for target in targets
        if target["source_candidate_id"] == source_candidate_id
        and target["requirement_id"] == requirement_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"target not found: {source_candidate_id}/{requirement_id}"
        )
    return matches[0]


def _source_binding_outcome(
    prompt: SourceBindingTurnPrompt,
    *,
    targets: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    candidates = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())
    invocations = []
    for target in targets:
        candidate = candidates[str(target["source_candidate_id"])]
        fulfillment_decisions = (
            source_fulfills_keys_for_candidate(
                candidate,
                key_ids_by_answer_output={"answer_1": "staff_key"},
            )
            if target["requirement_id"] == "candidate_set"
            else {}
        )
        invocations.append(
            {
                "binding_target_id": target["binding_target_id"],
                "answer_population": {
                    "population_binding_id": _population_binding_id(candidate),
                    "intent_text": f"{target['requirement_id']} rows",
                    "match_basis_explanation": (
                        f"The {target['requirement_id']} target uses this source."
                    ),
                    "population_test_results": (
                        satisfying_source_population_test_results(target)
                    ),
                },
                "fulfillment_decisions": fulfillment_decisions,
                "param_decisions": {},
                "resolved_input_applications": [],
                "row_predicate_reviews": {},
                "finite_choice_param_reviews": {},
            }
        )
    fact_bindings: dict[str, dict[str, Any]] = {}
    for target, invocation in zip(targets, invocations, strict=True):
        fact_id = str(target["requested_fact_id"])
        plan_shape = str(target["plan_shape"])
        field_id = source_binding_fact_field_id(fact_id)
        fact_binding = fact_bindings.setdefault(
            field_id,
            {"plan_shape": plan_shape},
        )
        if fact_binding["plan_shape"] != plan_shape:
            raise AssertionError("test outcome mixes source-binding plan shapes")
        fact_binding[str(target["requirement_id"])] = invocation
    return {
        "kind": "source_bindings",
        "metric_fit_bases": {},
        "fit_basis_interpretations": {},
        **fact_bindings,
    }


def _source_binding_plan_payload(*invocations: dict[str, Any]) -> dict[str, Any]:
    assert len(invocations) == 1
    return {
        "outcome": {
            "kind": "source_bindings",
            "metric_fit_bases": {},
            "fit_basis_interpretations": {},
            "bindings_for_fact_1": {
                "plan_shape": "test_shape",
                "primary": invocations[0],
            },
        }
    }


def _test_fact_binding(
    *,
    requested_fact_id: str,
    plan_shape: str,
    requirement_id: str,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    return {
        source_binding_fact_field_id(requested_fact_id): {
            "plan_shape": plan_shape,
            requirement_id: invocation,
        }
    }


def _test_plan_families(*target_ids: str) -> tuple[SourceBindingPlanFamily, ...]:
    return (
        SourceBindingPlanFamily(
            requested_fact_id="fact_1",
            plan_shape="test_shape",
            member_constraint=SourceMemberConstraint.ANY,
            required_answer_output_ids=(),
            role_targets=(
                (
                    "primary",
                    tuple(
                        SourceBindingTarget(
                            binding_target_id=target_id,
                            requested_fact_id="fact_1",
                            plan_shape="test_shape",
                            source_candidate_id=target_id.removeprefix("target."),
                            requirement_id="primary",
                        )
                        for target_id in target_ids
                    ),
                ),
            ),
        ),
    )


def _minimal_source_invocation(
    binding_target_id: str,
    population_binding_id: str,
) -> dict[str, Any]:
    return {
        "binding_target_id": binding_target_id,
        "answer_population": {
            "population_binding_id": population_binding_id,
            "intent_text": "selected rows",
            "match_basis_explanation": "This target uses the selected rows.",
            "population_test_results": {},
        },
        "fulfillment_decisions": {},
        "param_decisions": {},
        "resolved_input_applications": [],
        "row_predicate_reviews": {},
        "finite_choice_param_reviews": {},
    }


def _prompt_candidates_by_id(
    payload: dict[str, object],
) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["source_candidate_id"]): candidate
        for group in payload.get("requested_fact_sources") or ()
        if isinstance(group, dict)
        for context in group.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict) and candidate.get("source_candidate_id")
    }


def _population_binding_id(candidate: dict[str, Any]) -> str:
    bindings = candidate.get("population_bindings")
    if not bindings:
        bindings = (candidate.get("binding_surface") or {}).get("population_bindings")
    return str(bindings[0]["population_binding_id"])


def _source_invocation_variants_by_target(
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        target_schema = properties.get("binding_target_id")
        if not isinstance(target_schema, dict):
            continue
        target_ids = target_schema.get("enum")
        if isinstance(target_ids, list) and len(target_ids) == 1:
            output[str(target_ids[0])] = node
    return output


def _schema_nodes(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _schema_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _schema_nodes(child)


def _source_binding_outcome_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return next(
        variant
        for variant in schema["properties"]["outcome"]["oneOf"]
        if variant["properties"]["kind"].get("enum") == ["source_bindings"]
    )


def _flatten_source_invocation_item_variants(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    properties = schema.get("properties")
    if isinstance(properties, dict) and "binding_target_id" in properties:
        return (schema,)
    variants = schema.get("oneOf")
    if not isinstance(variants, list):
        return ()
    return tuple(
        item
        for variant in variants
        if isinstance(variant, dict)
        for item in _flatten_source_invocation_item_variants(variant)
    )


__all__ = (
    "FulfillmentDecisionOutput",
    "PlanSelectionSet",
    "ProviderObject",
    "QuestionContract",
    "RequestedFact",
    "RequestedFactAnswerExpression",
    "RequestedFactAnswerExpressionFamily",
    "RequestedFactAnswerOutput",
    "RequestedFactAnswerSubject",
    "SimpleNamespace",
    "SourceBindingPlan",
    "SourceBindingTarget",
    "SourceBindingTargetCompatibility",
    "SourceBindingTurnPrompt",
    "SourceCandidate",
    "ValidationError",
    "_binding_targets",
    "_bound_plan_selection_from_plan_selection",
    "_closed_key_grouped_staff_sales_request",
    "_closed_key_grouped_staff_sales_today_request",
    "_closed_key_model_output_with_single_staff_param",
    "_grain_safe_fulfillment_supports",
    "_minimal_source_invocation",
    "_only_binding_target",
    "_only_metric_evidence_id",
    "_param_proofs_by_invocation",
    "_population_binding_id",
    "_prompt_candidates_by_id",
    "_set_difference_request",
    "_source_binding_outcome",
    "_source_binding_outcome_schema",
    "_source_binding_plan_payload",
    "_source_invocation_variants_by_target",
    "_target_for",
    "_test_fact_binding",
    "_test_plan_families",
    "build_source_binding_schema",
    "build_turn_prompt_context",
    "compile_pattern_answer_program",
    "compiler_input_context",
    "fields",
    "parse_evidence_item",
    "parse_fulfillment_support_set",
    "parse_pattern_answer",
    "parse_source_binding",
    "parse_source_fulfillments",
    "pytest",
    "replace",
    "source_binding_candidates_xml",
    "source_binding_target_id_for_candidate",
    "source_binding_target_index_for_plan_selection",
    "source_fulfills_by_row_population_for_candidate",
    "source_fulfills_fields_for_candidate",
    "source_fulfills_keys_for_candidate",
    "validate",
)
