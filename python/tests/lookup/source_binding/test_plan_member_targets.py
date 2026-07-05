from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any

import pytest

from fervis.lookup.relation_catalog import (
    CatalogField,
    EndpointRead,
    IdentityMetadata,
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
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
)
from fervis.lookup.read_eligibility import (
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.surface import read_eligibility_candidate_surface
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.source_binding import SourceBindingRequest, SourceBindingTurnPrompt
from fervis.lookup.source_binding.schema import build_source_binding_schema
from fervis.lookup.source_binding.plan_targets import SourceBindingTargetCompatibility
from fervis.lookup.source_binding.role_selection import value_only_source_binding_plan
from fervis.lookup.source_binding.parser import parse_source_binding
from fervis.lookup.source_binding.model import SourceBindingPlan
from fervis.lookup.orchestration.pipeline import _bound_plan_selection_from_plan_selection
from tests.lookup.source_binding_helpers import (
    source_binding_target_id_for_candidate,
    source_fulfills_for_candidate,
)


def test_source_binding_schema_uses_compact_role_target_handles():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    invocation_schemas = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )

    target_ids = {target["binding_target_id"] for target in targets}

    assert set(invocation_schemas) == target_ids
    for invocation_schema in invocation_schemas.values():
        assert "source_candidate_id" not in invocation_schema["properties"]
    candidate_target_schema = invocation_schemas[
        "target.fact_1.set_difference.source_1.candidate_set"
    ]
    fulfillment_schema = candidate_target_schema["properties"][
        "fulfillment_decisions"
    ]
    assert "answer_1" in fulfillment_schema["properties"]
    assert "anyOf" not in fulfillment_schema


def test_source_binding_schema_scopes_param_surfaces_to_binding_target():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {
                "start_date": ("param_decision.source_1.start_date.bind.month",),
            },
            "target.source_2": {
                "status": ("param_decision.source_2.status.use_default.completed",),
                "group_by": ("param_decision.source_2.group_by.bind.location",),
            },
        },
        target_required_param_ids={
            "target.source_1": ("start_date",),
            "target.source_2": ("group_by",),
        },
        target_finite_choice_values={
            "target.source_1": {"status": ("DRAFT", "COMPLETED")},
            "target.source_2": {"sale_type": ("STORE", "ONLINE")},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_membership_test_ids={
            "target.source_1": ("subject_identity",),
            "target.source_2": ("subject_identity",),
        },
        target_normal_instance_test_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_population_roles={
            "target.source_1": ({"role_id": "role.source_1.rows"},),
            "target.source_2": ({"role_id": "role.source_2.rows"},),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={"fact_1": ("source_2.data.amount",)},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {"answer_1": ("support.source_1.answer_1",)},
            "target.source_2": {"answer_1": ("support.source_2.answer_1",)},
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
            "target.source_2": ("pop.source_2.candidate_population",),
        },
    )

    variants = _source_invocation_variants_by_target(schema)
    source_1 = variants["target.source_1"]
    source_2 = variants["target.source_2"]

    assert set(source_1["properties"]["param_decisions"]["properties"]) == {
        "start_date"
    }
    assert set(
        source_1["properties"]["finite_choice_param_reviews"]["properties"]
    ) == {"status"}
    assert set(source_2["properties"]["param_decisions"]["properties"]) == {
        "group_by",
        "status",
    }
    assert set(
        source_2["properties"]["finite_choice_param_reviews"]["properties"]
    ) == {"sale_type"}


def test_source_binding_prompt_distinguishes_role_targets_from_fulfillment_outputs():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    prompt_text = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
        )
    ).prompt_text
    observed_target = _target_for(
        _binding_targets(prompt),
        "source_2",
        "observed_set",
    )

    assert observed_target["answer_output_ids"] == []
    assert "Bind every operation-required role target" in prompt_text
    assert "including targets with no answer outputs" in prompt_text


def test_parse_source_binding_binds_observed_target_without_answer_fulfillment():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    candidate_target = _target_for(targets, "source_1", "candidate_set")
    observed_target = _target_for(targets, "source_2", "observed_set")

    result = parse_source_binding(
        {
            "outcome": _source_binding_outcome(
                prompt,
                targets=(candidate_target, observed_target),
            )
        },
        request=request,
    )

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_by_requirement = {
        bound.requirement_id: bound for bound in result.outcome.bound_sources
    }
    assert set(bound_by_requirement) == {"candidate_set", "observed_set"}
    assert bound_by_requirement["candidate_set"].fulfillments
    assert bound_by_requirement["observed_set"].fulfillments == ()


def test_parse_source_binding_rejects_incomplete_compact_role_targets():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    candidate_target = _target_for(targets, "source_1", "candidate_set")

    with pytest.raises(ValueError, match="complete source binding role set"):
        parse_source_binding(
            {
                "outcome": _source_binding_outcome(
                    prompt,
                    targets=(candidate_target,),
                )
            },
            request=request,
        )


def test_bound_plan_assembly_rejects_mixed_strategy_targets():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    mixed_targets = (
        _target_for(targets, "source_1", "candidate_set"),
        _target_for(targets, "source_1", "observed_set"),
    )
    with pytest.raises(ValueError, match="complete source binding role set"):
        parse_source_binding(
            {"outcome": _source_binding_outcome(prompt, targets=mixed_targets)},
            request=request,
        )


def test_bound_plan_assembly_rejects_untargeted_bound_source():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    complete_targets = (
        _target_for(targets, "source_1", "candidate_set"),
        _target_for(targets, "source_2", "observed_set"),
    )
    result = parse_source_binding(
        {"outcome": _source_binding_outcome(prompt, targets=complete_targets)},
        request=request,
    )
    assert isinstance(result.outcome, SourceBindingPlan)
    valid_bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=request.plan_selection,
            source_binding_outcome=result.outcome,
            question_contract=request.question_contract,
        )
    )
    assert valid_bound is not None

    legacy_bound = replace(
        result.outcome.bound_sources[0],
        id="source_binding.legacy",
        binding_target_id="",
        requirement_id="",
    )
    bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=request.plan_selection,
            source_binding_outcome=SourceBindingPlan(
                bound_sources=(legacy_bound, *result.outcome.bound_sources),
            ),
            question_contract=request.question_contract,
        )
    )

    assert bound is None


def test_bound_plan_assembly_keeps_auxiliary_values_out_of_target_matching():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    complete_targets = (
        _target_for(targets, "source_1", "candidate_set"),
        _target_for(targets, "source_2", "observed_set"),
    )
    result = parse_source_binding(
        {"outcome": _source_binding_outcome(prompt, targets=complete_targets)},
        request=request,
    )
    assert isinstance(result.outcome, SourceBindingPlan)
    auxiliary_value = replace(
        result.outcome.bound_sources[0],
        id="source_binding.aux_value",
        binding_target_id="",
        requirement_id="",
        source=None,
        source_invocations=(),
        value_id="value.prior_total_sales",
        source_candidate_id="value.prior_total_sales",
        fulfillments=(),
        available_field_ids=(),
        available_fields=(),
    )

    bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=request.plan_selection,
            source_binding_outcome=SourceBindingPlan(
                bound_sources=(*result.outcome.bound_sources, auxiliary_value),
            ),
            question_contract=request.question_contract,
        )
    )

    assert bound is not None
    selected_source_binding_ids = {
        source_binding_id
        for plan in bound.plan_selections
        for member in plan.source_members
        for source_binding_id in member.source_binding_ids
    }
    assert "source_binding.aux_value" not in selected_source_binding_ids


def test_parse_source_binding_rejects_compact_equivalent_plans_with_different_fields():
    request = _set_difference_request()
    plan = request.plan_selection.plan_selections[0]
    ambiguous_request = replace(
        request,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                replace(
                    plan,
                    source_members=(
                        replace(plan.source_members[0], field_ids=("staff_name",)),
                        plan.source_members[1],
                    ),
                ),
                replace(
                    plan,
                    plan_selection_id="plan.fact_1.field_variant",
                    source_strategy_id="source_strategy.fact_1.field_variant",
                    source_members=(
                        replace(plan.source_members[0], field_ids=("staff_id",)),
                        plan.source_members[1],
                    ),
                ),
            )
        ),
    )
    prompt = SourceBindingTurnPrompt(ambiguous_request)
    targets = _binding_targets(prompt)
    with pytest.raises(ValueError, match="complete source binding role set"):
        parse_source_binding(
            {
                "outcome": _source_binding_outcome(
                    prompt,
                    targets=(
                        _target_for(targets, "source_1", "candidate_set"),
                        _target_for(targets, "source_2", "observed_set"),
                    ),
                )
            },
            request=ambiguous_request,
        )


def test_source_binding_target_compatibility_does_not_carry_evidence_selection():
    compatibility_fields = {field.name for field in fields(SourceBindingTargetCompatibility)}

    assert "fulfillment_support_set_ids" not in compatibility_fields
    assert "answer_output_ids" not in compatibility_fields


def test_value_only_bypass_selects_one_role_bound_plan_instead_of_private_union():
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="percentage increase",
                answer_subject=RequestedFactAnswerSubject(subject_text="increase"),
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer_1", description="increase"),
                ),
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.COMPUTED_SCALAR,
                ),
            ),
        )
    )
    plan_selection = PlanSelectionSet(
        plan_selections=(
            _value_only_plan(
                plan_id="plan.fact_1.values.a",
                value_1_candidate_id="value.current",
                value_2_candidate_id="value.previous",
            ),
            _value_only_plan(
                plan_id="plan.fact_1.values.b",
                value_1_candidate_id="value.previous",
                value_2_candidate_id="value.current",
            ),
        )
    )
    source_binding = value_only_source_binding_plan(
        plan_selection,
        requested_facts=question_contract.requested_facts,
    )

    bound = _bound_plan_selection_from_plan_selection(
        SimpleNamespace(
            plan_selection_outcome=plan_selection,
            source_binding_outcome=source_binding,
            question_contract=question_contract,
        )
    )

    assert bound is not None
    assert tuple(plan.plan_selection_id for plan in bound.plan_selections) == (
        "plan.fact_1.values.a",
    )


def test_source_binding_target_construction_dedupes_duplicate_private_targets():
    request = _set_difference_request()
    plan = request.plan_selection.plan_selections[0]
    duplicated_member_request = replace(
        request,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                replace(
                    plan,
                    source_members=(plan.source_members[0], plan.source_members[0]),
                ),
            ),
        ),
    )

    targets = SourceBindingTurnPrompt(
        duplicated_member_request
    ).transport_context_payload()["binding_targets"]

    assert len(targets) == 1
    assert targets[0]["binding_target_id"] == (
        "target.fact_1.set_difference.source_1.candidate_set"
    )


def test_source_binding_fixture_selector_returns_compact_role_target():
    request = _set_difference_request()
    plan = request.plan_selection.plan_selections[0]
    ambiguous_request = replace(
        request,
        plan_selection=PlanSelectionSet(
            plan_selections=(
                plan,
                replace(
                    plan,
                    plan_selection_id="plan.fact_1.c",
                    source_strategy_id="source_strategy.plan.fact_1.c",
                ),
            ),
        ),
    )
    prompt_text = SourceBindingTurnPrompt(ambiguous_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=ambiguous_request.question,
            conversation_context={},
        )
    ).prompt_text

    assert source_binding_target_id_for_candidate(
        prompt_text,
        requested_fact_id="fact_1",
        source_candidate_id=plan.source_members[0].source_candidate_id,
        source_role="candidate",
        plan_shape="set_difference",
    ) == "target.fact_1.set_difference.source_1.candidate_set"


def _set_difference_request() -> SourceBindingRequest:
    fact = RequestedFact(
        id="fact_1",
        description="Staff who have not made a sale this month.",
        answer_subject=RequestedFactAnswerSubject(subject_text="staff"),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1", description="staff"),),
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
            available_values=(),
        )
    ).candidate_scopes
    scopes_by_read = {scope.read_id: scope for scope in scopes}
    read_eligibility = ReadEligibilityResult(
        read_assessments=tuple(
            ReadAssessment(
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
                retention_decision="RETAIN",
            )
            for read_id in ("staff", "sales")
        )
    )
    staff_candidate_id = scopes_by_read["staff"].source_candidate_id
    sales_candidate_id = scopes_by_read["sales"].source_candidate_id
    return SourceBindingRequest(
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


def _set_difference_plan(
    *,
    plan_id: str,
    candidate_source_id: str,
    observed_source_id: str,
) -> SelectedSourceStrategy:
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
                fulfillment_support_set_ids=(
                    "support.source_1.answer_1.slot.source_1.answer_1.group.source_1.data.staff_name",
                ),
            ),
            SourceStrategyMember(
                source_candidate_id=observed_source_id,
                requirement_ids=("observed_set",),
            ),
        ),
        basis="Fixture-selected set-difference strategy.",
    )


def _value_only_plan(
    *,
    plan_id: str,
    value_1_candidate_id: str,
    value_2_candidate_id: str,
) -> SelectedSourceStrategy:
    return SelectedSourceStrategy(
        plan_selection_id=plan_id,
        requested_fact_id="fact_1",
        source_strategy_id=f"source_strategy.{plan_id}",
        plan_shape="computed_scalar",
        required_answer_output_ids=("answer_1",),
        source_members=(
            SourceStrategyMember(
                source_candidate_id=value_1_candidate_id,
                requirement_ids=("value_1",),
                value_id=value_1_candidate_id,
            ),
            SourceStrategyMember(
                source_candidate_id=value_2_candidate_id,
                requirement_ids=("value_2",),
                value_id=value_2_candidate_id,
            ),
        ),
        basis="Fixture-selected value-only strategy.",
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
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
            CatalogField(
                ref="staff.field.staff_name",
                path="data.staff_name",
                row_path_id="data",
                type="string",
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
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    stable=True,
                ),
            ),
        ),
    )


def _binding_targets(prompt: SourceBindingTurnPrompt) -> tuple[dict[str, Any], ...]:
    return tuple(prompt.transport_context_payload()["binding_targets"])


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
        raise AssertionError(f"target not found: {source_candidate_id}/{requirement_id}")
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
            source_fulfills_for_candidate(candidate, field_ids=("staff_name",))
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
                },
                "fulfillment_decisions": fulfillment_decisions,
                "param_decisions": {},
                "row_predicate_reviews": {},
                "finite_choice_param_reviews": {},
            }
        )
    return {
        "kind": "source_bindings",
        "metric_fit_bases": {},
        "fit_basis_interpretations": {},
        "source_invocations": invocations,
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


def _source_invocation_items_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
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
                    found = _source_invocation_items_schema(value)
                    return found
                except AssertionError:
                    pass
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        try:
                            found = _source_invocation_items_schema(item)
                            return found
                        except AssertionError:
                            pass
    raise AssertionError("source invocation item schema not found")


def _source_invocation_variants_by_target(
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    item_schema = _source_invocation_items_schema(schema)
    variants = item_schema.get("oneOf")
    if not isinstance(variants, list):
        variants = [item_schema]
    output = {}
    for variant in variants:
        target_ids = variant["properties"]["binding_target_id"]["enum"]
        assert len(target_ids) == 1
        output[target_ids[0]] = variant
    return output
