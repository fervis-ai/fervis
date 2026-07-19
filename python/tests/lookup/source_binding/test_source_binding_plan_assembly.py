from __future__ import annotations

from tests.lookup.source_binding._plan_member_targets_fixtures import (
    PlanSelectionSet,
    SimpleNamespace,
    SourceBindingPlan,
    SourceBindingTargetCompatibility,
    SourceBindingTurnPrompt,
    _binding_targets,
    _bound_plan_selection_from_plan_selection,
    _set_difference_request,
    _source_binding_outcome,
    _target_for,
    build_turn_prompt_context,
    fields,
    parse_source_binding,
    pytest,
    replace,
    source_binding_target_id_for_candidate,
    source_binding_target_index_for_plan_selection,
)


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
    assert "bind every role shown for that shape exactly once" in prompt_text
    assert "including roles with no answer outputs" in prompt_text


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


def test_source_binding_target_compatibility_does_not_carry_evidence_selection():
    compatibility_fields = {
        field.name for field in fields(SourceBindingTargetCompatibility)
    }

    assert "fulfillment_support_set_ids" not in compatibility_fields
    assert "answer_output_ids" not in compatibility_fields


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

    targets = source_binding_target_index_for_plan_selection(
        duplicated_member_request.plan_selection,
        requested_facts=duplicated_member_request.requested_facts,
    ).targets

    assert len(targets) == 1
    assert targets[0].binding_target_id == (
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
    prompt_text = (
        SourceBindingTurnPrompt(ambiguous_request)
        .to_model_invocation(
            build_turn_prompt_context(
                current_question=ambiguous_request.question,
                conversation_context={},
            )
        )
        .prompt_text
    )

    assert (
        source_binding_target_id_for_candidate(
            prompt_text,
            requested_fact_id="fact_1",
            source_candidate_id=plan.source_members[0].source_candidate_id,
            source_role="candidate",
            plan_shape="set_difference",
        )
        == "target.fact_1.set_difference.source_1.candidate_set"
    )
