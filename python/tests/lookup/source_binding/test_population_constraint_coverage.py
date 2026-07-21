from __future__ import annotations

from fervis.lookup.answer_program.relations import PopulationCoverageRole
from fervis.lookup.source_binding.population_effects import (
    population_coverage_claims_for_satisfied_tests,
)
from fervis.lookup.source_binding.parser.membership_effects import (
    population_choice_proof_refs,
)

from tests.lookup.source_binding._plan_member_targets_fixtures import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    FactValue,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactLiteralInput,
    SourceBindingPlan,
    SourceBindingTurnPrompt,
    _binding_targets,
    _set_difference_request,
    _source_binding_outcome,
    _target_for,
    entity_key_value,
    parse_source_binding,
    replace,
)


def test_satisfied_input_owned_test_preserves_its_input_proof() -> None:
    test = RequestedFactAnswerPopulationMembershipTest(
        id="sale_type_constraint",
        kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
        polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
        test_question="Is this an in-person sale?",
        owned_question_input_refs=("q1",),
    )

    claims = population_coverage_claims_for_satisfied_tests(
        (test.id,),
        tests_by_id={test.id: test},
        requested_fact_id="fact_1",
        coverage_role=PopulationCoverageRole.ROW_POPULATION,
        proof_refs=("population_choice:sale_type",),
    )

    assert claims[0].proof_refs == (
        "population_choice:sale_type",
        "known_input:q1",
    )


def test_population_choice_proof_does_not_depend_on_an_out_of_scope_parameter() -> None:
    proof_refs = population_choice_proof_refs("population_choice:sale_type")

    assert proof_refs == ("population_choice:sale_type",)


def test_observed_only_input_application_does_not_constrain_anti_join_candidates():
    request = _named_staff_set_difference_request()
    _, outcome, _, observed_target = _set_difference_outcome(request)
    _apply_staff_input(
        outcome,
        target=observed_target,
        role="observed_set",
        basis="Restrict observed sales to Nadia.",
    )

    result = parse_source_binding({"outcome": outcome}, request=request)

    assert isinstance(result.outcome, SourceBindingPlan)
    observed = next(
        source
        for source in result.outcome.bound_sources
        if source.requirement_id == "observed_set"
    )
    assert observed.source is not None
    assert observed.source.population_coverage_claims == ()


def test_candidate_only_input_application_constrains_anti_join_result():
    request = _named_staff_set_difference_request()
    _, outcome, candidate_target, _ = _set_difference_outcome(request)
    _apply_staff_input(
        outcome,
        target=candidate_target,
        role="candidate_set",
        basis="Restrict candidate staff to Nadia.",
    )

    result = parse_source_binding({"outcome": outcome}, request=request)

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_by_role = {
        source.requirement_id: source for source in result.outcome.bound_sources
    }
    assert bound_by_role["candidate_set"].applied_filters
    assert not bound_by_role["observed_set"].applied_filters
    candidate = bound_by_role["candidate_set"]
    assert candidate.source is not None
    assert tuple(
        claim.test_ref.membership_test_id
        for claim in candidate.source.population_coverage_claims
    ) == ("specified_staff",)


def _named_staff_set_difference_request():
    base = _set_difference_request()
    staff_input = RequestedFactLiteralInput(
        id="staff_1",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="Nadia",
        resolved_value_text="Nadia",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
    )
    fact = replace(
        base.requested_facts[0],
        answer_population=RequestedFactAnswerPopulation(
            counted_unit="staff",
            membership_tests=(
                RequestedFactAnswerPopulationMembershipTest(
                    id="staff_subject",
                    kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Does this row represent a staff member?",
                ),
                RequestedFactAnswerPopulationMembershipTest(
                    id="specified_staff",
                    kind=(
                        AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT
                    ),
                    polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                    test_question="Is this the staff member named by the question?",
                    owned_question_input_refs=("staff_1",),
                ),
            ),
        ),
        known_inputs=(staff_input,),
        input_refs=("staff_1",),
    )
    return replace(
        base,
        question="Which staff member named Nadia has not made a sale?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        available_values=(
            FactValue.identity(
                id="staff_nadia",
                known_input_id="staff_1",
                key=entity_key_value(
                    "staff",
                    "staff_key",
                    {"staff_id": "staff-nadia"},
                ),
                display_value="Nadia",
                proof_refs=("known_input:staff_1",),
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
    )


def _set_difference_outcome(request):
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    candidate_target = _target_for(targets, "source_1", "candidate_set")
    observed_target = _target_for(targets, "source_2", "observed_set")
    return (
        prompt,
        _source_binding_outcome(
            prompt,
            targets=(candidate_target, observed_target),
        ),
        candidate_target,
        observed_target,
    )


def _apply_staff_input(outcome, *, target, role: str, basis: str) -> None:
    application = target["resolved_input_application"]
    value = application["resolved_values"][0]
    target_kind, components = next(
        iter(value["components_by_target_kind"].items())
    )
    population_test_basis = value["population_test_basis"]
    selected_target = application["targets_by_kind"][target_kind][0]
    outcome["bindings_for_fact_1"][role]["resolved_input_applications"] = [
        {
            "value_id": value["value_id"],
            "applications": [
                {
                    "application_target_id": selected_target[
                        "application_target_id"
                    ],
                    "value_component": components[0],
                    "match_basis_explanation": basis,
                }
            ],
            "population_test_results": {
                test_id: {
                    "test_id": test_id,
                    "test_question": test_basis["test_question"],
                    "role_scoped_test_question": test_basis[
                        "role_scoped_test_question"
                    ],
                    "because": basis,
                    "test_effect": "SATISFIES_TEST",
                }
                for test_id, test_basis in population_test_basis.items()
            },
        }
    ]
