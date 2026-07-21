"""Finite-choice param review parsing."""

from __future__ import annotations

from fervis.lookup.source_binding.compiler_ir import (
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import (
    PopulationCoverageClaim,
    PopulationCoverageRole,
    PopulationChoiceControllerKind,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.candidates.model import SourceCandidate
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.parser.membership_effects import (
    answer_population_tests_by_id,
    population_choice_proof_refs,
    relation_review_scope_decisions,
    review_finite_choice_sets,
)
from fervis.lookup.source_binding.parser.types import (
    DerivedFiniteChoiceParamDecisions,
    NormalizedParamDecision,
    PopulationChoiceSet,
)
from fervis.lookup.source_binding.parser_common import _text
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.review_surface import source_binding_review_surface
from fervis.lookup.source_binding.population_effects import (
    population_coverage_claims_for_satisfied_tests,
)


__all__ = [
    "derive_finite_choice_param_decisions",
]


def derive_finite_choice_param_decisions(
    reviews: dict[str, provider_output.FiniteChoiceParamReviewOutput],
    *,
    candidate: SourceCandidate,
    requested_fact_id: str,
    binding_target_id: str,
    request: SourceBindingRequest,
    review_scope: SourceBindingReviewScope,
    answer_population: provider_output.AnswerPopulationOutput,
    raw_param_decision_ids: tuple[str, ...],
    resolved_input_param_values: dict[str, tuple[object, ...]],
    coverage_role: PopulationCoverageRole,
) -> DerivedFiniteChoiceParamDecisions:
    review_surface = source_binding_review_surface(candidate)
    scoped_test_ids_by_param = {
        param_id: test_ids
        for param_id in review_surface.finite_choice_params
        if (
            test_ids := review_scope.finite_choice_param_test_ids(
                binding_target_id,
                param_id,
            )
        )
    }
    finite_choice_axes = {
        param_id: review_surface.finite_choice_params[param_id]
        for param_id in scoped_test_ids_by_param
    }
    expected_param_ids = set(finite_choice_axes)
    if set(reviews) != expected_param_ids:
        raise ValueError(
            "finite choice param reviews must cover every finite-choice population param"
        )
    authored_choice_decisions = expected_param_ids & set(raw_param_decision_ids)
    if authored_choice_decisions:
        raise ValueError(
            "finite-choice population params must be derived from choice reviews"
        )
    output: dict[str, NormalizedParamDecision] = {}
    population_choices: list[DraftRelationSourcePopulationChoice] = []
    coverage_claims: list[PopulationCoverageClaim] = []
    population_roles_by_id = _candidate_population_roles_by_id(candidate)
    for param_id, axis in finite_choice_axes.items():
        out_of_scope_decisions = (
            review_scope.finite_choice_param_out_of_scope_decisions(
                binding_target_id,
                param_id,
            )
        )
        tests_by_id = answer_population_tests_by_id(
            request=request,
            requested_fact_id=requested_fact_id,
            scoped_test_ids=scoped_test_ids_by_param[param_id],
        )
        review = reviews[param_id]
        _controlled_population_role(
            review,
            population_roles_by_id=population_roles_by_id,
            path=f"finite_choice_param_reviews.{param_id}",
        )
        include_values, exclude_values, satisfied_test_ids = review_finite_choice_sets(
            review,
            axis=axis,
            tests_by_id=tests_by_id,
        )
        applied_values = resolved_input_param_values.get(param_id, ())
        if any(value not in include_values for value in applied_values):
            raise ValueError(
                "resolved input application selects an excluded finite choice"
            )
        proof_refs = population_choice_proof_refs(f"population_choice:{param_id}")
        population_choices.append(
            DraftRelationSourcePopulationChoice(
                controller_kind=PopulationChoiceControllerKind.QUERY_PARAM,
                controller_id=param_id,
                field_id=param_id,
                requested_fact_ids=(requested_fact_id,),
                included_values=include_values,
                excluded_values=exclude_values,
                parameter_id=(
                    f"semantic.{requested_fact_id}.{binding_target_id}.param.{param_id}"
                ),
                proof_refs=proof_refs,
                review_scope_decisions=relation_review_scope_decisions(
                    out_of_scope_decisions
                ),
            )
        )
        coverage_claims.extend(
            population_coverage_claims_for_satisfied_tests(
                satisfied_test_ids,
                tests_by_id=tests_by_id,
                requested_fact_id=requested_fact_id,
                coverage_role=coverage_role,
                proof_refs=proof_refs,
            )
        )
        if applied_values or axis.can_be_omitted(include_values=include_values):
            continue
        output[param_id] = NormalizedParamDecision(
            population_intent=_text(answer_population.intent_text),
            match_basis_explanation=(
                "Derived from finite_choice_param_reviews because at least one shown "
                "choice does not satisfy the requested answer population tests."
            ),
            population_choice_set=PopulationChoiceSet(
                included_values=include_values,
                excluded_values=exclude_values,
            ),
        )
    return DerivedFiniteChoiceParamDecisions(
        param_decisions=output,
        population_choices=tuple(population_choices),
        population_coverage_claims=tuple(coverage_claims),
    )


def _controlled_population_role(
    review: provider_output.FiniteChoiceParamReviewOutput,
    *,
    population_roles_by_id: dict[str, dict[str, str]],
    path: str,
) -> dict[str, str]:
    if not population_roles_by_id:
        raise ValueError("finite choice population tests require population_roles")
    role_id = _text(review.controlled_population_role_id)
    expected = population_roles_by_id.get(role_id)
    if expected is None:
        raise ValueError("finite choice param references unknown role")
    if not role_id:
        raise ValueError("finite choice param requires role id")
    if not _text(review.role_selection_basis).strip():
        raise ValueError("finite choice param requires role selection basis")
    return expected


def _candidate_population_roles_by_id(
    candidate: SourceCandidate,
) -> dict[str, dict[str, str]]:
    return {
        role.role_id: {"role_kind": "", "role_text": ""}
        for role in source_binding_review_surface(candidate).population_roles
    }
