"""Finite-choice param review parsing."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import PopulationChoiceControllerKind
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.parser.membership_effects import (
    answer_population_tests_by_id,
    population_choice_proof_refs,
    relation_review_scope_decisions,
    review_finite_choice_sets,
)
from fervis.lookup.source_binding.parser.types import DerivedFiniteChoiceParamDecisions
from fervis.lookup.source_binding.parser_common import _dict, _text
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.review_surface import source_binding_review_surface


__all__ = [
    "derive_finite_choice_param_decisions",
]


def derive_finite_choice_param_decisions(
    raw_reviews: Any,
    *,
    candidate: Any,
    requested_fact_id: str,
    binding_target_id: str,
    request: SourceBindingRequest,
    review_scope: SourceBindingReviewScope,
    answer_population: provider_output.AnswerPopulationOutput,
    raw_param_decision_ids: tuple[str, ...],
) -> DerivedFiniteChoiceParamDecisions:
    reviews = _dict(raw_reviews, "finite_choice_param_reviews")
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
    output: dict[str, dict[str, Any]] = {}
    population_choices: list[DraftRelationSourcePopulationChoice] = []
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
        _controlled_population_role(
            reviews.get(param_id),
            population_roles_by_id=population_roles_by_id,
            path=f"finite_choice_param_reviews.{param_id}",
        )
        include_values, exclude_values = review_finite_choice_sets(
            reviews.get(param_id),
            axis=axis,
            tests_by_id=tests_by_id,
        )
        population_choices.append(
            DraftRelationSourcePopulationChoice(
                controller_kind=PopulationChoiceControllerKind.QUERY_PARAM,
                controller_id=param_id,
                field_id=param_id,
                requested_fact_ids=(requested_fact_id,),
                included_values=include_values,
                excluded_values=exclude_values,
                parameter_id=(
                    f"semantic.{requested_fact_id}.{binding_target_id}."
                    f"param.{param_id}"
                ),
                proof_refs=population_choice_proof_refs(
                    f"population_choice:{param_id}",
                    out_of_scope_decisions,
                ),
                review_scope_decisions=relation_review_scope_decisions(
                    out_of_scope_decisions
                ),
            )
        )
        if axis.can_be_omitted(include_values=include_values):
            continue
        output[param_id] = {
            "population_intent": _text(answer_population.intent_text),
            "match_basis_explanation": (
                "Derived from finite_choice_param_reviews because at least one shown "
                "choice does not satisfy the requested answer population tests."
            ),
            "population_choice_set": {
                "include_values": list(include_values),
                "exclude_values": list(exclude_values),
            },
        }
    return DerivedFiniteChoiceParamDecisions(
        param_decisions=output,
        population_choices=tuple(population_choices),
    )


def _controlled_population_role(
    raw: Any,
    *,
    population_roles_by_id: dict[str, dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    if not population_roles_by_id:
        raise ValueError("finite choice population tests require population_roles")
    role = provider_output.FiniteChoiceParamReviewOutput.parse(raw)
    role_id = _text(role.controlled_population_role_id)
    expected = population_roles_by_id.get(role_id)
    if expected is None:
        raise ValueError("finite choice param references unknown role")
    if not role_id:
        raise ValueError("finite choice param requires role id")
    if not _text(role.role_selection_basis).strip():
        raise ValueError("finite choice param requires role selection basis")
    return expected


def _candidate_population_roles_by_id(candidate: Any) -> dict[str, dict[str, Any]]:
    return {
        role.role_id: {"role_kind": "", "role_text": ""}
        for role in source_binding_review_surface(candidate).population_roles
    }
