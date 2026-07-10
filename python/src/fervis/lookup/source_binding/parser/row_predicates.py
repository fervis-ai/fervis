"""Row-predicate review parsing."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import PopulationChoiceControllerKind
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.parser.membership_effects import answer_population_tests_by_id, choice_is_included, population_choice_proof_refs, population_tests_allow_choice, relation_review_scope_decisions
from fervis.lookup.source_binding.parser.types import RowPredicateParse
from fervis.lookup.source_binding.parser_common import _dict, _required_dicts, _text
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope
from fervis.lookup.source_binding.review_surface import source_binding_review_surface


__all__ = [
    "parse_row_predicate_filters",
]


def parse_row_predicate_filters(
    raw_reviews: Any,
    *,
    candidate: Any,
    request: SourceBindingRequest,
    requested_fact_id: str,
    binding_target_id: str,
    review_scope: SourceBindingReviewScope,
) -> RowPredicateParse:
    reviews = _dict(raw_reviews, "row_predicate_reviews")
    candidate_predicates_by_id = source_binding_review_surface(candidate).row_predicates
    scoped_test_ids_by_predicate = {
        predicate_id: test_ids
        for predicate_id in candidate_predicates_by_id
        if (
            test_ids := review_scope.row_predicate_test_ids(
                binding_target_id,
                predicate_id,
            )
        )
    }
    predicates_by_id = {
        predicate_id: candidate_predicates_by_id[predicate_id]
        for predicate_id in scoped_test_ids_by_predicate
    }
    if not predicates_by_id and not reviews:
        return RowPredicateParse()
    missing_predicate_ids = set(predicates_by_id) - set(reviews)
    if missing_predicate_ids:
        raise ValueError("source binding missing row predicate review")
    population_choices: list[DraftRelationSourcePopulationChoice] = []
    for predicate_id, raw in reviews.items():
        predicate = predicates_by_id.get(predicate_id)
        if predicate is None:
            raise ValueError("row predicate review references unknown predicate")
        out_of_scope_decisions = review_scope.row_predicate_out_of_scope_decisions(
            binding_target_id,
            predicate_id,
        )
        tests_by_id = answer_population_tests_by_id(
            request=request,
            requested_fact_id=requested_fact_id,
            scoped_test_ids=scoped_test_ids_by_predicate[predicate_id],
        )
        allowed_values = predicate.allowed_values
        values = _row_predicate_include_values(
            raw,
            allowed_values=allowed_values,
            tests_by_id=tests_by_id,
            path=f"row_predicate_reviews.{predicate_id}",
        )
        excluded_values = tuple(
            value for value in allowed_values if value not in values
        )
        field_id = predicate.field_id
        if not field_id:
            raise ValueError("row predicate missing field")
        population_choices.append(
            DraftRelationSourcePopulationChoice(
                controller_kind=PopulationChoiceControllerKind.ROW_PREDICATE,
                controller_id=predicate_id,
                field_id=field_id,
                requested_fact_ids=(requested_fact_id,),
                included_values=values,
                excluded_values=excluded_values,
                parameter_id=(
                    f"semantic.{requested_fact_id}.{binding_target_id}."
                    f"row_predicate.{predicate_id}"
                ),
                proof_refs=population_choice_proof_refs(
                    f"row_predicate:{predicate_id}",
                    out_of_scope_decisions,
                ),
                review_scope_decisions=relation_review_scope_decisions(
                    out_of_scope_decisions
                ),
            )
        )
    return RowPredicateParse(
        population_choices=tuple(population_choices),
    )


def _row_predicate_include_values(
    raw_review: Any,
    *,
    allowed_values: tuple[str, ...],
    tests_by_id: dict[str, Any],
    path: str,
) -> tuple[str, ...]:
    if not allowed_values:
        raise ValueError("row predicate requires allowed values")
    reviewed_effects = _reviewed_row_predicate_effects(
        raw_review,
        allowed_values=allowed_values,
        tests_by_id=tests_by_id,
        path=f"{path}.choice_reviews",
    )
    active_test_ids = _row_predicate_filter_test_ids(
        reviewed_effects,
        tests_by_id=tests_by_id,
        path=f"{path}.choice_reviews",
    )
    if not _has_decisive_row_predicate_effect(
        reviewed_effects,
        test_ids=active_test_ids,
    ):
        return allowed_values
    values = tuple(
        value
        for value, test_effects in reviewed_effects
        if choice_is_included(
            test_effects=test_effects,
            tests_by_id=tests_by_id,
            test_ids=active_test_ids,
            choice_inclusion=None,
        )
    )
    if not values:
        raise ValueError("row predicate review must include at least one value")
    return values


def _row_predicate_filter_test_ids(
    reviewed_effects: list[tuple[str, dict[str, str]]],
    *,
    tests_by_id: dict[str, Any],
    path: str,
) -> tuple[str, ...]:
    output: list[str] = []
    for test_id in tests_by_id:
        effects = tuple(test_effects[test_id] for _, test_effects in reviewed_effects)
        unique_effects = set(effects)
        if unique_effects <= {"DOES_NOT_DECIDE_TEST", "UNKNOWN_TEST_EFFECT"}:
            continue
        if all(
            population_tests_allow_choice(
                test_effects={test_id: effect},
                tests_by_id=tests_by_id,
                test_ids=(test_id,),
            )
            for effect in effects
        ):
            continue
        output.append(test_id)
    return tuple(output)


def _has_decisive_row_predicate_effect(
    reviewed_effects: list[tuple[str, dict[str, str]]],
    *,
    test_ids: tuple[str, ...],
) -> bool:
    return any(
        test_effects[test_id] in {"SATISFIES_TEST", "CONFLICTS_WITH_TEST"}
        for _, test_effects in reviewed_effects
        for test_id in test_ids
    )


def _reviewed_row_predicate_effects(
    raw_review: Any,
    *,
    allowed_values: tuple[str, ...],
    tests_by_id: dict[str, Any],
    path: str,
) -> list[tuple[str, dict[str, str]]]:
    review = provider_output.RowPredicateReviewOutput.parse(raw_review)
    raw_choices = _required_dicts(review.choice_reviews, path)
    seen: set[str] = set()
    output: list[tuple[str, dict[str, str]]] = []
    for raw_value in raw_choices:
        raw = provider_output.RowPredicateChoiceReviewOutput.parse(raw_value)
        value = _text(raw.choice_option_id)
        if value not in allowed_values:
            raise ValueError("row predicate review references unknown value")
        if value in seen:
            raise ValueError("duplicate row predicate value review")
        seen.add(value)
        if not _text(raw.choice_domain_meaning).strip():
            raise ValueError("row predicate value review requires domain meaning")
        output.append(
            (
                value,
                _row_predicate_population_test_effects(
                    raw.population_test_results,
                    tests_by_id=tests_by_id,
                    path=f"{path}.{value}.population_test_results",
                ),
            )
        )
    if seen != set(allowed_values):
        raise ValueError("row predicate reviews must cover every value")
    return output


def _row_predicate_population_test_effects(
    raw_results: Any,
    *,
    tests_by_id: dict[str, Any],
    path: str,
) -> dict[str, str]:
    results = _dict(raw_results, path)
    if set(results) != set(tests_by_id):
        raise ValueError("row predicate reviews must answer membership tests")
    effects: dict[str, str] = {}
    for test_id, test in tests_by_id.items():
        raw = provider_output.RowPredicatePopulationTestResultOutput.parse(results.get(test_id))
        if _text(raw.test_id) != test_id:
            raise ValueError("row predicate population test id must match result key")
        if not _text(raw.test_question).strip():
            raise ValueError("row predicate population test requires question")
        if not _text(raw.role_scoped_test_question).strip():
            raise ValueError(
                "row predicate population test requires role-scoped question"
            )
        if not _text(raw.because).strip():
            raise ValueError("row predicate population test requires reason")
        effect = _text(raw.test_effect)
        if effect not in {
            "SATISFIES_TEST",
            "CONFLICTS_WITH_TEST",
            "DOES_NOT_DECIDE_TEST",
            "UNKNOWN_TEST_EFFECT",
        }:
            raise ValueError("unsupported row predicate population test effect")
        effects[test_id] = effect
    return effects
