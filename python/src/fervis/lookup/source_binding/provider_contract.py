"""Typed provider-output contracts for source binding."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class AnswerPopulationOutput(ProviderOutput):
    population_binding_id: str
    intent_text: str
    match_basis_explanation: str
    population_test_results: dict[str, RowPredicatePopulationTestResultOutput]


@dataclass(frozen=True)
class FulfillmentDecisionOutput(ProviderOutput):
    match_basis_explanation: str
    fulfillment_choice_id: str


@dataclass(frozen=True)
class ParamDecisionOutput(ProviderOutput):
    population_intent: str
    match_basis_explanation: str
    param_decision_id: str


@dataclass(frozen=True)
class ResolvedInputTargetApplicationOutput(ProviderOutput):
    application_target_id: str
    value_component: str
    match_basis_explanation: str


@dataclass(frozen=True)
class ResolvedInputApplicationOutput(ProviderOutput):
    value_id: str
    applications: tuple[ResolvedInputTargetApplicationOutput, ...]
    population_test_results: dict[str, RowPredicatePopulationTestResultOutput]


@dataclass(frozen=True)
class SourceInvocationOutput(ProviderOutput):
    binding_target_id: str
    answer_population: AnswerPopulationOutput
    fulfillment_decisions: dict[str, FulfillmentDecisionOutput]
    param_decisions: dict[str, ParamDecisionOutput]
    row_predicate_reviews: dict[str, RowPredicateReviewOutput]
    finite_choice_param_reviews: dict[str, FiniteChoiceParamReviewOutput]
    resolved_input_applications: tuple[ResolvedInputApplicationOutput, ...]


@dataclass(frozen=True)
class MetricFitBasisOutput(ProviderOutput):
    metric_meaning: str
    fit_basis: str


@dataclass(frozen=True)
class FitBasisInterpretationOutput(ProviderOutput):
    interpretation: str


@dataclass(frozen=True)
class PopulationTestBasisOutput(ProviderOutput):
    test_question: str
    role_scoped_test_question: str


@dataclass(frozen=True)
class FiniteChoiceParamReviewOutput(ProviderOutput):
    controlled_population_role_id: str
    role_selection_basis: str
    population_test_basis: dict[str, PopulationTestBasisOutput]
    choice_reviews: tuple[FiniteChoiceReviewOutput, ...]


@dataclass(frozen=True)
class FiniteChoiceReviewOutput(ProviderOutput):
    choice_option_id: str
    choice_domain_meaning: str
    choice_inclusion_basis: str
    choice_inclusion: str
    population_test_results: dict[str, ProviderObject]


@dataclass(frozen=True)
class StandardPopulationTestResultOutput(ProviderOutput):
    test_basis: str
    population_consequence: str
    test_effect: str


@dataclass(frozen=True)
class NormalInstanceDispositionOutput(ProviderOutput):
    matched_excluded_role: str
    test_effect: str


@dataclass(frozen=True)
class NormalInstanceTestResultOutput(ProviderOutput):
    role_match_basis: str
    population_consequence: str
    disposition: NormalInstanceDispositionOutput


@dataclass(frozen=True)
class RowPredicateReviewOutput(ProviderOutput):
    choice_reviews: tuple[RowPredicateChoiceReviewOutput, ...]


@dataclass(frozen=True)
class RowPredicateChoiceReviewOutput(ProviderOutput):
    choice_option_id: str
    choice_domain_meaning: str
    population_test_results: dict[str, RowPredicatePopulationTestResultOutput]


@dataclass(frozen=True)
class RowPredicatePopulationTestResultOutput(ProviderOutput):
    test_id: str
    test_question: str
    role_scoped_test_question: str
    because: str
    test_effect: str
