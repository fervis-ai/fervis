"""Provider-output DTOs for source binding."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


SourceBindingPlanOutput = provider_output_type(
    "SourceBindingPlanOutput",
    ("kind", "metric_fit_bases", "fit_basis_interpretations", "source_invocations"),
)
SourceInvocationOutput = provider_output_type(
    "SourceInvocationOutput",
    (
        "binding_target_id",
        "answer_population",
        "fulfillment_decisions",
        "param_decisions",
        "row_predicate_reviews",
        "finite_choice_param_reviews",
    ),
)
AnswerPopulationOutput = provider_output_type(
    "AnswerPopulationOutput",
    ("population_binding_id", "intent_text", "match_basis_explanation"),
)
FulfillmentDecisionOutput = provider_output_type(
    "FulfillmentDecisionOutput",
    ("match_basis_explanation", "fulfillment_choice_id"),
)
MetricFitBasisOutput = provider_output_type(
    "MetricFitBasisOutput",
    ("metric_meaning", "fit_basis"),
)
FitBasisInterpretationOutput = provider_output_type(
    "FitBasisInterpretationOutput",
    ("interpretation",),
)
ParamDecisionOutput = provider_output_type(
    "ParamDecisionOutput",
    ("population_intent", "match_basis_explanation", "param_decision_id"),
)
FiniteChoiceParamReviewOutput = provider_output_type(
    "FiniteChoiceParamReviewOutput",
    (
        "controlled_population_role_id",
        "role_selection_basis",
        "population_test_basis",
        "choice_reviews",
    ),
)
PopulationTestBasisOutput = provider_output_type(
    "PopulationTestBasisOutput",
    ("test_question", "role_scoped_test_question"),
)
FiniteChoiceReviewOutput = provider_output_type(
    "FiniteChoiceReviewOutput",
    (
        "choice_option_id",
        "choice_domain_meaning",
        "choice_inclusion_basis",
        "choice_inclusion",
        "population_test_results",
    ),
)
StandardPopulationTestResultOutput = provider_output_type(
    "StandardPopulationTestResultOutput",
    ("test_basis", "population_consequence", "test_effect"),
)
NormalInstanceTestResultOutput = provider_output_type(
    "NormalInstanceTestResultOutput",
    (
        "role_match_basis",
        "explicit_user_override_evidence",
        "explicit_user_override_applies",
        "population_consequence",
        "disposition",
    ),
)
NormalInstanceDispositionOutput = provider_output_type(
    "NormalInstanceDispositionOutput",
    ("matched_excluded_role", "test_effect"),
)
NormalInstanceOverrideEvidenceOutput = provider_output_type(
    "NormalInstanceOverrideEvidenceOutput",
    ("source_text", "reason"),
)
RowPredicateReviewOutput = provider_output_type(
    "RowPredicateReviewOutput",
    ("choice_reviews",),
)
RowPredicateChoiceReviewOutput = provider_output_type(
    "RowPredicateChoiceReviewOutput",
    ("choice_option_id", "choice_domain_meaning", "population_test_results"),
)
RowPredicatePopulationTestResultOutput = provider_output_type(
    "RowPredicatePopulationTestResultOutput",
    (
        "test_id",
        "test_question",
        "role_scoped_test_question",
        "because",
        "test_effect",
    ),
)
