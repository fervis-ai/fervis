"""Public fact-plan verification boundary."""

from ._shared import (
    AnswerPlan,
    AuthorizedExecutionSources,
    CatalogSelectionResult,
    FactPlan,
    FactValue,
    PlanClarification,
    PlanImpossible,
    QuestionContract,
    RelationCatalog,
    RelationRows,
    VerificationError,
)
from .answer_plan import _verify_answer_plan
from .normalization import _normalize_fact_plan_for_verification
from .terminals import _verify_plan_clarification, _verify_plan_impossible


def verify_fact_plan(
    plan: FactPlan,
    *,
    question_contract: QuestionContract,
    catalog: RelationCatalog | None = None,
    available_values: tuple[FactValue, ...] = (),
    available_value_uses: tuple[object, ...] = (),
    memory_relations: tuple[RelationRows, ...] = (),
    catalog_selection: CatalogSelectionResult | None = None,
    authorized_sources: AuthorizedExecutionSources | None = None,
) -> FactPlan:
    catalog = _execution_catalog(catalog, authorized_sources)
    plan = _normalize_fact_plan_for_verification(
        plan,
        catalog=catalog,
        memory_relations=memory_relations,
    )
    outcome = plan.outcome
    if isinstance(outcome, AnswerPlan):
        _verify_answer_plan(
            outcome,
            question_contract=question_contract,
            catalog=catalog,
            available_values=available_values,
            available_value_uses=available_value_uses,
            memory_relations=memory_relations,
            catalog_selection=catalog_selection,
            authorized_sources=authorized_sources,
        )
    elif isinstance(outcome, PlanImpossible):
        _verify_plan_impossible(
            outcome,
            question_contract=question_contract,
            catalog=catalog,
            catalog_selection=catalog_selection,
        )
    elif isinstance(outcome, PlanClarification):
        _verify_plan_clarification(
            outcome,
            question_contract=question_contract,
            catalog=catalog,
            available_values=available_values,
            available_value_uses=available_value_uses,
            memory_relations=memory_relations,
        )
    else:
        raise VerificationError("fact plan requires a valid plan outcome")
    return plan


def _execution_catalog(
    catalog: RelationCatalog | None,
    authorized_sources: AuthorizedExecutionSources | None,
) -> RelationCatalog | None:
    if authorized_sources is None:
        return catalog
    return authorized_sources.relation_catalog


__all__ = ["verify_fact_plan"]
