"""Typed relational fact-plan model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.values import BindingSet


class PlanOutcomeKind(StrEnum):
    FACT_PLAN = "fact_plan"
    NEEDS_CLARIFICATION = "needs_clarification"
    IMPOSSIBLE = "impossible"


class BlockedFactBasis(StrEnum):
    CATALOG_ACCESS = "catalog_access"
    POLICY_ACCESS = "policy_access"


class MissingCatalogInputKind(StrEnum):
    REQUIRED_INPUT = "missing_catalog_required_input"
    CHOICE_INPUT = "missing_catalog_choice_input"


@dataclass(frozen=True)
class BlockedFactField:
    read_id: str
    field_id: str


@dataclass(frozen=True)
class BlockedFact:
    requested_fact_id: str
    basis: BlockedFactBasis
    evidence_refs: tuple[str, ...] = ()
    reviewed_read_ids: tuple[str, ...] = ()
    nearest_fields: tuple[BlockedFactField, ...] = ()
    explanation: str = ""


@dataclass(frozen=True)
class MissingCatalogRequiredInput:
    id: str
    requested_fact_id: str
    required_catalog_input_id: str
    kind: MissingCatalogInputKind = MissingCatalogInputKind.REQUIRED_INPUT

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("missing catalog required input requires id")
        if not self.requested_fact_id:
            raise ValueError("missing catalog required input requires requested fact")
        if not self.required_catalog_input_id:
            raise ValueError("missing catalog required input requires catalog input")


@dataclass(frozen=True)
class MissingCatalogChoiceInput:
    id: str
    requested_fact_id: str
    required_catalog_choice_input_id: str
    kind: MissingCatalogInputKind = MissingCatalogInputKind.CHOICE_INPUT

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("missing catalog choice input requires id")
        if not self.requested_fact_id:
            raise ValueError("missing catalog choice input requires requested fact")
        if not self.required_catalog_choice_input_id:
            raise ValueError(
                "missing catalog choice input requires catalog choice input"
            )


MissingCatalogInput = MissingCatalogRequiredInput | MissingCatalogChoiceInput


@dataclass(frozen=True)
class PlanClarification:
    missing_catalog_inputs: tuple[MissingCatalogInput, ...]
    kind: PlanOutcomeKind = PlanOutcomeKind.NEEDS_CLARIFICATION

    def __post_init__(self) -> None:
        if not self.missing_catalog_inputs:
            raise ValueError("plan clarification requires missing catalog inputs")


@dataclass(frozen=True)
class PlanImpossible:
    blocked_facts: tuple[BlockedFact, ...]
    kind: PlanOutcomeKind = PlanOutcomeKind.IMPOSSIBLE

    def __post_init__(self) -> None:
        if not self.blocked_facts:
            raise ValueError("plan impossible requires blocked facts")


PlanOutcome = AnswerProgram | PlanClarification | PlanImpossible


@dataclass(frozen=True)
class FactPlan:
    outcome: PlanOutcome
    bindings: BindingSet = BindingSet()
