"""Internal source-binding parse result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSourcePopulationChoice,
)


__all__ = [
    "DerivedFiniteChoiceParamDecisions",
    "NormalizedParamDecision",
    "ParamDecisionParse",
    "PopulationChoiceSet",
    "RowPredicateParse",
]


@dataclass(frozen=True)
class ParamDecisionParse:
    binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...]


@dataclass(frozen=True)
class RowPredicateParse:
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...] = ()
    discharged_membership_test_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PopulationChoiceSet:
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedParamDecision:
    population_intent: str
    match_basis_explanation: str
    param_decision_id: Optional[str] = None
    population_choice_set: Optional[PopulationChoiceSet] = None


@dataclass(frozen=True)
class DerivedFiniteChoiceParamDecisions:
    param_decisions: dict[str, NormalizedParamDecision]
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...]
    discharged_membership_test_ids: tuple[str, ...] = ()
