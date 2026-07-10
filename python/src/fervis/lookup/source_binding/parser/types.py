"""Internal source-binding parse result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSourcePopulationChoice,
)


__all__ = [
    "DerivedFiniteChoiceParamDecisions",
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


@dataclass(frozen=True)
class PopulationChoiceSet:
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...]


@dataclass(frozen=True)
class DerivedFiniteChoiceParamDecisions:
    param_decisions: dict[str, dict[str, Any]]
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...]
