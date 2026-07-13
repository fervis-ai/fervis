"""Parsed source-binding provider decisions."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.source_binding.compiler_ir import DraftRelationSourcePopulationChoice
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.plan_targets import SourceBindingTarget
from fervis.lookup.source_binding.parser.types import NormalizedParamDecision


@dataclass(frozen=True)
class ParsedRoleBinding:
    target: SourceBindingTarget
    invocation: provider_output.SourceInvocationOutput
    param_decisions: dict[str, NormalizedParamDecision]
    effective_param_ids: tuple[str, ...]
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...]
    discharged_membership_test_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParsedSourceBindingPlan:
    metric_fit_bases: object
    fit_basis_interpretations: object
    role_bindings: tuple[ParsedRoleBinding, ...]
