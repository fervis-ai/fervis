"""Provider DTOs for semantic Source Binding."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class SetRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    rows_ref: str


@dataclass(frozen=True)
class ReturnedFactRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    field_ref: str


@dataclass(frozen=True)
class AssociationRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    from_rows_ref: str
    to_rows_ref: str
    realization_ref: str
    reference_from_set_ref: str | None = None


@dataclass(frozen=True)
class ResolvedInputApplicationOutput(ProviderOutput):
    kind: str
    mapping_basis: str
    owner_ref: str
    value_ref: str
    value_component: str
    target_ref: str


@dataclass(frozen=True)
class UnappliedInputOutput(ProviderOutput):
    kind: str
    mapping_basis: str
    owner_ref: str
    value_ref: str


@dataclass(frozen=True)
class FiniteChoiceApplicationOutput(ProviderOutput):
    application_basis: str
    surface_ref: str
    selected_choice_values: tuple[str, ...]


@dataclass(frozen=True)
class ChoiceRequirementApplicationOutput(ProviderOutput):
    mapping_basis: str
    selected_by_requirements: tuple[str, ...]


@dataclass(frozen=True)
class SourceRealizationUnavailableOutput(ProviderOutput):
    kind: str
    unmet_requirement_refs: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class SourceRealizationOutput(ProviderOutput):
    set_bindings: dict[str, tuple[SetRealizationOutput, ...]]
    fact_bindings: dict[str, tuple[ReturnedFactRealizationOutput, ...]]
    association_bindings: dict[str, tuple[AssociationRealizationOutput, ...]]


@dataclass(frozen=True)
class SemanticSourceBindingOutput(ProviderOutput):
    resolved_input_applications: dict[
        str,
        tuple[ProviderObject, ...],
    ]
    finite_choice_applications: dict[
        str,
        dict[str, FiniteChoiceApplicationOutput | None],
    ]
    choice_requirement_applications: dict[str, dict[str, dict[str, ChoiceRequirementApplicationOutput]]]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
