"""Provider DTOs for semantic Source Binding."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class SetRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    source_ref: str
    identity_ref: str | None


@dataclass(frozen=True)
class IdentifierFactRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    source_ref: str


@dataclass(frozen=True)
class ReturnedFactRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    source_ref: str
    field_ref: str


@dataclass(frozen=True)
class AssociationRealizationOutput(ProviderOutput):
    branch_id: str
    mapping_basis: str
    realization_ref: str


@dataclass(frozen=True)
class ResolvedInputApplicationOutput(ProviderOutput):
    mapping_basis: str
    owner_ref: str
    value_ref: str
    value_component: str
    target_ref: str


@dataclass(frozen=True)
class FiniteChoiceApplicationOutput(ProviderOutput):
    application_basis: str
    surface_ref: str
    selected_choice_values: tuple[str, ...]


@dataclass(frozen=True)
class SubjectChoiceReviewOutput(ProviderOutput):
    choice_domain_meaning: str
    role_match_basis: str
    matched_excluded_role: str
    choice_inclusion_basis: str
    choice_inclusion: str


@dataclass(frozen=True)
class SubjectSurfaceReviewOutput(ProviderOutput):
    surface_mapping_basis: str
    choice_reviews: dict[str, SubjectChoiceReviewOutput]


@dataclass(frozen=True)
class SubjectObligationRealizationOutput(ProviderOutput):
    branch_id: str
    finite_choice_reviews: dict[str, SubjectSurfaceReviewOutput]


@dataclass(frozen=True)
class SubjectObligationBindingOutput(ProviderOutput):
    subject_ref: str
    branch_realizations: tuple[SubjectObligationRealizationOutput, ...]


@dataclass(frozen=True)
class SemanticSourceBindingOutput(ProviderOutput):
    set_bindings: dict[str, tuple[SetRealizationOutput, ...]]
    resolved_input_applications: dict[
        str,
        tuple[ResolvedInputApplicationOutput, ...],
    ]
    finite_choice_applications: dict[
        str,
        dict[str, FiniteChoiceApplicationOutput],
    ]
    fact_bindings: dict[
        str,
        tuple[IdentifierFactRealizationOutput | ReturnedFactRealizationOutput, ...],
    ]
    association_bindings: dict[str, tuple[AssociationRealizationOutput, ...]]
    subject_binding: SubjectObligationBindingOutput


__all__ = tuple(name for name in globals() if not name.startswith("_"))
