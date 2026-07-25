"""Canonical source-binding request, plan, and evidence types."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from fervis.lookup.relation_catalog.model import requires_caller_supplied_input
from fervis.lookup.relation_catalog.row_sources import (
    row_source_type_supports_semantic_type,
)

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceChoiceSurface,
    SourceChoiceSurfaceKind,
    SourceChoiceValue,
)
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralType,
    LiteralValuePayload,
    ValueProjectionKind,
)
from fervis.lookup.plan_selection import CandidateSourceStrategy
from fervis.lookup.question_contract import (
    FactLocalKind,
    FactLocalRef,
    FactTerm,
    RawDataRecord,
    RequestedFactSemanticIndex,
)
from fervis.lookup.qualification import BooleanPolarity
from fervis.lookup.relation_catalog.row_sources import RowSourceValueType
from fervis.lookup.semantic_types import BooleanType
from fervis.lookup.source_binding.param_values import compatible_fact_value_projections
from fervis.lookup.source_binding.subject_obligations import (
    derive_subject_choice_membership,
)
from fervis.types.enums import StrEnum


class FactRealizationKind(StrEnum):
    RETURNED_FIELD = "RETURNED_FIELD"
    ENTITY_KEY = "ENTITY_KEY"


class AssociationRealizationKind(StrEnum):
    CO_RESIDENT = "CO_RESIDENT"
    DECLARED_RELATION = "DECLARED_RELATION"


class SourceMechanicKind(StrEnum):
    INVOCATION_PREDICATE = "INVOCATION_PREDICATE"
    RETURNED_ROW_PREDICATE = "RETURNED_ROW_PREDICATE"


@dataclass(frozen=True)
class SetRealization:
    branch_id: str
    mapping_basis: str
    source_ref: str
    identity_ref: str | None
    identity_field_refs: tuple[str, ...]
    contract_evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class FactRealization:
    branch_id: str
    mapping_basis: str
    source_ref: str
    kind: FactRealizationKind
    identity_ref: str | None
    field_refs: tuple[str, ...]
    contract_evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class AssociationRealization:
    branch_id: str
    mapping_basis: str
    kind: AssociationRealizationKind
    source_refs: tuple[str, ...]
    relation_evidence_ref: str | None
    contract_evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class SourceMechanic:
    mapping_basis: str
    source_ref: str
    application_refs: tuple[str, ...]
    contract_evidence_refs: tuple[str, ...]
    kind: SourceMechanicKind


@dataclass(frozen=True)
class InvocationTargetApplication:
    mapping_basis: str
    target_ref: str
    value_ref: str
    projection: ValueProjectionKind
    component_ref: str | None


@dataclass(frozen=True)
class InvocationValueApplication:
    application_ref: str
    branch_id: str
    source_ref: str
    value_ref: str
    owner_ref: str | None
    target_applications: tuple[InvocationTargetApplication, ...]


@dataclass(frozen=True)
class InvocationProjectionOption:
    option_ref: str
    source_ref: str
    target_ref: str
    value_ref: str
    projection: ValueProjectionKind
    component_ref: str | None


@dataclass(frozen=True)
class MissingCatalogValue:
    catalog_input_ref: str
    source_ref: str
    parameter_id: str
    target_ref: str
    label: str
    value_type: str
    allowed_values: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class CatalogProvidedValue:
    catalog_input_ref: str
    target_ref: str
    value_id: str
    typed_value: FactValue
    certification_refs: tuple[str, ...]


@dataclass(frozen=True)
class BooleanRequirementRealization:
    branch_id: str
    mechanics: tuple[SourceMechanic, ...]


@dataclass(frozen=True)
class SubjectChoiceReview:
    choice_ref: str
    choice_domain_meaning: str
    role_match_basis: str
    matched_excluded_role: str | None
    explicit_user_override_basis: str
    choice_inclusion_basis: str
    choice_included: bool
    authored_explicit_user_override_applies: bool

    @property
    def included(self) -> bool:
        return derive_subject_choice_membership(
            choice_included=self.choice_included,
            matched_excluded_role=self.matched_excluded_role,
            explicit_user_override_applies=(
                self.authored_explicit_user_override_applies
            ),
        ).included

    @property
    def explicit_user_override_applies(self) -> bool:
        return derive_subject_choice_membership(
            choice_included=self.choice_included,
            matched_excluded_role=self.matched_excluded_role,
            explicit_user_override_applies=(
                self.authored_explicit_user_override_applies
            ),
        ).explicit_user_override_applies


@dataclass(frozen=True)
class SubjectSurfaceReview:
    surface_ref: str
    surface_mapping_basis: str
    choice_reviews: tuple[SubjectChoiceReview, ...]
    included_choice_refs: tuple[str, ...]
    mechanics: tuple[SourceMechanic, ...]


@dataclass(frozen=True)
class SubjectObligationRealization:
    branch_id: str
    surface_reviews: tuple[SubjectSurfaceReview, ...]


@dataclass(frozen=True)
class SubjectObligationBinding:
    subject_ref: str
    branch_realizations: tuple[SubjectObligationRealization, ...]


@dataclass(frozen=True)
class SourceBindingPlan:
    strategy: CandidateSourceStrategy
    set_bindings: dict[str, tuple[SetRealization, ...]]
    fact_bindings: dict[str, tuple[FactRealization, ...]]
    association_bindings: dict[str, tuple[AssociationRealization, ...]]
    invocation_applications: tuple[InvocationValueApplication, ...]
    boolean_bindings: dict[str, tuple[BooleanRequirementRealization, ...]]
    subject_binding: SubjectObligationBinding


@dataclass(frozen=True)
class BoundSourcePlan:
    plan: SourceBindingPlan


@dataclass(frozen=True)
class SourceBindingClarification:
    requested_fact_id: str
    missing_catalog_values: tuple[MissingCatalogValue, ...]


@dataclass(frozen=True)
class SemanticSourceBindingRequest:
    index: RequestedFactSemanticIndex
    strategy: CandidateSourceStrategy
    source_catalog: AvailableSourceCatalog
    canonical_values: tuple[CanonicalInputValue, ...]
    catalog_values: tuple[CatalogProvidedValue, ...] = ()

    def __post_init__(self) -> None:
        value_ids = tuple(
            item.canonical_value_id for item in self.canonical_values
        ) + tuple(item.value_id for item in self.catalog_values)
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("source binding value IDs must be unique")
        target_refs = {
            param.param_ref
            for source in self.source_catalog.sources
            for param in source.params
        }
        if any(item.target_ref not in target_refs for item in self.catalog_values):
            raise ValueError("catalog-provided value references an unknown target")

    def requirement_value_refs(self, requirement_ref: str) -> tuple[str, ...]:
        requirement_ref_value = self._requirement_value_ref(requirement_ref)
        input_refs = {
            item
            for item in self.index.direct_dependencies_by_ref.get(
                requirement_ref_value, ()
            )
            if isinstance(item, str)
        }
        canonical_refs = tuple(
            value.canonical_value_id
            for value in self.canonical_values
            if value.input_ref in input_refs
        )
        allows_declared_choice = any(
            value.canonical_value_id in canonical_refs
            and isinstance(value.typed_value.payload, LiteralValuePayload)
            and value.typed_value.payload.literal_type is LiteralType.STRING
            for value in self.canonical_values
        )
        return canonical_refs + (
            tuple(value.value_ref for value in self.source_catalog.choice_values)
            if allows_declared_choice
            else ()
        )

    def identity_refs_for_fact(self, fact_ref: str) -> tuple[str, ...]:
        """Declared identities compatible with an already-certified input identity."""

        use_refs = {
            use.use_ref
            for use in self.index.input_use_sites
            if use.reference_fact_ref is not None
            and use.reference_fact_ref.token == fact_ref
        }
        contracts = {
            (payload.entity_kind, payload.key_id)
            for value in self.canonical_values
            if use_refs.intersection(value.use_refs)
            for payload in (value.typed_value.payload,)
            if isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload))
        }
        return tuple(
            evidence.identity_ref
            for evidence in self.source_catalog.identity_evidence
            if not contracts or (evidence.entity_kind, evidence.key_id) in contracts
        )

    def requirement_fact_refs(self, requirement_ref: str) -> tuple[str, ...]:
        requirement_ref_value = self._requirement_value_ref(requirement_ref)
        dependencies = {
            requirement_ref_value,
            *self.index.transitive_dependencies_by_ref.get(requirement_ref_value, ()),
        }
        return tuple(
            ref.token
            for ref in dependencies
            if isinstance(ref, FactLocalRef) and ref.kind is FactLocalKind.FACT
        )

    def returned_field_refs_for_fact(
        self,
        fact_ref: str,
        *,
        branch_id: str | None = None,
    ) -> tuple[str, ...]:
        """Fields that can realize one fact in the selected source boundary."""

        semantic_ref = next(
            ref
            for ref in self.index.inferred_type_by_ref
            if getattr(ref, "token", None) == fact_ref
        )
        semantic_type = self.index.inferred_type_by_ref[semantic_ref]
        source_refs = (
            frozenset(
                next(
                    branch
                    for branch in self.strategy.branches
                    if branch.branch_id == branch_id
                ).source_refs
            )
            if branch_id is not None
            else frozenset(source.id for source in self.source_catalog.sources)
        )
        return tuple(
            dict.fromkeys(
                field.field_ref
                for source in self.source_catalog.sources
                if source.id in source_refs
                for field in source.fields
                if row_source_type_supports_semantic_type(field.type, semantic_type)
            )
        )

    def _requirement_value_ref(self, requirement_ref: str) -> FactLocalRef:
        requirement = next(
            item
            for item in self.index.boolean_requirements
            if item.requirement_ref == requirement_ref
        )
        return next(
            ref
            for ref in (
                *self.index.term_by_ref,
                *self.index.expression_by_ref,
            )
            if ref.token == requirement.atom_ref.value_ref
        )

    def required_fact_branches(
        self,
        applications: tuple[InvocationValueApplication, ...],
    ) -> dict[str, tuple[str, ...]]:
        branch_ids = tuple(branch.branch_id for branch in self.strategy.branches)
        required: dict[str, set[str]] = {
            fact_ref: set(branch_ids) for fact_ref in self._observed_fact_refs
        }
        application_owners = {
            (application.branch_id, application.owner_ref)
            for application in applications
        }
        for requirement in self.index.boolean_requirements:
            fact_refs = self.requirement_fact_refs(requirement.requirement_ref)
            for branch_id in branch_ids:
                if (branch_id, requirement.requirement_ref) in application_owners:
                    continue
                for fact_ref in fact_refs:
                    required.setdefault(fact_ref, set()).add(branch_id)
        return {
            ref: tuple(branch_id for branch_id in branch_ids if branch_id in branches)
            for ref, branches in required.items()
        }

    @property
    def _observed_fact_refs(self) -> frozenset[str]:
        dependencies = {
            dependency
            for output_requirement in self.index.output_requirements
            for dependency in output_requirement.dependencies
        }
        for ref in (*self.index.grouping_refs, *self.index.ordering_refs):
            dependencies.add(ref)
            dependencies.update(self.index.transitive_dependencies_by_ref.get(ref, ()))
        return frozenset(
            ref.token
            for ref in dependencies
            if isinstance(ref, FactLocalRef) and ref.kind is FactLocalKind.FACT
        )

    @property
    def invocation_projection_options(self) -> tuple[InvocationProjectionOption, ...]:
        return (
            tuple(
                option
                for source in self.source_catalog.sources
                for value in self.canonical_values
                for param in source.params
                for option in _canonical_projection_options(
                    source_ref=source.id,
                    target=param,
                    value_ref=value.canonical_value_id,
                    typed_value=value.typed_value,
                )
            )
            + tuple(
                option
                for source in self.source_catalog.sources
                for value in self.catalog_values
                for param in source.params
                if param.param_ref == value.target_ref
                for option in _canonical_projection_options(
                    source_ref=source.id,
                    target=param,
                    value_ref=value.value_id,
                    typed_value=value.typed_value,
                )
            )
            + tuple(
                _invocation_projection_option(
                    source_ref=value.source_ref,
                    target_ref=value.surface_ref,
                    value_ref=value.value_ref,
                    projection=ValueProjectionKind.WHOLE_VALUE,
                    component_ref=None,
                )
                for value in self.source_catalog.choice_values
                if value.surface_kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
            )
        )

    @property
    def authored_invocation_projection_options(
        self,
    ) -> tuple[InvocationProjectionOption, ...]:
        """Bounded value-to-parameter options shown for model-authored applications."""

        reviewed_targets = self.reviewed_choice_parameter_target_refs
        return tuple(
            option
            for option in self.invocation_projection_options
            if option.target_ref not in reviewed_targets
        )

    @property
    def reviewed_choice_parameter_target_refs(self) -> frozenset[str]:
        if isinstance(self.index.subject_obligation, RawDataRecord):
            return frozenset()
        return frozenset(
            surface.surface_ref
            for surface in self.source_catalog.choice_surfaces
            if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
        )

    @property
    def invocation_application_owner_refs(self) -> tuple[str, ...]:
        requirement_refs = tuple(
            item.requirement_ref for item in self.index.boolean_requirements
        )
        source_required_refs = tuple(
            f"source_required:{target_ref}"
            for target_ref in self._unowned_required_target_refs
        )
        return (*requirement_refs, *source_required_refs)

    def invocation_options_for_owner(
        self,
        owner_ref: str,
        *,
        branch_id: str,
    ) -> tuple[InvocationProjectionOption, ...]:
        branch = next(
            item for item in self.strategy.branches if item.branch_id == branch_id
        )
        requirement_refs = {
            item.requirement_ref for item in self.index.boolean_requirements
        }
        if owner_ref in requirement_refs:
            value_refs = frozenset(self.requirement_value_refs(owner_ref))
            return tuple(
                option
                for option in self.authored_invocation_projection_options
                if option.source_ref in branch.source_refs
                and option.value_ref in value_refs
            )
        prefix = "source_required:"
        if not owner_ref.startswith(prefix):
            return ()
        target_ref = owner_ref[len(prefix) :]
        if target_ref not in self._unowned_required_target_refs:
            return ()
        return tuple(
            option
            for option in self.authored_invocation_projection_options
            if option.source_ref in branch.source_refs
            and option.target_ref == target_ref
        )

    def direct_value_options_for_owner(
        self,
        owner_ref: str,
        *,
        branch_id: str,
    ) -> tuple[InvocationProjectionOption, ...]:
        choice_refs = {value.value_ref for value in self.source_catalog.choice_values}
        boolean_literal_refs = {
            value.canonical_value_id
            for value in self.canonical_values
            if isinstance(value.typed_value.payload, LiteralValuePayload)
            and value.typed_value.payload.literal_type is LiteralType.BOOLEAN
        }
        return tuple(
            option
            for option in self.invocation_options_for_owner(
                owner_ref,
                branch_id=branch_id,
            )
            if option.value_ref not in choice_refs
            and option.value_ref not in boolean_literal_refs
        )

    def required_invocation_target_refs(
        self,
        owner_ref: str,
        *,
        branch_id: str,
    ) -> tuple[str, ...]:
        branch = next(
            item for item in self.strategy.branches if item.branch_id == branch_id
        )
        return tuple(
            target_ref
            for target_ref, target_owner in self._required_target_owners.items()
            if target_owner == owner_ref
            and any(
                source.id in branch.source_refs
                and any(param.param_ref == target_ref for param in source.params)
                for source in self.source_catalog.sources
            )
        )

    def choice_requirement_refs(
        self,
        surface: SourceChoiceSurface,
        *,
        branch_id: str,
    ) -> tuple[str, ...]:
        """Requirement owners compatible with one declared choice surface."""

        branch = next(
            item for item in self.strategy.branches if item.branch_id == branch_id
        )
        if surface.source_ref not in branch.source_refs:
            return ()
        source = self.source_catalog.source(surface.source_ref)
        source_type = next(
            (
                item.type
                for item in (
                    source.params
                    if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
                    else source.fields
                )
                if (
                    getattr(item, "param_ref", None)
                    or getattr(item, "field_ref", None)
                )
                == surface.surface_ref
            ),
            None,
        )
        if source_type is None:
            raise ValueError("choice surface lacks its declared source contract")
        boolean_refs = tuple(
            requirement.requirement_ref
            for requirement in self.index.boolean_requirements
            if not self.direct_value_options_for_owner(
                requirement.requirement_ref,
                branch_id=branch_id,
            )
            and (
                fact_refs := self.requirement_fact_refs(
                    requirement.requirement_ref
                )
            )
            and all(
                self._choice_surface_supports_fact(
                    source_type,
                    fact_ref=fact_ref,
                )
                for fact_ref in fact_refs
            )
        )
        required_owner = self._required_target_owners.get(surface.surface_ref)
        return (
            *boolean_refs,
            *(
                (required_owner,)
                if required_owner == f"source_required:{surface.surface_ref}"
                else ()
            ),
        )

    def finite_choice_options_for_owner(
        self,
        owner_ref: str,
        *,
        branch_id: str,
    ) -> tuple[tuple[SourceChoiceSurface, tuple[SourceChoiceValue, ...]], ...]:
        """Request-parameter choice surfaces that can realize one owner."""

        options: list[
            tuple[SourceChoiceSurface, tuple[SourceChoiceValue, ...]]
        ] = []
        for surface in self.source_catalog.choice_surfaces:
            if surface.kind is not SourceChoiceSurfaceKind.REQUEST_PARAMETER:
                continue
            compatible_choices = tuple(
                choice
                for choice in surface.values
                if owner_ref
                in self.choice_value_requirement_refs(
                    choice,
                    branch_id=branch_id,
                )
            )
            if compatible_choices:
                options.append((surface, compatible_choices))
        return tuple(options)

    def _choice_surface_supports_fact(
        self,
        source_type: RowSourceValueType,
        *,
        fact_ref: str,
    ) -> bool:
        semantic_type = self.index.inferred_type_by_ref[
            next(
                ref
                for ref in self.index.inferred_type_by_ref
                if getattr(ref, "token", None) == fact_ref
            )
        ]
        return row_source_type_supports_semantic_type(
            source_type,
            semantic_type,
        ) or (
            source_type is RowSourceValueType.CHOICE
            and isinstance(semantic_type, BooleanType)
        )

    def choice_value_requirement_refs(
        self,
        choice: SourceChoiceValue,
        *,
        branch_id: str,
    ) -> tuple[str, ...]:
        """Requirement owners mechanically compatible with one declared choice."""

        surface = self.source_catalog.choice_surface(choice.surface_ref)
        requirement_refs = self.choice_requirement_refs(
            surface,
            branch_id=branch_id,
        )
        choice_truth = self._declared_boolean_choice_truth(choice, surface=surface)
        if choice_truth is None:
            return requirement_refs
        return tuple(
            requirement_ref
            for requirement_ref in requirement_refs
            if (
                required_truth := self._direct_boolean_requirement_truth(
                    requirement_ref
                )
            )
            is None
            or required_truth is choice_truth
        )

    def _declared_boolean_choice_truth(
        self,
        choice: SourceChoiceValue,
        *,
        surface: SourceChoiceSurface,
    ) -> bool | None:
        source = self.source_catalog.source(surface.source_ref)
        declarations = (
            source.params
            if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
            else source.fields
        )
        declaration = next(
            item
            for item in declarations
            if (
                getattr(item, "param_ref", None)
                or getattr(item, "field_ref", None)
            )
            == surface.surface_ref
        )
        if declaration.type is not RowSourceValueType.BOOLEAN:
            return None
        if isinstance(choice.value, bool):
            return choice.value
        if isinstance(choice.value, str):
            normalized = choice.value.casefold()
            if normalized in {"true", "false"}:
                return normalized == "true"
        return None

    def _direct_boolean_requirement_truth(
        self,
        requirement_ref: str,
    ) -> bool | None:
        requirement = next(
            (
                item
                for item in self.index.boolean_requirements
                if item.requirement_ref == requirement_ref
            ),
            None,
        )
        if requirement is None:
            return None
        value_ref = FactLocalRef.from_token(requirement.atom_ref.value_ref)
        term = self.index.term_by_ref.get(value_ref)
        if not isinstance(term, FactTerm) or not isinstance(
            term.value_type, BooleanType
        ):
            return None
        return requirement.atom_ref.polarity is BooleanPolarity.POSITIVE

    def choice_surface_requires_application(
        self,
        surface: SourceChoiceSurface,
    ) -> bool:
        """Whether a request choice must be supplied for source invocation."""

        if surface.kind is not SourceChoiceSurfaceKind.REQUEST_PARAMETER:
            return False
        source = self.source_catalog.source(surface.source_ref)
        param = next(
            item for item in source.params if item.param_ref == surface.surface_ref
        )
        return requires_caller_supplied_input(param)

    @property
    def _required_target_owners(self) -> dict[str, str]:
        requirement_refs = tuple(
            item.requirement_ref for item in self.index.boolean_requirements
        )
        owners: dict[str, str] = {}
        for source in self.source_catalog.sources:
            for param in source.params:
                if not param.required or param.default is not None:
                    continue
                matching_requirements = tuple(
                    requirement_ref
                    for requirement_ref in requirement_refs
                    if any(
                        option.target_ref == param.param_ref
                        and option.value_ref
                        in self.requirement_value_refs(requirement_ref)
                        for option in self.authored_invocation_projection_options
                    )
                )
                owners[param.param_ref] = (
                    matching_requirements[0]
                    if len(matching_requirements) == 1
                    else f"source_required:{param.param_ref}"
                )
        return owners

    @property
    def _unowned_required_target_refs(self) -> tuple[str, ...]:
        return tuple(
            target_ref
            for target_ref, owner_ref in self._required_target_owners.items()
            if owner_ref == f"source_required:{target_ref}"
        )


def source_binding_clarification(
    request: SemanticSourceBindingRequest,
) -> SourceBindingClarification | None:
    options = {
        (item.source_ref, item.target_ref)
        for item in request.invocation_projection_options
    }
    missing: list[MissingCatalogValue] = []
    for branch in request.strategy.branches:
        for source_ref in branch.source_refs:
            source = request.source_catalog.source(source_ref)
            for param in source.params:
                if (
                    not param.required
                    or param.default is not None
                    or (source_ref, param.param_ref) in options
                ):
                    continue
                missing.append(
                    MissingCatalogValue(
                        catalog_input_ref=(
                            f"catalog_input:{branch.branch_id}:{param.param_ref}"
                        ),
                        source_ref=source_ref,
                        parameter_id=param.id,
                        target_ref=param.param_ref,
                        label=param.description or param.name,
                        value_type=param.type.value,
                        allowed_values=tuple(str(item) for item in param.choices),
                        evidence_refs=(
                            request.source_catalog.contract_snapshot.ref,
                            source_ref,
                            param.param_ref,
                        ),
                    )
                )
    return (
        SourceBindingClarification(request.index.requested_fact_id, tuple(missing))
        if missing
        else None
    )


def source_required_inputs_are_satisfiable(
    source,
    *,
    values: tuple[FactValue, ...],
) -> bool:
    """Whether current values can supply every caller-required source input."""

    return all(
        not requires_caller_supplied_input(param)
        or bool(param.choices)
        or any(
            compatible_fact_value_projections(
                value,
                type_name=param.type.value,
                choices=param.choices,
                entity_target=param.entity_target,
            )
            for value in values
        )
        for param in source.params
    )


def _canonical_projection_options(
    *,
    source_ref: str,
    target,
    value_ref: str,
    typed_value: FactValue,
) -> tuple[InvocationProjectionOption, ...]:
    return tuple(
        _invocation_projection_option(
            source_ref=source_ref,
            target_ref=target.param_ref,
            value_ref=value_ref,
            projection=projection,
            component_ref=component_ref,
        )
        for projection, component_ref in compatible_fact_value_projections(
            typed_value,
            type_name=target.type.value,
            choices=target.choices,
            entity_target=target.entity_target,
        )
    )


def _invocation_projection_option(
    *,
    source_ref: str,
    target_ref: str,
    value_ref: str,
    projection: ValueProjectionKind,
    component_ref: str | None,
) -> InvocationProjectionOption:
    identity = "\x1f".join(
        (
            source_ref,
            target_ref,
            value_ref,
            projection.value,
            component_ref or "",
        )
    )
    return InvocationProjectionOption(
        option_ref=(
            f"invocation_projection:{projection.value}:"
            f"{sha256(identity.encode()).hexdigest()[:16]}"
        ),
        source_ref=source_ref,
        target_ref=target_ref,
        value_ref=value_ref,
        projection=projection,
        component_ref=component_ref,
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
