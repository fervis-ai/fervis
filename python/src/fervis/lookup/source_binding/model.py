"""Canonical source-binding request, plan, and evidence types."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from fervis.host_api.contracts import ParameterSemantics
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
    SourceFieldBinding,
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
from fervis.lookup.question_contract import (
    Aggregate,
    Quantify,
    RelatedRow,
    Coverage,
    Singleton,
    SetTerm,
    FactLocalRef,
    FactTerm,
    RawDataRecord,
    RequestedFactSemanticIndex,
)
from fervis.lookup.qualification import BooleanRequirementUseSite
from fervis.lookup.relation_catalog.row_sources import RowSourceValueType
from fervis.lookup.semantic_types import BooleanType, IdentifierType, TextType
from fervis.lookup.source_binding.param_values import compatible_fact_value_projections
from fervis.lookup.source_binding.subject_obligations import (
    derive_subject_choice_membership,
)
from fervis.types.enums import StrEnum


@dataclass(frozen=True)
class SourceStrategyBranch:
    branch_id: str
    source_refs: tuple[str, ...]
    relation_evidence_refs: tuple[str, ...]
    qualification_clause_refs: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSourceStrategy:
    """Available source scopes; only realized bindings authorize execution."""

    requested_fact_id: str
    branches: tuple[SourceStrategyBranch, ...]


class FactRealizationKind(StrEnum):
    RETURNED_FIELD = "RETURNED_FIELD"
    ENTITY_KEY = "ENTITY_KEY"


class AssociationRealizationKind(StrEnum):
    CO_RESIDENT = "CO_RESIDENT"
    DECLARED_RELATION = "DECLARED_RELATION"


class SourceMechanicKind(StrEnum):
    INVOCATION_PREDICATE = "INVOCATION_PREDICATE"
    RETURNED_ROW_PREDICATE = "RETURNED_ROW_PREDICATE"
    RETURNED_CHOICE_PREDICATE = "RETURNED_CHOICE_PREDICATE"


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
    reference_from_set_ref: str | None = None


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
    decision_basis: str
    baseline_included: bool
    explicit_user_override_applies: bool
    selection_requirement_refs: tuple[str, ...] = ()

    @property
    def included(self) -> bool:
        return derive_subject_choice_membership(
            baseline_included=self.baseline_included,
            explicit_user_override_applies=self.explicit_user_override_applies,
        ).included


@dataclass(frozen=True)
class SubjectSurfaceReview:
    owner_set_ref: str | None
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

    def choice_mechanics(
        self, branch_id: str, requirement_ref: str
    ) -> tuple[SourceMechanic, ...]:
        return tuple(
            mechanic
            for branch in self.branch_realizations
            if branch.branch_id == branch_id
            for surface in branch.surface_reviews
            if any(
                requirement_ref in choice.selection_requirement_refs
                for choice in surface.choice_reviews
            )
            for mechanic in surface.mechanics
            if mechanic.kind is SourceMechanicKind.RETURNED_CHOICE_PREDICATE
        )


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
class SourceRealization:
    request: SemanticSourceBindingRequest
    set_bindings: dict[str, tuple[SetRealization, ...]]
    fact_bindings: dict[str, tuple[FactRealization, ...]]
    association_bindings: dict[str, tuple[AssociationRealization, ...]]


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
    unrestricted_parameter_surfaces: tuple[tuple[str, str], ...] = ()
    realized_fact_fields: tuple[tuple[str, str, str, tuple[str, ...]], ...] = ()
    realized_set_sources: tuple[tuple[str, str, str], ...] = ()

    def row_references_for_set(self, set_ref: str) -> tuple[str, ...]:
        return self._connected_row_domains[set_ref]

    @cached_property
    def _connected_row_domains(self) -> dict[str, tuple[str, ...]]:
        from fervis.lookup.source_binding.row_domains import connected_row_domains

        return connected_row_domains(self)

    def _local_row_references_for_set(self, set_ref: str) -> tuple[str, ...]:
        """Row identities (or anonymous row sources) with the needed fields."""
        semantic_ref = FactLocalRef.from_token(set_ref)
        owned_refs = tuple(
            ref for ref in self.index.source_requirement_refs
            if isinstance((term := self.index.term_by_ref[ref]), FactTerm)
            and term.owner_ref == semantic_ref.local_id
            and not isinstance(term.value_type, IdentifierType)
        )
        observed_refs = tuple(ref for ref in owned_refs if ref in self.index.observed_fact_refs)
        eligible_identities = frozenset(self.identity_refs_for_set(set_ref))
        identity_required = any(
            isinstance(term, FactTerm)
            and isinstance(term.value_type, IdentifierType)
            and term.value_type.set_ref == semantic_ref.local_id
            for ref, term in self.index.term_by_ref.items()
            if ref in self.index.source_requirement_refs
        ) or any(
            output.value_ref == semantic_ref
            for output in self.index.output_requirements
        )
        refs: list[str] = []
        for source in self.source_catalog.sources:
            if not all(
                any(self._field_supports_fact(field, ref) for field in source.fields)
                or (
                    ref not in self.index.observed_fact_refs
                    and any((row_source_type_supports_semantic_type(param.type, self.index.value_type(ref))
                            or (param.choices and self._choice_surface_supports_fact(param.type, fact_ref=ref.token)))
                            and self._parameter_controls_rows(source.id, param.param_ref)
                            for param in source.params)
                )
                for ref in owned_refs
            ):
                continue
            from fervis.lookup.relation_catalog.row_sources.model import (
                RowSourceIdentityKind,
            )

            identities = tuple(
                identity.identity_ref
                for identity in source.identity_evidence
                if identity.identity_ref in eligible_identities
                and (
                    not observed_refs or identity.kind is RowSourceIdentityKind.ENTITY_ROW
                )
            )
            refs.extend(identities)
            if not identities and not identity_required:
                refs.append(source.id)
        return tuple(refs)

    def for_bindings(
        self,
        set_bindings: dict[str, tuple[SetRealization, ...]],
        fact_bindings: dict[str, tuple[FactRealization, ...]],
        association_bindings: dict[str, tuple[AssociationRealization, ...]],
    ) -> SemanticSourceBindingRequest:
        """Close an available scope over the sources its realizations use."""
        used: dict[str, set[str]] = {
            branch.branch_id: set() for branch in self.strategy.branches
        }
        for bindings in (set_bindings, fact_bindings):
            for values in bindings.values():
                for value in values:
                    if value.branch_id not in used:
                        raise ValueError("binding references an unknown branch")
                    used[value.branch_id].add(value.source_ref)
        for associations in association_bindings.values():
            for association in associations:
                if association.branch_id not in used:
                    raise ValueError("association references an unknown branch")
                used[association.branch_id].update(association.source_refs)
        branches = []
        for branch in self.strategy.branches:
            refs = used[branch.branch_id]
            if not refs or not refs <= set(branch.source_refs):
                raise ValueError(
                    "bound sources must be a nonempty subset of their scope"
                )
            subject_ref = self.index.subject_obligation.subject_set_ref.token
            roots = tuple(
                value.source_ref
                for value in set_bindings.get(subject_ref, ())
                if value.branch_id == branch.branch_id
            )
            if len(roots) != 1:
                raise ValueError("bound source graph requires one subject realization")
            ordered = [roots[0]]
            edges = tuple(
                set(value.source_refs)
                for values in association_bindings.values()
                for value in values
                if value.branch_id == branch.branch_id
                and value.relation_evidence_ref is not None
            )
            pending = refs - set(ordered)
            while pending:
                next_ref = next(
                    (
                        ref
                        for ref in branch.source_refs
                        if ref in pending
                        and any(
                            ref in edge and edge.intersection(ordered) for edge in edges
                        )
                    ),
                    None,
                )
                if next_ref is None:
                    # Independent aggregate domains may use disconnected producers.
                    next_ref = next(ref for ref in branch.source_refs if ref in pending)
                ordered.append(next_ref)
                pending.remove(next_ref)
            branches.append(
                replace(
                    branch,
                    source_refs=tuple(ordered),
                    relation_evidence_refs=tuple(
                        evidence.evidence_ref
                        for evidence in self.source_catalog.relation_evidence
                        if evidence.evidence_ref in branch.relation_evidence_refs
                        and {evidence.left_source_ref, evidence.right_source_ref}
                        <= refs
                    ),
                )
            )
        sources = frozenset(ref for branch in branches for ref in branch.source_refs)
        catalog = self.source_catalog.select(
            source_refs=sources,
            relation_evidence_refs=frozenset(
                ref for branch in branches for ref in branch.relation_evidence_refs
            ),
        )
        targets = {
            param.param_ref for source in catalog.sources for param in source.params
        }
        return replace(
            self,
            realized_set_sources=tuple(
                (ref, value.branch_id, value.source_ref)
                for ref, values in set_bindings.items() for value in values
            ),
            realized_fact_fields=self.realized_fact_fields or tuple(
                (ref, value.branch_id, value.source_ref, value.field_refs)
                for ref, values in fact_bindings.items() for value in values),
            strategy=replace(self.strategy, branches=tuple(branches)),
            source_catalog=catalog,
            catalog_values=tuple(
                value for value in self.catalog_values if value.target_ref in targets
            ),
        )

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

    def identity_refs_for_set(self, set_ref: str) -> tuple[str, ...]:
        """Identities compatible with every certified identifier of one set."""

        local_set_ref = FactLocalRef.from_token(set_ref)
        constrained_fact_refs = tuple(
            ref.token
            for ref in self.index.source_requirement_refs
            if isinstance(term := self.index.term_by_ref.get(ref), FactTerm)
            and isinstance(term.value_type, IdentifierType)
            and self.index.fact_local_ref_by_local_id[term.value_type.set_ref]
            == local_set_ref
        )
        allowed = tuple(
            frozenset(self.identity_refs_for_fact(fact_ref))
            for fact_ref in constrained_fact_refs
        )
        return tuple(
            evidence.identity_ref
            for evidence in self.source_catalog.identity_evidence
            if not allowed or all(evidence.identity_ref in refs for refs in allowed)
        )

    def requirement_fact_refs(self, requirement_ref: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                ref.token
                for ref in self.index.value_fact_refs(
                    self._requirement_value_ref(requirement_ref)
                )
            )
        )

    @property
    def model_authored_fact_refs(self) -> tuple[str, ...]:
        """Facts whose returned-field realization is not owned by a set identity."""

        return tuple(
            ref.token
            for ref in self.index.source_requirement_refs
            if isinstance(term := self.index.term_by_ref.get(ref), FactTerm)
            and not isinstance(term.value_type, IdentifierType)
        )

    @property
    def identifier_fact_refs(self) -> tuple[str, ...]:
        """Identifier facts derived from their identified set's selected identity."""

        return tuple(
            ref.token
            for ref in self.index.source_requirement_refs
            if isinstance(term := self.index.term_by_ref.get(ref), FactTerm)
            and isinstance(term.value_type, IdentifierType)
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
        assert isinstance(semantic_ref, FactLocalRef)
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
        term = self.index.term_by_ref[FactLocalRef.from_token(fact_ref)]
        assert isinstance(term, FactTerm)
        owner_ref = self.index.fact_local_ref_by_local_id[term.owner_ref]
        if isinstance(self.index.term_by_ref[owner_ref], SetTerm):
            catalog_source_ids = {source.id for source in self.source_catalog.sources}
            owner_sources = {
                ref
                if ref in catalog_source_ids
                else self.source_catalog.identity(ref).source_ref
                for ref in self.row_references_for_set(owner_ref.token)
            }
            source_refs = source_refs.intersection(owner_sources)
        return tuple(
            dict.fromkeys(
                SourceFieldBinding(source.id, field).ref
                for source in self.source_catalog.sources
                if source.id in source_refs
                for field in source.fields
                if self._field_supports_fact(field, semantic_ref)
            )
        )

    def _field_supports_fact(self, field, ref: FactLocalRef) -> bool:
        return bool(
            row_source_type_supports_semantic_type(field.type, self.index.value_type(ref))
            or (ref not in self.index.observed_fact_refs and field.finite_choices
                and self._choice_surface_supports_fact(field.type, fact_ref=ref.token))
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
        subject_binding: SubjectObligationBinding | None = None,
    ) -> dict[str, tuple[str, ...]]:
        branch_ids = tuple(branch.branch_id for branch in self.strategy.branches)
        required: dict[str, set[str]] = {
            fact_ref: set(branch_ids) for fact_ref in self.observed_fact_refs
        }
        application_owners = {
            (application.branch_id, application.owner_ref)
            for application in applications
        }
        for requirement in self.index.boolean_requirements:
            fact_refs = self.requirement_fact_refs(requirement.requirement_ref)
            for branch_id in branch_ids:
                if (branch_id, requirement.requirement_ref) in application_owners or (
                    subject_binding is not None
                    and subject_binding.choice_mechanics(
                        branch_id, requirement.requirement_ref
                    )
                ):
                    continue
                for fact_ref in fact_refs:
                    required.setdefault(fact_ref, set()).add(branch_id)
        return {
            ref: tuple(branch_id for branch_id in branch_ids if branch_id in branches)
            for ref, branches in required.items()
        }

    @property
    def observed_fact_refs(self) -> frozenset[str]:
        return frozenset(ref.token for ref in self.index.observed_fact_refs)

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
                    target_ref=self.source_catalog.choice_surface(
                        value.surface_ref
                    ).target_ref,
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
            surface.target_ref
            for surface in self.source_catalog.choice_surfaces
            if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
        )

    @property
    def raw_subject_scope_surfaces(self) -> dict[str, SourceChoiceSurface]:
        """A raw-record subject must explicitly reconsider scoped defaults."""
        if not isinstance(self.index.subject_obligation, RawDataRecord):
            return {}
        return {
            f"subject_scope:{surface.surface_ref}": surface
            for surface in self.source_catalog.choice_surfaces
            if surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
            and self._parameter_controls_rows(surface.source_ref, surface.target_ref)
            and any(
                param.param_ref == surface.target_ref and param.default is not None
                for param in self.source_catalog.source(surface.source_ref).params
            )
        }

    @property
    def invocation_application_owner_refs(self) -> tuple[str, ...]:
        requirement_refs = tuple(
            item.requirement_ref for item in self.index.boolean_requirements
        )
        source_required_refs = tuple(
            f"source_required:{target_ref}"
            for target_ref in self._unowned_required_target_refs
        )
        return (
            *requirement_refs,
            *source_required_refs,
            *self.raw_subject_scope_surfaces,
        )

    def invocation_preserves_population(
        self, owner_ref: str, *, branch_id: str | None = None
    ) -> bool:
        """A local expression predicate cannot narrow other read consumers."""
        requirement = next(
            (
                item
                for item in self.index.boolean_requirements
                if item.requirement_ref == owner_ref
            ),
            None,
        )
        if requirement is None:
            return True
        if requirement.use_site is BooleanRequirementUseSite.POPULATION:
            clauses = self.index.qualification.clauses
            if branch_id is not None:
                branch = next(
                    item
                    for item in self.strategy.branches
                    if item.branch_id == branch_id
                )
                clauses = tuple(
                    clause
                    for clause in clauses
                    if clause.clause_ref in branch.qualification_clause_refs
                )
            return bool(clauses) and all(
                requirement.atom_ref in clause.atom_refs for clause in clauses
            )
        if requirement.use_site is not BooleanRequirementUseSite.AGGREGATE_FILTER:
            return False
        # Grouped aggregates must retain groups with no matching rows, too.
        if not isinstance(self.index.result_grain, Singleton):
            return False
        aggregates = tuple(
            node
            for node in self.index.requested_fact.expressions
            if isinstance(node, Aggregate)
        )
        owner = next(
            (
                node
                for node in aggregates
                if self.index.fact_local_ref_by_local_id[node.id].token
                == requirement.owner_expression_ref
            ),
            None,
        )
        return owner is not None and all(
            node.filter_ref == owner.filter_ref for node in aggregates
        )

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
        scope_surface = self.raw_subject_scope_surfaces.get(owner_ref)
        if scope_surface is not None:
            return tuple(
                option
                for option in self.invocation_projection_options
                if option.source_ref == scope_surface.source_ref
                and option.source_ref in branch.source_refs
                and option.target_ref == scope_surface.target_ref
            )
        if owner_ref in requirement_refs:
            if not self.invocation_preserves_population(owner_ref, branch_id=branch_id):
                return ()
            value_refs = frozenset(self.requirement_value_refs(owner_ref))
            from fervis.lookup.question_contract.domains import value_set_dependencies

            dependencies = value_set_dependencies(self.index, self._requirement_value_ref(owner_ref))
            owner_sources = {
                source for ref, branch_ref, source in self.realized_set_sources
                if branch_ref == branch_id and ref in dependencies
            }
            return tuple(
                option
                for option in self.authored_invocation_projection_options
                if option.source_ref in branch.source_refs
                and option.value_ref in value_refs
                and (not self.realized_set_sources or option.source_ref in owner_sources)
                and self._parameter_controls_rows(option.source_ref, option.target_ref)
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
        scope_surface = self.raw_subject_scope_surfaces.get(owner_ref)
        if scope_surface is not None:
            return (scope_surface.target_ref,)
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

    def _parameter_controls_rows(self, source_ref: str, target_ref: str) -> bool:
        return any(param.param_ref == target_ref and param.semantics is ParameterSemantics.OPAQUE_QUERY_PARAM
                   for param in self.source_catalog.source(source_ref).params)

    def _surface_matches_realized_fact(self, surface: SourceChoiceSurface, fact_ref: str, branch_id: str) -> bool:
        if surface.kind is not SourceChoiceSurfaceKind.RETURNED_FIELD:
            return self._parameter_controls_rows(surface.source_ref, surface.target_ref)
        chosen = tuple((source, fields) for ref, branch, source, fields in self.realized_fact_fields
                       if ref == fact_ref and branch == branch_id)
        return not chosen or any(surface.source_ref == source and surface.target_ref in fields
                                 for source, fields in chosen)

    def _requirement_is_row_predicate(self, requirement_ref: str) -> bool:
        ref = self._requirement_value_ref(requirement_ref)
        dependencies = {ref, *self.index.transitive_dependencies_by_ref.get(ref, ())}
        return not any(
            isinstance(
                self.index.expression_by_ref.get(item),
                (Aggregate, Quantify, RelatedRow, Coverage),
            )
            for item in dependencies
            if isinstance(item, FactLocalRef)
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
                    getattr(item, "param_ref", None) or getattr(item, "field_ref", None)
                )
                == surface.target_ref
            ),
            None,
        )
        if source_type is None:
            raise ValueError("choice surface lacks its declared source contract")
        boolean_refs = tuple(
            requirement.requirement_ref
            for requirement in self.index.boolean_requirements
            if self._requirement_is_row_predicate(requirement.requirement_ref)
            and (
                surface.kind is SourceChoiceSurfaceKind.RETURNED_FIELD
                or not self.direct_value_options_for_owner(
                    requirement.requirement_ref,
                    branch_id=branch_id,
                )
            )
            and (fact_refs := self.requirement_fact_refs(requirement.requirement_ref))
            and len(fact_refs) == 1
            and all(self._surface_matches_realized_fact(surface, ref, branch_id) for ref in fact_refs)
            and all(
                self._choice_surface_supports_fact(
                    source_type,
                    fact_ref=fact_ref,
                )
                for fact_ref in fact_refs
            )
        )
        required_owner = self._required_target_owners.get(surface.target_ref)
        return (
            *boolean_refs,
            *(
                owner
                for owner, scoped in self.raw_subject_scope_surfaces.items()
                if scoped.surface_ref == surface.surface_ref
            ),
            *(
                (required_owner,)
                if required_owner == f"source_required:{surface.target_ref}"
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

        if not self.invocation_preserves_population(owner_ref, branch_id=branch_id):
            return ()
        options: list[tuple[SourceChoiceSurface, tuple[SourceChoiceValue, ...]]] = []
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
        if isinstance(semantic_type, IdentifierType):
            return False
        return row_source_type_supports_semantic_type(
            source_type,
            semantic_type,
        ) or (
            source_type in {RowSourceValueType.CHOICE, RowSourceValueType.BOOLEAN}
            and isinstance(semantic_type, (BooleanType, TextType))
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
        # A logical Boolean property need not have the same polarity as an
        # API flag (for example, inactive can bind is_active=false).
        return requirement_refs

    def explicit_subject_requirement_refs(
        self, choice: SourceChoiceValue, *, branch_id: str
    ) -> tuple[str, ...]:
        """Explicit row predicates retain their scope when overriding membership."""
        return self.choice_value_requirement_refs(choice, branch_id=branch_id)

    def choice_surface_requires_application(
        self,
        surface: SourceChoiceSurface,
    ) -> bool:
        """Whether a request choice must be supplied for source invocation."""

        if surface.kind is not SourceChoiceSurfaceKind.REQUEST_PARAMETER:
            return False
        source = self.source_catalog.source(surface.source_ref)
        param = next(
            item for item in source.params if item.param_ref == surface.target_ref
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
                    if self._parameter_controls_rows(source.id, param.param_ref) and any(
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
