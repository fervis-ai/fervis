"""Closed relation model for canonical executable answer programs."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum

from fervis.lookup.answer_program.values import ParameterRef, ValueExpression
from fervis.lookup.question_contract import MembershipTestRef


class FieldBindingRole(StrEnum):
    IDENTITY = "identity"
    OUTPUT = "output"
    PREDICATE = "predicate"


class SourceKind(StrEnum):
    API_READ = "api_read"
    GENERATED_CALENDAR = "generated_calendar"
    MEMORY_READ = "memory_read"


class PopulationChoiceControllerKind(StrEnum):
    QUERY_PARAM = "query_param"
    ROW_PREDICATE = "row_predicate"


class ReviewScopeDecisionKind(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


class PopulationCoverageRole(StrEnum):
    ROW_POPULATION = "row_population"
    OPERATION_CONDITION = "operation_condition"


def population_binding_proof_ref(population_binding_id: str) -> str:
    if not population_binding_id:
        raise ValueError("population binding proof requires binding id")
    return f"source_population:{population_binding_id}"


@dataclass(frozen=True)
class RelationSourcePopulationBinding:
    id: str
    supporting_proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        population_binding_proof_ref(self.id)

    @property
    def proof_refs(self) -> tuple[str, ...]:
        return (
            population_binding_proof_ref(self.id),
            *self.supporting_proof_refs,
        )


@dataclass(frozen=True)
class PopulationCoverageClaim:
    test_ref: MembershipTestRef
    role: PopulationCoverageRole
    proof_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.proof_refs:
            raise ValueError("population coverage claim requires proof references")


def merge_population_coverage_claims(
    claims: tuple[PopulationCoverageClaim, ...],
) -> tuple[PopulationCoverageClaim, ...]:
    """Merge independent mechanics proving the same canonical claim."""

    merged: dict[
        tuple[MembershipTestRef, PopulationCoverageRole], PopulationCoverageClaim
    ] = {}
    for claim in claims:
        key = (claim.test_ref, claim.role)
        existing = merged.get(key)
        if existing is None:
            merged[key] = claim
            continue
        merged[key] = PopulationCoverageClaim(
            test_ref=claim.test_ref,
            role=claim.role,
            proof_refs=tuple(
                dict.fromkeys((*existing.proof_refs, *claim.proof_refs))
            ),
        )
    return tuple(merged.values())


@dataclass(frozen=True)
class RelationSourceReviewScopeDecision:
    membership_test_id: str
    decision: ReviewScopeDecisionKind
    axis_kind: str
    axis_id: str
    owner_surface_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.membership_test_id:
            raise ValueError("review scope decision requires membership test")
        if not self.axis_kind:
            raise ValueError("review scope decision requires axis kind")
        if not self.axis_id:
            raise ValueError("review scope decision requires axis id")


@dataclass(frozen=True)
class EndpointParamBinding:
    param_id: str
    value_expr: ValueExpression
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.param_id:
            raise ValueError("endpoint param binding requires param")


@dataclass(frozen=True)
class RelationSourceAppliedFilter:
    predicate_field_ids: tuple[str, ...]
    value_expr: ValueExpression
    operator: str = "equals"
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.predicate_field_ids:
            raise ValueError("relation source applied filter requires predicate fields")
        if not self.operator:
            raise ValueError("relation source applied filter requires operator")


@dataclass(frozen=True)
class RelationSourceRowFilter:
    field_id: str
    operator: str
    value_expr: ValueExpression
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("relation source row filter requires field")
        if not self.operator:
            raise ValueError("relation source row filter requires operator")


@dataclass(frozen=True)
class RelationSourcePopulationChoice:
    controller_kind: PopulationChoiceControllerKind
    controller_id: str
    field_id: str
    requested_fact_ids: tuple[str, ...]
    selection_expr: ParameterRef
    allowed_values: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    review_scope_decisions: tuple[RelationSourceReviewScopeDecision, ...] = ()

    def __post_init__(self) -> None:
        if not self.controller_id:
            raise ValueError("relation source population choice requires controller")
        if not self.field_id:
            raise ValueError("relation source population choice requires field")
        if not self.requested_fact_ids:
            raise ValueError(
                "relation source population choice requires requested facts"
            )
        if len(set(self.requested_fact_ids)) != len(self.requested_fact_ids):
            raise ValueError(
                "relation source population choice requested facts must be unique"
            )


@dataclass(frozen=True)
class RelationSource:
    kind: SourceKind
    read_id: str = ""
    row_source_id: str = ""
    calendar_id: str = ""
    memory_relation_id: str = ""
    param_bindings: tuple[EndpointParamBinding, ...] = ()
    applied_filters: tuple[RelationSourceAppliedFilter, ...] = ()
    row_filters: tuple[RelationSourceRowFilter, ...] = ()
    population_choices: tuple[RelationSourcePopulationChoice, ...] = ()
    population_binding: RelationSourcePopulationBinding | None = None
    population_coverage_claims: tuple[PopulationCoverageClaim, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationField:
    field_id: str
    roles: tuple[FieldBindingRole, ...]


@dataclass(frozen=True)
class Relation:
    id: str
    source: RelationSource
    fields: tuple[RelationField, ...] = ()

    @property
    def grain_keys(self) -> tuple[str, ...]:
        return tuple(
            item.field_id
            for item in self.fields
            if FieldBindingRole.IDENTITY in item.roles
        )

    def field(self, field_id: str) -> RelationField | None:
        for item in self.fields:
            if item.field_id == field_id:
                return item
        return None
