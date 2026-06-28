"""Generic relational operation model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OperationKind(StrEnum):
    FILTER = "filter"
    PROJECT = "project"
    PROJECT_TO_IDENTITY = "project_to_identity"
    JOIN = "join"
    UNION = "union"
    ROLE_EXPAND = "role_expand"
    CROSS_JOIN = "cross_join"
    ANTI_JOIN = "anti_join"
    UNIVERSAL_CONDITION = "universal_condition"
    AGGREGATE = "aggregate"
    RANK = "rank"
    COMPUTE = "compute"


class PredicateOperator(StrEnum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class AggregationFunction(StrEnum):
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    AVG = "avg"


class TiePolicy(StrEnum):
    FIELD = "field"


class RelationRole(StrEnum):
    ANTI_JOIN_CANDIDATE = "anti_join.candidate"
    ANTI_JOIN_OBSERVED = "anti_join.observed"
    UNIVERSAL_CANDIDATE_SUBJECT = "universal_condition.candidate_subject"
    UNIVERSAL_REQUIRED_DIMENSION = "universal_condition.required_dimension"
    UNIVERSAL_OBSERVATION = "universal_condition.observation"


@dataclass(frozen=True)
class JoinKey:
    left: str
    right: str


@dataclass(frozen=True)
class RelationRoleRef:
    relation_id: str
    role: RelationRole
    required_identity_fields: tuple[str, ...]


@dataclass(frozen=True)
class Predicate:
    left: str
    operator: PredicateOperator
    right: str = ""
    right_scalar: str = ""


@dataclass(frozen=True)
class SortKey:
    field: str
    direction: SortDirection


@dataclass(frozen=True)
class ProjectField:
    source: str
    output: str = ""


@dataclass(frozen=True)
class RoleMapping:
    role: str
    source_field: str
    output_field: str


@dataclass(frozen=True)
class AggregationSpec:
    function: AggregationFunction
    output_field: str
    input_field: str = ""


@dataclass(frozen=True)
class FilterSpec:
    input_relation: str
    predicate: Predicate
    kind: OperationKind = OperationKind.FILTER


@dataclass(frozen=True)
class ProjectSpec:
    input_relation: str
    fields: tuple[ProjectField, ...]
    kind: OperationKind = OperationKind.PROJECT


@dataclass(frozen=True)
class ProjectToIdentitySpec:
    input_relation: str
    identity_fields: tuple[str, ...]
    fields: tuple[ProjectField, ...] = ()
    kind: OperationKind = OperationKind.PROJECT_TO_IDENTITY


@dataclass(frozen=True)
class JoinSpec:
    left: str
    right: str
    join_keys: tuple[JoinKey, ...]
    kind: OperationKind = OperationKind.JOIN


@dataclass(frozen=True)
class UnionSpec:
    inputs: tuple[str, ...]
    output_fields: tuple[str, ...]
    identity_fields: tuple[str, ...] = ()
    kind: OperationKind = OperationKind.UNION


@dataclass(frozen=True)
class RoleExpandSpec:
    input_relation: str
    mappings: tuple[RoleMapping, ...]
    output_fields: tuple[str, ...]
    carry_fields: tuple[str, ...] = ()
    role_field: str = "role"
    kind: OperationKind = OperationKind.ROLE_EXPAND


@dataclass(frozen=True)
class CrossJoinSpec:
    left: str
    right: str
    kind: OperationKind = OperationKind.CROSS_JOIN


@dataclass(frozen=True)
class AntiJoinSpec:
    candidate: RelationRoleRef
    observed: RelationRoleRef
    join_keys: tuple[JoinKey, ...]
    output_fields: tuple[ProjectField, ...]
    kind: OperationKind = OperationKind.ANTI_JOIN


@dataclass(frozen=True)
class UniversalConditionSpec:
    candidate_subject: RelationRoleRef
    required_dimension: RelationRoleRef
    observation: RelationRoleRef
    subject_keys: tuple[JoinKey, ...]
    dimension_keys: tuple[JoinKey, ...]
    predicate: Predicate
    output_fields: tuple[ProjectField, ...]
    kind: OperationKind = OperationKind.UNIVERSAL_CONDITION


@dataclass(frozen=True)
class AggregateSpec:
    input_relation: str
    group_by: tuple[str, ...]
    aggregations: tuple[AggregationSpec, ...]
    carry_fields: tuple[ProjectField, ...] = ()
    kind: OperationKind = OperationKind.AGGREGATE


@dataclass(frozen=True)
class RankSpec:
    input_relation: str
    order_by: tuple[SortKey, ...]
    tie_policy: TiePolicy
    limit: int
    tie_breakers: tuple[SortKey, ...] = ()
    kind: OperationKind = OperationKind.RANK


@dataclass(frozen=True)
class ComputeSpec:
    expression: str
    scalar_inputs: tuple[str, ...]
    output_scalar: str = ""
    kind: OperationKind = OperationKind.COMPUTE


OperationSpec = (
    FilterSpec
    | ProjectSpec
    | ProjectToIdentitySpec
    | JoinSpec
    | UnionSpec
    | RoleExpandSpec
    | CrossJoinSpec
    | AntiJoinSpec
    | UniversalConditionSpec
    | AggregateSpec
    | RankSpec
    | ComputeSpec
)


@dataclass(frozen=True)
class Operation:
    id: str
    spec: OperationSpec
    output_relation: str = ""

    @property
    def kind(self) -> OperationKind:
        return self.spec.kind
