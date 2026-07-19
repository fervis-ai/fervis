"""Generic relational operation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from typing import TypeAlias
from typing_extensions import assert_never

from fervis.lookup.answer_program.expressions import Expression, expression_input_id
from fervis.lookup.answer_program.values import ConstantRef, ParameterRef
from fervis.lookup.answer_program.relations import PopulationCoverageClaim


class OperationKind(StrEnum):
    FILTER = "filter"
    PROJECT = "project"
    PROJECT_TO_KEY = "project_to_key"
    JOIN = "join"
    UNION = "union"
    ROLE_EXPAND = "role_expand"
    CROSS_JOIN = "cross_join"
    ANTI_JOIN = "anti_join"
    UNIVERSAL_CONDITION = "universal_condition"
    AGGREGATE = "aggregate"
    ORDER = "order"
    COMPUTE = "compute"


class PredicateOperator(StrEnum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
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
    left: Expression
    operator: PredicateOperator
    right: Expression | None = None


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
    proof_refs: tuple[str, ...] = ()
    population_coverage_claims: tuple[PopulationCoverageClaim, ...] = ()
    kind: OperationKind = field(default=OperationKind.FILTER, init=False)


@dataclass(frozen=True)
class ProjectSpec:
    input_relation: str
    fields: tuple[ProjectField, ...]
    kind: OperationKind = field(default=OperationKind.PROJECT, init=False)


@dataclass(frozen=True)
class ProjectToKeySpec:
    input_relation: str
    key_fields: tuple[str, ...]
    kind: OperationKind = field(default=OperationKind.PROJECT_TO_KEY, init=False)


@dataclass(frozen=True)
class JoinSpec:
    left: str
    right: str
    join_keys: tuple[JoinKey, ...]
    kind: OperationKind = field(default=OperationKind.JOIN, init=False)


@dataclass(frozen=True)
class UnionSpec:
    inputs: tuple[str, ...]
    output_fields: tuple[str, ...]
    identity_fields: tuple[str, ...] = ()
    kind: OperationKind = field(default=OperationKind.UNION, init=False)


@dataclass(frozen=True)
class RoleExpandSpec:
    input_relation: str
    mappings: tuple[RoleMapping, ...]
    output_fields: tuple[str, ...]
    carry_fields: tuple[str, ...] = ()
    role_field: str = "role"
    kind: OperationKind = field(default=OperationKind.ROLE_EXPAND, init=False)


@dataclass(frozen=True)
class CrossJoinSpec:
    left: str
    right: str
    kind: OperationKind = field(default=OperationKind.CROSS_JOIN, init=False)


@dataclass(frozen=True)
class AntiJoinSpec:
    candidate: RelationRoleRef
    observed: RelationRoleRef
    join_keys: tuple[JoinKey, ...]
    output_fields: tuple[ProjectField, ...]
    kind: OperationKind = field(default=OperationKind.ANTI_JOIN, init=False)


@dataclass(frozen=True)
class UniversalConditionSpec:
    candidate_subject: RelationRoleRef
    required_dimension: RelationRoleRef
    observation: RelationRoleRef
    subject_keys: tuple[JoinKey, ...]
    dimension_keys: tuple[JoinKey, ...]
    predicate: Predicate
    output_fields: tuple[ProjectField, ...]
    kind: OperationKind = field(
        default=OperationKind.UNIVERSAL_CONDITION,
        init=False,
    )


@dataclass(frozen=True)
class AggregateSpec:
    input_relation: str
    group_by: tuple[str, ...]
    aggregations: tuple[AggregationSpec, ...]
    kind: OperationKind = field(default=OperationKind.AGGREGATE, init=False)


@dataclass(frozen=True)
class KeepAll:
    pass


@dataclass(frozen=True)
class Take:
    limit: Expression


OrderSelection: TypeAlias = KeepAll | Take


@dataclass(frozen=True)
class OrderSpec:
    input_relation: str
    order_by: tuple[SortKey, ...]
    selection: OrderSelection
    tie_breakers: tuple[SortKey, ...] = ()
    kind: OperationKind = field(default=OperationKind.ORDER, init=False)


@dataclass(frozen=True)
class ComputeInputPopulationCoverage:
    input_id: str
    claims: tuple[PopulationCoverageClaim, ...]

    def __post_init__(self) -> None:
        if not self.input_id:
            raise ValueError("compute input population coverage requires input")


@dataclass(frozen=True)
class ComputeSpec:
    expression: Expression
    output_scalar: str = ""
    input_population_coverage: tuple[ComputeInputPopulationCoverage, ...] = ()
    kind: OperationKind = field(default=OperationKind.COMPUTE, init=False)

    def __post_init__(self) -> None:
        input_ids = tuple(item.input_id for item in self.input_population_coverage)
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("compute input population coverage must be unique")


def compute_value_input_id(expression: Expression) -> str:
    if isinstance(expression, (ParameterRef, ConstantRef)):
        return expression_input_id(expression)
    raise ValueError("compute input coverage requires a parameter or constant")


OperationSpec: TypeAlias = (
    FilterSpec
    | ProjectSpec
    | ProjectToKeySpec
    | JoinSpec
    | UnionSpec
    | RoleExpandSpec
    | CrossJoinSpec
    | AntiJoinSpec
    | UniversalConditionSpec
    | AggregateSpec
    | OrderSpec
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

    @property
    def input_relation_ids(self) -> tuple[str, ...]:
        return operation_input_relation_ids(self.spec)

    @property
    def output_scalar(self) -> str:
        return self.spec.output_scalar if isinstance(self.spec, ComputeSpec) else ""


def operation_input_relation_ids(spec: OperationSpec) -> tuple[str, ...]:
    """Project relation dependencies from the closed operation union."""

    if isinstance(
        spec,
        (
            FilterSpec,
            ProjectSpec,
            ProjectToKeySpec,
            RoleExpandSpec,
            AggregateSpec,
            OrderSpec,
        ),
    ):
        return (spec.input_relation,)
    if isinstance(spec, (JoinSpec, CrossJoinSpec)):
        return (spec.left, spec.right)
    if isinstance(spec, UnionSpec):
        return spec.inputs
    if isinstance(spec, AntiJoinSpec):
        return (spec.candidate.relation_id, spec.observed.relation_id)
    if isinstance(spec, UniversalConditionSpec):
        return (
            spec.candidate_subject.relation_id,
            spec.required_dimension.relation_id,
            spec.observation.relation_id,
        )
    if isinstance(spec, ComputeSpec):
        return ()
    assert_never(spec)
