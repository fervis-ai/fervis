"""Generic relational operation model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, TypeVar, assert_never

from fervis.lookup.answer_program.values import (
    ConstantRef,
    NodeOutputRef,
    ParameterRef,
    ValueExpression,
)


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


class ComputeBinaryOperator(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


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
    kind: OperationKind = field(default=OperationKind.FILTER, init=False)


@dataclass(frozen=True)
class ProjectSpec:
    input_relation: str
    fields: tuple[ProjectField, ...]
    kind: OperationKind = field(default=OperationKind.PROJECT, init=False)


@dataclass(frozen=True)
class ProjectToIdentitySpec:
    input_relation: str
    identity_fields: tuple[str, ...]
    fields: tuple[ProjectField, ...] = ()
    kind: OperationKind = field(default=OperationKind.PROJECT_TO_IDENTITY, init=False)


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
    carry_fields: tuple[ProjectField, ...] = ()
    kind: OperationKind = field(default=OperationKind.AGGREGATE, init=False)


@dataclass(frozen=True)
class RankSpec:
    input_relation: str
    order_by: tuple[SortKey, ...]
    tie_policy: TiePolicy
    limit: ValueExpression
    tie_breakers: tuple[SortKey, ...] = ()
    kind: OperationKind = field(default=OperationKind.RANK, init=False)


@dataclass(frozen=True)
class ComputeNegation:
    operand: ComputeExpression


@dataclass(frozen=True)
class ComputeBinary:
    operator: ComputeBinaryOperator
    left: ComputeExpression
    right: ComputeExpression


ComputeExpression: TypeAlias = (
    ParameterRef | NodeOutputRef | ConstantRef | ComputeNegation | ComputeBinary
)
ComputeExpressionLeaf: TypeAlias = ParameterRef | NodeOutputRef | ConstantRef

_FoldResult = TypeVar("_FoldResult")


def fold_compute_expression(
    expression: ComputeExpression,
    *,
    parameter: Callable[[ParameterRef], _FoldResult],
    output: Callable[[NodeOutputRef], _FoldResult],
    constant: Callable[[ConstantRef], _FoldResult],
    negation: Callable[[ComputeNegation, _FoldResult], _FoldResult],
    binary: Callable[
        [ComputeBinary, _FoldResult, _FoldResult],
        _FoldResult,
    ],
) -> _FoldResult:
    """Exhaustively interpret one closed compute-expression tree."""

    def visit(current: ComputeExpression) -> _FoldResult:
        if isinstance(current, ParameterRef):
            return parameter(current)
        if isinstance(current, NodeOutputRef):
            return output(current)
        if isinstance(current, ConstantRef):
            return constant(current)
        if isinstance(current, ComputeNegation):
            return negation(current, visit(current.operand))
        if isinstance(current, ComputeBinary):
            return binary(current, visit(current.left), visit(current.right))
        assert_never(current)

    return visit(expression)


@dataclass(frozen=True)
class ComputeExpressionReferences:
    leaves: tuple[ComputeExpressionLeaf, ...] = ()
    parameters: tuple[ParameterRef, ...] = ()
    outputs: tuple[NodeOutputRef, ...] = ()
    constants: tuple[ConstantRef, ...] = ()


def compute_expression_references(
    expression: ComputeExpression,
) -> ComputeExpressionReferences:
    """Return the complete typed reference projection of an expression tree."""

    return fold_compute_expression(
        expression,
        parameter=lambda item: ComputeExpressionReferences(
            leaves=(item,),
            parameters=(item,),
        ),
        output=lambda item: ComputeExpressionReferences(
            leaves=(item,),
            outputs=(item,),
        ),
        constant=lambda item: ComputeExpressionReferences(
            leaves=(item,),
            constants=(item,),
        ),
        negation=lambda _expression, operand: operand,
        binary=lambda _expression, left, right: ComputeExpressionReferences(
            leaves=(*left.leaves, *right.leaves),
            parameters=(*left.parameters, *right.parameters),
            outputs=(*left.outputs, *right.outputs),
            constants=(*left.constants, *right.constants),
        ),
    )


@dataclass(frozen=True)
class ComputeSpec:
    expression: ComputeExpression
    output_scalar: str = ""
    kind: OperationKind = field(default=OperationKind.COMPUTE, init=False)


def compute_expression_leaves(
    expression: ComputeExpression,
) -> tuple[ValueExpression, ...]:
    return compute_expression_references(expression).leaves


OperationSpec: TypeAlias = (
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
            ProjectToIdentitySpec,
            RoleExpandSpec,
            AggregateSpec,
            RankSpec,
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
