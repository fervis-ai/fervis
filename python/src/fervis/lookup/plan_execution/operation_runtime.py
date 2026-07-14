"""Runtime data for deterministic relation operation execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping, TypeAlias, TypeVar
from typing_extensions import assert_never

from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AntiJoinSpec,
    ComputeBinaryOperator,
    CrossJoinSpec,
    FilterSpec,
    JoinSpec,
    OperationKind,
    ProjectSpec,
    ProjectToKeySpec,
    RoleExpandSpec,
    SortKey,
    TiePolicy,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.outcomes.errors import ExecutionIssue
from fervis.lookup.outcomes.model import Undefined


@dataclass(frozen=True)
class ScalarInput:
    id: str
    value: RuntimeValue
    value_type: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedOperationInput:
    operation_id: str
    input_id: str
    value: RuntimeValue
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedRankSpec:
    input_relation: str
    order_by: tuple[SortKey, ...]
    tie_policy: TiePolicy
    limit: int
    tie_breakers: tuple[SortKey, ...] = ()
    kind: OperationKind = OperationKind.RANK


@dataclass(frozen=True)
class ResolvedComputeValue:
    input_ref: str
    value: object
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedComputeOutput:
    node_id: str
    output_id: str


@dataclass(frozen=True)
class ResolvedComputeNegation:
    operand: ResolvedComputeExpression


@dataclass(frozen=True)
class ResolvedComputeBinary:
    operator: ComputeBinaryOperator
    left: ResolvedComputeExpression
    right: ResolvedComputeExpression


ResolvedComputeExpression: TypeAlias = (
    ResolvedComputeValue
    | ResolvedComputeOutput
    | ResolvedComputeNegation
    | ResolvedComputeBinary
)

_FoldResult = TypeVar("_FoldResult")


def fold_resolved_compute_expression(
    expression: ResolvedComputeExpression,
    *,
    value: Callable[[ResolvedComputeValue], _FoldResult],
    output: Callable[[ResolvedComputeOutput], _FoldResult],
    negation: Callable[[ResolvedComputeNegation, _FoldResult], _FoldResult],
    binary: Callable[
        [ResolvedComputeBinary, _FoldResult, _FoldResult],
        _FoldResult,
    ],
) -> _FoldResult:
    """Exhaustively interpret one resolved compute-expression tree."""

    def visit(current: ResolvedComputeExpression) -> _FoldResult:
        if isinstance(current, ResolvedComputeValue):
            return value(current)
        if isinstance(current, ResolvedComputeOutput):
            return output(current)
        if isinstance(current, ResolvedComputeNegation):
            return negation(current, visit(current.operand))
        if isinstance(current, ResolvedComputeBinary):
            return binary(current, visit(current.left), visit(current.right))
        assert_never(current)

    return visit(expression)


@dataclass(frozen=True)
class ResolvedComputeReferences:
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()


def resolved_compute_references(
    expression: ResolvedComputeExpression,
) -> ResolvedComputeReferences:
    """Return all deterministic input and prior-output references."""

    return fold_resolved_compute_expression(
        expression,
        value=lambda item: ResolvedComputeReferences(input_refs=(item.input_ref,)),
        output=lambda item: ResolvedComputeReferences(
            input_refs=(item.output_id,),
            output_refs=(item.output_id,),
        ),
        negation=lambda _expression, operand: operand,
        binary=lambda _expression, left, right: ResolvedComputeReferences(
            input_refs=_dedupe_refs((*left.input_refs, *right.input_refs)),
            output_refs=_dedupe_refs((*left.output_refs, *right.output_refs)),
        ),
    )


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(refs))


@dataclass(frozen=True)
class ResolvedComputeSpec:
    expression: ResolvedComputeExpression
    output_scalar: str
    kind: OperationKind = OperationKind.COMPUTE


ExecutableOperationSpec: TypeAlias = (
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
    | ResolvedRankSpec
    | ResolvedComputeSpec
)


@dataclass(frozen=True)
class ExecutableOperation:
    id: str
    spec: ExecutableOperationSpec
    output_relation: str = ""


@dataclass(frozen=True)
class RelationEngineInput:
    relations: tuple[RelationRows, ...] = ()
    operations: tuple[ExecutableOperation, ...] = ()
    scalar_inputs: tuple[ScalarInput, ...] = ()
    operation_proof_refs: Mapping[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class RelationEngineOutput:
    relations: tuple[RelationRows, ...] = ()
    scalars: Mapping[str, RuntimeValue] | None = None
    scalar_proofs: Mapping[str, tuple[str, ...]] | None = None
    scalar_types: Mapping[str, str] | None = None
    undefined: Undefined | None = None
    issue: ExecutionIssue | None = None

    def relation(self, relation_id: str) -> RelationRows:
        for relation in self.relations:
            if relation.id == relation_id:
                return relation
        raise RelationEngineError(f"unknown relation {relation_id}")
