"""One closed expression language for answer-program values and row fields."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, TypeVar
from typing_extensions import assert_never

from fervis.types.enums import StrEnum

from .values import ConstantRef, EnvironmentRef, NodeOutputRef, ParameterRef


class ExpressionUnaryOperator(StrEnum):
    NEGATE = "negate"


class ExpressionBinaryOperator(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class ExpressionFunction(StrEnum):
    TEMPORAL_BUCKET = "temporal_bucket"


@dataclass(frozen=True)
class FieldRef:
    field_id: str

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field reference requires field id")


@dataclass(frozen=True)
class UnaryExpression:
    operator: ExpressionUnaryOperator
    operand: Expression


@dataclass(frozen=True)
class BinaryExpression:
    operator: ExpressionBinaryOperator
    left: Expression
    right: Expression


@dataclass(frozen=True)
class FunctionExpression:
    function: ExpressionFunction
    arguments: tuple[Expression, ...]

    def __post_init__(self) -> None:
        if not self.arguments:
            raise ValueError("function expression requires arguments")


Expression: TypeAlias = (
    FieldRef
    | ParameterRef
    | NodeOutputRef
    | ConstantRef
    | EnvironmentRef
    | UnaryExpression
    | BinaryExpression
    | FunctionExpression
)
ExpressionLeaf: TypeAlias = (
    FieldRef | ParameterRef | NodeOutputRef | ConstantRef | EnvironmentRef
)

_FoldResult = TypeVar("_FoldResult")


def fold_expression(
    expression: Expression,
    *,
    field: Callable[[FieldRef], _FoldResult],
    parameter: Callable[[ParameterRef], _FoldResult],
    output: Callable[[NodeOutputRef], _FoldResult],
    constant: Callable[[ConstantRef], _FoldResult],
    environment: Callable[[EnvironmentRef], _FoldResult],
    unary: Callable[[UnaryExpression, _FoldResult], _FoldResult],
    binary: Callable[[BinaryExpression, _FoldResult, _FoldResult], _FoldResult],
    function: Callable[[FunctionExpression, tuple[_FoldResult, ...]], _FoldResult],
) -> _FoldResult:
    """Exhaustively interpret one canonical expression tree."""

    def visit(current: Expression) -> _FoldResult:
        if isinstance(current, FieldRef):
            return field(current)
        if isinstance(current, ParameterRef):
            return parameter(current)
        if isinstance(current, NodeOutputRef):
            return output(current)
        if isinstance(current, ConstantRef):
            return constant(current)
        if isinstance(current, EnvironmentRef):
            return environment(current)
        if isinstance(current, UnaryExpression):
            return unary(current, visit(current.operand))
        if isinstance(current, BinaryExpression):
            return binary(current, visit(current.left), visit(current.right))
        if isinstance(current, FunctionExpression):
            return function(current, tuple(visit(item) for item in current.arguments))
        assert_never(current)

    return visit(expression)


@dataclass(frozen=True)
class ExpressionReferences:
    leaves: tuple[ExpressionLeaf, ...] = ()
    fields: tuple[FieldRef, ...] = ()
    parameters: tuple[ParameterRef, ...] = ()
    outputs: tuple[NodeOutputRef, ...] = ()
    constants: tuple[ConstantRef, ...] = ()
    environments: tuple[EnvironmentRef, ...] = ()


def expression_references(
    expression: Expression,
    *additional: Expression,
) -> ExpressionReferences:
    """Project every typed reference from one expression tree."""

    return _merge_references(
        tuple(
            fold_expression(
                current,
                field=lambda item: ExpressionReferences(leaves=(item,), fields=(item,)),
                parameter=lambda item: ExpressionReferences(
                    leaves=(item,), parameters=(item,)
                ),
                output=lambda item: ExpressionReferences(
                    leaves=(item,), outputs=(item,)
                ),
                constant=lambda item: ExpressionReferences(
                    leaves=(item,), constants=(item,)
                ),
                environment=lambda item: ExpressionReferences(
                    leaves=(item,), environments=(item,)
                ),
                unary=lambda _expression, operand: operand,
                binary=lambda _expression, left, right: _merge_references(
                    (left, right)
                ),
                function=lambda _expression, arguments: _merge_references(arguments),
            )
            for current in (expression, *additional)
        )
    )


def expression_constant(expression: Expression) -> ConstantRef | None:
    references = expression_references(expression)
    if len(references.leaves) == 1 and len(references.constants) == 1:
        return references.constants[0]
    return None


def expression_input_id(expression: ParameterRef | ConstantRef) -> str:
    """Return the stable materialized-input ID for one scalar expression leaf."""

    if isinstance(expression, ParameterRef):
        return f"parameter:{expression.parameter_id}"
    return f"constant:{expression.constant_id}@{expression.version_ref}"


def _merge_references(
    references: tuple[ExpressionReferences, ...],
) -> ExpressionReferences:
    return ExpressionReferences(
        leaves=tuple(item for refs in references for item in refs.leaves),
        fields=tuple(item for refs in references for item in refs.fields),
        parameters=tuple(item for refs in references for item in refs.parameters),
        outputs=tuple(item for refs in references for item in refs.outputs),
        constants=tuple(item for refs in references for item in refs.constants),
        environments=tuple(item for refs in references for item in refs.environments),
    )
