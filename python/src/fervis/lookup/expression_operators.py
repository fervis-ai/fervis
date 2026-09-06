"""One signature registry for semantic and executable expressions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fervis.types.enums import StrEnum
from fervis.lookup.semantic_types import (
    BooleanType,
    CollectionType,
    DateTimeType,
    DateType,
    DecimalType,
    IntegerType,
    IdentifierType,
    NumericType,
    OrderableType,
    RatioMeasure,
    TemporalPointType,
    TemporalScopeType,
    TextType,
    UnitlessMeasure,
    UnspecifiedScalarType,
    ValueType,
    is_numeric,
    normalized_numeric_type,
)


class ExpressionUnaryOperator(StrEnum):
    NEGATE = "negate"
    NOT = "not"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class ExpressionBinaryOperator(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AND = "and"
    OR = "or"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    CONTAINS = "contains"
    WITHIN = "within"


class OperatorKind(StrEnum):
    ARITHMETIC = "arithmetic"
    BOOLEAN = "boolean"
    COMPARISON = "comparison"
    MEMBERSHIP = "membership"
    NULL_CHECK = "null_check"


@dataclass(frozen=True)
class OperatorSignature:
    arity: int
    kind: OperatorKind


_OPERATOR_SIGNATURES: Mapping[
    ExpressionUnaryOperator | ExpressionBinaryOperator, OperatorSignature
] = MappingProxyType(
    {
        ExpressionUnaryOperator.NEGATE: OperatorSignature(1, OperatorKind.ARITHMETIC),
        ExpressionUnaryOperator.NOT: OperatorSignature(1, OperatorKind.BOOLEAN),
        ExpressionUnaryOperator.IS_NULL: OperatorSignature(1, OperatorKind.NULL_CHECK),
        ExpressionUnaryOperator.NOT_NULL: OperatorSignature(1, OperatorKind.NULL_CHECK),
        ExpressionBinaryOperator.ADD: OperatorSignature(2, OperatorKind.ARITHMETIC),
        ExpressionBinaryOperator.SUBTRACT: OperatorSignature(
            2, OperatorKind.ARITHMETIC
        ),
        ExpressionBinaryOperator.MULTIPLY: OperatorSignature(
            2, OperatorKind.ARITHMETIC
        ),
        ExpressionBinaryOperator.DIVIDE: OperatorSignature(2, OperatorKind.ARITHMETIC),
        ExpressionBinaryOperator.AND: OperatorSignature(2, OperatorKind.BOOLEAN),
        ExpressionBinaryOperator.OR: OperatorSignature(2, OperatorKind.BOOLEAN),
        ExpressionBinaryOperator.EQUALS: OperatorSignature(2, OperatorKind.COMPARISON),
        ExpressionBinaryOperator.NOT_EQUALS: OperatorSignature(
            2, OperatorKind.COMPARISON
        ),
        ExpressionBinaryOperator.LT: OperatorSignature(2, OperatorKind.COMPARISON),
        ExpressionBinaryOperator.LTE: OperatorSignature(2, OperatorKind.COMPARISON),
        ExpressionBinaryOperator.GT: OperatorSignature(2, OperatorKind.COMPARISON),
        ExpressionBinaryOperator.GTE: OperatorSignature(2, OperatorKind.COMPARISON),
        ExpressionBinaryOperator.IN: OperatorSignature(2, OperatorKind.MEMBERSHIP),
        ExpressionBinaryOperator.CONTAINS: OperatorSignature(
            2, OperatorKind.MEMBERSHIP
        ),
        ExpressionBinaryOperator.WITHIN: OperatorSignature(2, OperatorKind.MEMBERSHIP),
    }
)


def operator_signature(
    operator: ExpressionUnaryOperator | ExpressionBinaryOperator,
) -> OperatorSignature:
    return _OPERATOR_SIGNATURES[operator]


def infer_operator_result(
    operator: ExpressionUnaryOperator | ExpressionBinaryOperator,
    operands: tuple[ValueType, ...],
) -> ValueType:
    signature = operator_signature(operator)
    if len(operands) != signature.arity:
        raise ValueError(f"{operator.value} requires {signature.arity} operands")
    operands = tuple(normalized_numeric_type(item) for item in operands)
    if signature.kind in {
        OperatorKind.BOOLEAN,
        OperatorKind.COMPARISON,
        OperatorKind.MEMBERSHIP,
        OperatorKind.NULL_CHECK,
    }:
        _validate_nonarithmetic_operands(operator, operands)
        return BooleanType()
    return _infer_arithmetic_result(operator, operands)


def infer_aggregate_result(function: str, operand: ValueType) -> ValueType:
    """Return an aggregate result after validating its operand capability."""

    if isinstance(operand, CollectionType):
        raise ValueError("aggregate argument must be row-level scalar")
    if function == "count":
        return IntegerType()
    if function in {"sum", "average"}:
        if not is_numeric(operand):
            raise ValueError(f"{function} requires numeric argument")
        return operand
    if function in {"minimum", "maximum"}:
        if not _is_orderable(operand):
            raise ValueError(f"{function} requires orderable argument")
        return operand
    raise ValueError(f"unsupported aggregate function {function}")


def _validate_nonarithmetic_operands(
    operator: ExpressionUnaryOperator | ExpressionBinaryOperator,
    operands: tuple[ValueType, ...],
) -> None:
    if operator is ExpressionUnaryOperator.NOT or operator in {
        ExpressionBinaryOperator.AND,
        ExpressionBinaryOperator.OR,
    }:
        if any(not isinstance(item, BooleanType) for item in operands):
            raise ValueError(f"{operator.value} requires Boolean operands")
        return
    if operator in {
        ExpressionUnaryOperator.IS_NULL,
        ExpressionUnaryOperator.NOT_NULL,
    }:
        return
    left, right = operands
    if operator in {
        ExpressionBinaryOperator.EQUALS,
        ExpressionBinaryOperator.NOT_EQUALS,
    }:
        if not _equality_compatible(left, right):
            raise ValueError("equality requires compatible scalar operands")
        return
    if operator in {
        ExpressionBinaryOperator.LT,
        ExpressionBinaryOperator.LTE,
        ExpressionBinaryOperator.GT,
        ExpressionBinaryOperator.GTE,
    }:
        if not (
            _numeric_types_compatible(left, right)
            or left == right
            and _is_orderable(left)
        ):
            raise ValueError("ordering comparison requires compatible operands")
        return
    if operator is ExpressionBinaryOperator.IN:
        if not isinstance(right, CollectionType) or not _equality_compatible(
            left, right.element_type
        ):
            raise ValueError("in requires scalar and compatible collection")
        return
    if operator is ExpressionBinaryOperator.CONTAINS:
        if not (
            isinstance(left, TextType)
            and isinstance(right, TextType)
            or isinstance(left, CollectionType)
            and left.element_type == right
        ):
            raise ValueError("contains requires text or compatible collection")
        return
    if operator is ExpressionBinaryOperator.WITHIN and not (
        isinstance(left, (DateType, DateTimeType, TemporalPointType))
        and isinstance(right, TemporalScopeType)
    ):
        raise ValueError("within requires temporal value and temporal scope")


def _equality_compatible(left: ValueType, right: ValueType) -> bool:
    return (
        isinstance(left, UnspecifiedScalarType)
        or isinstance(right, UnspecifiedScalarType)
        or _numeric_types_compatible(left, right)
        or left == right
        or (
        isinstance(left, IdentifierType)
        and isinstance(right, TextType)
        or isinstance(right, IdentifierType)
        and isinstance(left, TextType)
        )
    )


def _numeric_types_compatible(left: ValueType, right: ValueType) -> bool:
    if not (is_numeric(left) and is_numeric(right)):
        return False
    if isinstance(left, (IntegerType, NumericType)) or isinstance(right, (IntegerType, NumericType)):
        return True
    return left == right


def _is_orderable(value_type: ValueType) -> bool:
    return isinstance(
        value_type,
        (
            DateType,
            DateTimeType,
            DecimalType,
            IntegerType,
            NumericType,
            OrderableType,
            TemporalPointType,
            TextType,
            UnspecifiedScalarType,
        ),
    )


def _infer_arithmetic_result(
    operator: ExpressionUnaryOperator | ExpressionBinaryOperator,
    operands: tuple[ValueType, ...],
) -> ValueType:
    if not all(is_numeric(item) for item in operands):
        raise ValueError(f"{operator.value} requires numeric operands")
    if any(isinstance(item, NumericType) for item in operands):
        return NumericType()
    if operator is ExpressionUnaryOperator.NEGATE:
        return operands[0]
    left, right = operands
    if operator in {ExpressionBinaryOperator.ADD, ExpressionBinaryOperator.SUBTRACT}:
        if left != right:
            raise ValueError(f"{operator.value} requires identical numeric types")
        return left
    if operator is ExpressionBinaryOperator.MULTIPLY:
        if _is_unitless(left) and _is_unitless(right):
            return DecimalType(UnitlessMeasure())
        if _is_ratio(left) and isinstance(right, DecimalType):
            return right
        if _is_ratio(right) and isinstance(left, DecimalType):
            return left
        raise ValueError("multiply requires unitless values or ratio and measure")
    if operator is ExpressionBinaryOperator.DIVIDE:
        if left == right:
            return DecimalType(RatioMeasure())
        if _is_unitless(right):
            return left
        raise ValueError("divide requires unitless denominator or like measures")
    raise ValueError(f"unsupported arithmetic operator {operator.value}")


def _is_ratio(value_type: ValueType) -> bool:
    return isinstance(value_type, DecimalType) and isinstance(
        value_type.measure, RatioMeasure
    )


def _is_unitless(value_type: ValueType) -> bool:
    return isinstance(value_type, IntegerType) or (
        isinstance(value_type, DecimalType)
        and isinstance(value_type.measure, UnitlessMeasure)
    )


__all__ = [
    "ExpressionBinaryOperator",
    "ExpressionUnaryOperator",
    "OperatorKind",
    "OperatorSignature",
    "infer_aggregate_result",
    "infer_operator_result",
    "operator_signature",
]
