"""Predicate evaluation for relation operations."""

from __future__ import annotations

import operator
from collections.abc import Mapping

from fervis.lookup.plan_execution.operation_runtime import RelationEngineError
from fervis.lookup.plan_execution.relations import Row
from fervis.lookup.answer_program.operations import Predicate, PredicateOperator
from fervis.lookup.canonical_data import RuntimeValue

from .shared import _field
from fervis.lookup.plan_execution.declared_values import (
    declared_equal,
    declared_order_pair,
)


def _predicate(
    row: Row,
    predicate: Predicate,
    scalars: dict[str, RuntimeValue],
    field_types: Mapping[str, str],
    scalar_types: Mapping[str, str],
) -> bool:
    left = _field(row, predicate.left)
    if predicate.operator == PredicateOperator.IS_NULL:
        return left is None
    if predicate.operator == PredicateOperator.NOT_NULL:
        return left is not None
    right = _predicate_right(row, predicate, scalars)
    left_type = field_types.get(predicate.left)
    right_type = (
        field_types.get(predicate.right)
        if predicate.right
        else scalar_types.get(predicate.right_scalar)
    )
    if predicate.operator == PredicateOperator.EQUALS:
        return declared_equal(left, left_type, right, right_type)
    if predicate.operator == PredicateOperator.NOT_EQUALS:
        return not declared_equal(left, left_type, right, right_type)
    if predicate.operator in {
        PredicateOperator.LT,
        PredicateOperator.LTE,
        PredicateOperator.GT,
        PredicateOperator.GTE,
    }:
        left, right = declared_order_pair(left, left_type, right, right_type)
        operators = {
            PredicateOperator.LT: operator.lt,
            PredicateOperator.LTE: operator.le,
            PredicateOperator.GT: operator.gt,
            PredicateOperator.GTE: operator.ge,
        }
        return bool(operators[predicate.operator](left, right))
    if predicate.operator == PredicateOperator.CONTAINS:
        return _contains(left, right, right_type)
    raise RelationEngineError(f"unsupported predicate {predicate.operator}")


def _contains(left: RuntimeValue, right: RuntimeValue, right_type: str | None) -> bool:
    if isinstance(left, str):
        return isinstance(right, str) and right in left
    if isinstance(left, (tuple, list)):
        return any(declared_equal(item, None, right, right_type) for item in left)
    if isinstance(left, dict):
        return isinstance(right, str) and right in left
    raise RelationEngineError("contains requires a string, collection, or mapping")


def _predicate_right(
    row: Row,
    predicate: Predicate,
    scalars: dict[str, RuntimeValue],
) -> RuntimeValue:
    if predicate.right:
        return _field(row, predicate.right)
    if predicate.right_scalar not in scalars:
        raise RelationEngineError(f"unknown scalar input {predicate.right_scalar}")
    return scalars[predicate.right_scalar]


def _predicate_fact(
    row: Row,
    predicate: Predicate,
    scalars: dict[str, RuntimeValue],
    field_types: Mapping[str, str],
    scalar_types: Mapping[str, str],
) -> tuple[RuntimeValue, ...]:
    left = _field(row, predicate.left)
    if predicate.operator in {PredicateOperator.IS_NULL, PredicateOperator.NOT_NULL}:
        return (_predicate(row, predicate, scalars, field_types, scalar_types), left)
    right = _predicate_right(row, predicate, scalars)
    return (_predicate(row, predicate, scalars, field_types, scalar_types), left, right)
