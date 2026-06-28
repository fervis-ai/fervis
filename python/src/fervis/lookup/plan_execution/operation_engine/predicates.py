"""Predicate evaluation for relation operations."""

from __future__ import annotations

import operator

from fervis.lookup.plan_execution.operation_runtime import RelationEngineError
from fervis.lookup.plan_execution.relations import Row
from fervis.lookup.fact_plan.operations import Predicate, PredicateOperator

from .shared import _field, _ordered_predicate_values


def _predicate(
    row: Row,
    predicate: Predicate,
    scalars: dict[str, object],
) -> bool:
    left = _field(row, predicate.left)
    if predicate.operator == PredicateOperator.IS_NULL:
        return left is None
    if predicate.operator == PredicateOperator.NOT_NULL:
        return left is not None
    right = _predicate_right(row, predicate, scalars)
    if predicate.operator == PredicateOperator.EQUALS:
        return bool(operator.eq(left, right))
    if predicate.operator == PredicateOperator.NOT_EQUALS:
        return bool(operator.ne(left, right))
    if predicate.operator in {
        PredicateOperator.LT,
        PredicateOperator.LTE,
        PredicateOperator.GT,
        PredicateOperator.GTE,
    }:
        left, right = _ordered_predicate_values(left, right)
        operators = {
            PredicateOperator.LT: operator.lt,
            PredicateOperator.LTE: operator.le,
            PredicateOperator.GT: operator.gt,
            PredicateOperator.GTE: operator.ge,
        }
        return bool(operators[predicate.operator](left, right))
    if predicate.operator == PredicateOperator.CONTAINS:
        return bool(right in left)
    raise RelationEngineError(f"unsupported predicate {predicate.operator}")


def _predicate_right(
    row: Row,
    predicate: Predicate,
    scalars: dict[str, object],
) -> object:
    if predicate.right:
        return _field(row, predicate.right)
    if predicate.right_scalar not in scalars:
        raise RelationEngineError(f"unknown scalar input {predicate.right_scalar}")
    return scalars[predicate.right_scalar]


def _predicate_fact(
    row: Row,
    predicate: Predicate,
    scalars: dict[str, object],
) -> tuple[object, ...]:
    left = _field(row, predicate.left)
    if predicate.operator in {PredicateOperator.IS_NULL, PredicateOperator.NOT_NULL}:
        return (_predicate(row, predicate, scalars), left)
    right = _predicate_right(row, predicate, scalars)
    return (_predicate(row, predicate, scalars), left, right)
