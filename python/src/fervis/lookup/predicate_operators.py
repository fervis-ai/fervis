"""Canonical operators for fact-owned and executable predicates."""

from fervis.types.enums import StrEnum


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


__all__ = ["PredicateOperator"]
