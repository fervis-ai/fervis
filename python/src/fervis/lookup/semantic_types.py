"""Closed value algebra for semantic and executable relational expressions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import TypeAlias

from fervis.types.enums import StrEnum


INTEGER_OPERAND_PATTERN = r"^[+-]?[0-9]+$"
DECIMAL_OPERAND_PATTERN = r"^[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?$"


class SourceOriginKind(StrEnum):
    QUESTION_CONTEXT = "question_context"
    CONVERSATION_RESOLUTION = "conversation_resolution"


@dataclass(frozen=True)
class SourceOrigin:
    source: SourceOriginKind
    meaning: str
    resolved_input_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.meaning:
            raise ValueError("source origin requires meaning")
        if self.source is SourceOriginKind.QUESTION_CONTEXT:
            if self.resolved_input_ref is not None:
                raise ValueError(
                    "question source origin cannot reference resolved input"
                )
        elif not self.resolved_input_ref:
            raise ValueError("conversation source origin requires resolved input")


@dataclass(frozen=True)
class BooleanType:
    pass


@dataclass(frozen=True)
class IntegerType:
    pass


@dataclass(frozen=True)
class TextType:
    pass


@dataclass(frozen=True)
class DateType:
    pass


@dataclass(frozen=True)
class DateTimeType:
    pass


@dataclass(frozen=True)
class TemporalScopeType:
    pass


@dataclass(frozen=True)
class UnspecifiedScalarType:
    """A projected scalar whose source capability is not otherwise constrained."""


@dataclass(frozen=True)
class NumericType:
    """A scalar required to support numeric operations."""


@dataclass(frozen=True)
class OrderableType:
    """A scalar required to support deterministic ordering."""


@dataclass(frozen=True)
class TemporalPointType:
    """A date or datetime value required by a temporal operation."""


@dataclass(frozen=True)
class UnitlessMeasure:
    pass


@dataclass(frozen=True)
class CountMeasure:
    pass


@dataclass(frozen=True)
class RatioMeasure:
    pass


@dataclass(frozen=True)
class PercentageMeasure:
    pass


@dataclass(frozen=True)
class ContextualCurrency:
    pass


@dataclass(frozen=True)
class SourceNamedCurrency:
    origin: SourceOrigin


Currency: TypeAlias = ContextualCurrency | SourceNamedCurrency


@dataclass(frozen=True)
class MoneyMeasure:
    currency: Currency


@dataclass(frozen=True)
class ItemQuantityUnit:
    pass


@dataclass(frozen=True)
class SourceNamedQuantityUnit:
    origin: SourceOrigin


QuantityUnit: TypeAlias = ItemQuantityUnit | SourceNamedQuantityUnit


@dataclass(frozen=True)
class QuantityMeasure:
    unit: QuantityUnit


Measure: TypeAlias = (
    UnitlessMeasure
    | CountMeasure
    | MoneyMeasure
    | QuantityMeasure
    | RatioMeasure
    | PercentageMeasure
)


@dataclass(frozen=True)
class DecimalType:
    measure: Measure


class TimeUnit(StrEnum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass(frozen=True)
class DurationType:
    unit: TimeUnit


@dataclass(frozen=True)
class IdentifierType:
    set_ref: str

    def __post_init__(self) -> None:
        if not self.set_ref:
            raise ValueError("identifier type requires set reference")


ScalarType: TypeAlias = (
    BooleanType
    | IntegerType
    | DecimalType
    | TextType
    | DateType
    | DateTimeType
    | DurationType
    | IdentifierType
    | TemporalScopeType
    | UnspecifiedScalarType
    | NumericType
    | OrderableType
    | TemporalPointType
)


@dataclass(frozen=True)
class CollectionType:
    element_type: ScalarType


ValueType: TypeAlias = ScalarType | CollectionType


def value_type_kind(value_type: ValueType) -> str:
    """Return the canonical provider-facing kind for a semantic value type."""

    if isinstance(value_type, CollectionType):
        return "collection"
    return {
        BooleanType: "boolean",
        IntegerType: "integer",
        TextType: "text",
        DateType: "date",
        DateTimeType: "datetime",
        TemporalScopeType: "temporal_scope",
        DecimalType: "decimal",
        DurationType: "duration",
        IdentifierType: "identifier",
        UnspecifiedScalarType: "value",
        NumericType: "number",
        OrderableType: "orderable",
        TemporalPointType: "temporal",
    }[type(value_type)]


def is_numeric(value_type: ValueType) -> bool:
    return isinstance(value_type, (IntegerType, DecimalType, NumericType))


def normalized_numeric_type(value_type: ValueType) -> ValueType:
    if isinstance(value_type, DecimalType) and isinstance(
        value_type.measure, PercentageMeasure
    ):
        return DecimalType(RatioMeasure())
    return value_type


def input_operand_matches_value_type(
    operand: str | tuple[str, ...],
    value_type: ValueType,
) -> bool:
    """Return whether supplied operand syntax can represent its declared type."""

    if isinstance(value_type, CollectionType):
        return isinstance(operand, tuple) and all(
            input_operand_matches_value_type(item, value_type.element_type)
            for item in operand
        )
    if not isinstance(operand, str):
        return False
    if isinstance(value_type, BooleanType):
        return operand in {"true", "false"}
    if isinstance(value_type, IntegerType):
        return re.fullmatch(INTEGER_OPERAND_PATTERN, operand) is not None
    if isinstance(value_type, (DecimalType, NumericType)):
        if re.fullmatch(DECIMAL_OPERAND_PATTERN, operand) is None:
            return False
        try:
            Decimal(operand.removesuffix("%"))
        except InvalidOperation:
            return False
    return True


__all__ = tuple(name for name in globals() if not name.startswith("_"))
