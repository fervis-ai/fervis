"""Compatibility between semantic values and declared row-source fields."""

from __future__ import annotations

from fervis.lookup.relation_catalog.row_sources.model import RowSourceValueType
from fervis.lookup.semantic_types import (
    BooleanType,
    CollectionType,
    DateTimeType,
    DateType,
    DecimalType,
    DurationType,
    IdentifierType,
    IntegerType,
    NumericType,
    OrderableType,
    TemporalPointType,
    TemporalScopeType,
    TextType,
    TimeUnit,
    UnitlessMeasure,
    UnspecifiedScalarType,
    ValueType,
)


def semantic_type_for_row_source_type(source_type: RowSourceValueType) -> ValueType:
    """Project one concrete catalog field type into the expression type system."""

    if source_type is RowSourceValueType.BOOLEAN:
        return BooleanType()
    if source_type is RowSourceValueType.INTEGER:
        return IntegerType()
    if source_type in {
        RowSourceValueType.NUMBER,
        RowSourceValueType.DECIMAL,
        RowSourceValueType.FLOAT,
        RowSourceValueType.DOUBLE,
    }:
        return DecimalType(UnitlessMeasure())
    if source_type is RowSourceValueType.DATE:
        return DateType()
    if source_type is RowSourceValueType.DATETIME:
        return DateTimeType()
    if source_type is RowSourceValueType.DURATION:
        return DurationType(TimeUnit.SECOND)
    if source_type in {
        RowSourceValueType.CHOICE,
        RowSourceValueType.PK,
        RowSourceValueType.STRING,
        RowSourceValueType.TIME,
        RowSourceValueType.UUID,
    }:
        return TextType()
    raise ValueError(f"{source_type.value} is not a concrete scalar field type")


def row_source_type_supports_semantic_type(
    source_type: RowSourceValueType,
    semantic_type: ValueType,
) -> bool:
    """Return whether a declared source field can represent a semantic value."""

    if isinstance(semantic_type, CollectionType):
        return source_type in {RowSourceValueType.ARRAY, RowSourceValueType.LIST}
    if isinstance(semantic_type, BooleanType):
        return source_type is RowSourceValueType.BOOLEAN
    if isinstance(semantic_type, IntegerType):
        return source_type is RowSourceValueType.INTEGER
    if isinstance(semantic_type, DecimalType):
        return source_type in {
            RowSourceValueType.INTEGER,
            RowSourceValueType.NUMBER,
            RowSourceValueType.DECIMAL,
            RowSourceValueType.FLOAT,
            RowSourceValueType.DOUBLE,
        }
    if isinstance(semantic_type, NumericType):
        return source_type in {
            RowSourceValueType.INTEGER,
            RowSourceValueType.NUMBER,
            RowSourceValueType.DECIMAL,
            RowSourceValueType.FLOAT,
            RowSourceValueType.DOUBLE,
        }
    if isinstance(semantic_type, OrderableType):
        return source_type in {
            RowSourceValueType.CHOICE,
            RowSourceValueType.DATE,
            RowSourceValueType.DATETIME,
            RowSourceValueType.DECIMAL,
            RowSourceValueType.DOUBLE,
            RowSourceValueType.DURATION,
            RowSourceValueType.FLOAT,
            RowSourceValueType.INTEGER,
            RowSourceValueType.NUMBER,
            RowSourceValueType.STRING,
            RowSourceValueType.TIME,
        }
    if isinstance(semantic_type, TemporalPointType):
        return source_type in {
            RowSourceValueType.DATE,
            RowSourceValueType.DATETIME,
        }
    if isinstance(semantic_type, UnspecifiedScalarType):
        return source_type in {
            RowSourceValueType.BOOLEAN,
            RowSourceValueType.CHOICE,
            RowSourceValueType.DATE,
            RowSourceValueType.DATETIME,
            RowSourceValueType.DECIMAL,
            RowSourceValueType.DOUBLE,
            RowSourceValueType.DURATION,
            RowSourceValueType.FLOAT,
            RowSourceValueType.INTEGER,
            RowSourceValueType.NUMBER,
            RowSourceValueType.PK,
            RowSourceValueType.STRING,
            RowSourceValueType.TIME,
            RowSourceValueType.UUID,
        }
    if isinstance(semantic_type, TextType):
        return source_type in {
            RowSourceValueType.STRING,
            RowSourceValueType.CHOICE,
        }
    if isinstance(semantic_type, DateType):
        return source_type in {
            RowSourceValueType.DATE,
            RowSourceValueType.DATETIME,
        }
    if isinstance(semantic_type, DateTimeType):
        return source_type is RowSourceValueType.DATETIME
    if isinstance(semantic_type, DurationType):
        return source_type is RowSourceValueType.DURATION
    if isinstance(semantic_type, IdentifierType):
        return source_type in {
            RowSourceValueType.INTEGER,
            RowSourceValueType.PK,
            RowSourceValueType.STRING,
            RowSourceValueType.UUID,
        }
    if isinstance(semantic_type, TemporalScopeType):
        return False
    raise TypeError(f"unsupported semantic value type {type(semantic_type).__name__}")


__all__ = [
    "row_source_type_supports_semantic_type",
    "semantic_type_for_row_source_type",
]
