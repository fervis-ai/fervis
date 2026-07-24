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
    TemporalScopeType,
    TextType,
    ValueType,
)


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


__all__ = ["row_source_type_supports_semantic_type"]
