"""Compiled values retain their logical row domain."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationalValue:
    relation_id: str
    key_set_ref: str | None
    key_fields: tuple[str, ...]
    value_field: str
    operation_id: str
