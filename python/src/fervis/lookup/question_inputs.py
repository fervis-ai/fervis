"""Shared question-input contract tokens."""

from __future__ import annotations

from fervis.types.enums import StrEnum


class KnownInputKind(StrEnum):
    LITERAL = "literal_text"
    ROW_SET_REFERENCE = "row_set_reference"


class LiteralInputRole(StrEnum):
    REFERENCE_VALUE = "reference_value"
    TIME_VALUE = "time_value"
    RESULT_LIMIT = "result_limit"


_LITERAL_ROLE_PART_KINDS = {
    LiteralInputRole.REFERENCE_VALUE: "entity_identity",
    LiteralInputRole.TIME_VALUE: "time_scope",
    LiteralInputRole.RESULT_LIMIT: "limit",
}

def literal_role_part_kind(role: str | LiteralInputRole) -> str:
    literal_role = _literal_role(role)
    if literal_role is None:
        return ""
    return _LITERAL_ROLE_PART_KINDS[literal_role]


def _literal_role(value: str | LiteralInputRole) -> LiteralInputRole | None:
    if isinstance(value, LiteralInputRole):
        return value
    try:
        return LiteralInputRole(str(value))
    except ValueError:
        return None
