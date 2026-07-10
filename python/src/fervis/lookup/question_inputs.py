"""Shared question-input contract tokens."""

from __future__ import annotations

from enum import StrEnum


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

_LITERAL_ROLES_BY_PART_KIND = {
    part_kind: role for role, part_kind in _LITERAL_ROLE_PART_KINDS.items()
}


def literal_role_part_kind(role: str | LiteralInputRole) -> str:
    literal_role = _literal_role(role)
    if literal_role is None:
        return ""
    return _LITERAL_ROLE_PART_KINDS[literal_role]


def literal_role_from_part_kind(part_kind: str) -> LiteralInputRole | None:
    text = part_kind.strip()
    if not text:
        return None
    role = _LITERAL_ROLES_BY_PART_KIND.get(text)
    if role is not None:
        return role
    return _literal_role(text)


def _literal_role(value: str | LiteralInputRole) -> LiteralInputRole | None:
    if isinstance(value, LiteralInputRole):
        return value
    try:
        return LiteralInputRole(str(value))
    except ValueError:
        return None
