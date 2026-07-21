"""Shared question-input contract tokens."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from fervis.types.enums import StrEnum


class KnownInputKind(StrEnum):
    LITERAL = "literal_text"
    ROW_SET_REFERENCE = "row_set_reference"


class LiteralInputRole(StrEnum):
    REFERENCE_VALUE = "reference_value"
    PREDICATE_VALUE = "predicate_value"
    TIME_VALUE = "time_value"
    THRESHOLD_VALUE = "threshold_value"
    FORMULA_VALUE = "formula_value"
    RESULT_LIMIT = "result_limit"


_LITERAL_ROLE_PART_KINDS = {
    LiteralInputRole.REFERENCE_VALUE: "entity_identity",
    LiteralInputRole.PREDICATE_VALUE: "predicate_operand",
    LiteralInputRole.TIME_VALUE: "time_scope",
    LiteralInputRole.THRESHOLD_VALUE: "comparison_boundary",
    LiteralInputRole.FORMULA_VALUE: "computation_operand",
    LiteralInputRole.RESULT_LIMIT: "limit",
}


def literal_role_part_kind(role: str | LiteralInputRole) -> str:
    literal_role = _literal_role(role)
    if literal_role is None:
        return ""
    return _LITERAL_ROLE_PART_KINDS[literal_role]


def normalize_scalar_literal_text(value: str) -> tuple[str, str]:
    """Parse one supported question scalar into its canonical literal form."""

    text = value.strip()
    if not text:
        raise ValueError("scalar literal must not be empty")
    if text.endswith("%"):
        number = _decimal_number(normalize_decimal_text(text[:-1])) / Decimal("100")
        return "number", normalize_decimal_text(str(number))
    try:
        return "number", normalize_decimal_text(text)
    except ValueError:
        pass
    boolean = text.casefold()
    if boolean in {"true", "false"}:
        return "boolean", boolean
    return "string", text


def _literal_role(value: str | LiteralInputRole) -> LiteralInputRole | None:
    if isinstance(value, LiteralInputRole):
        return value
    try:
        return LiteralInputRole(str(value))
    except ValueError:
        return None


def normalize_decimal_text(value: str) -> str:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("number contains a non-numeric value") from exc
    if not parsed.is_finite():
        raise ValueError("number must be finite")
    if parsed == 0:
        return "0"
    return format(parsed.normalize(), "f")


def _decimal_number(value: str) -> Decimal:
    return Decimal(value)
