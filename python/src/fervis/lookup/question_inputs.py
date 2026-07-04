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
