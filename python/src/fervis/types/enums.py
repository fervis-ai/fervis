"""String enum contract available on every supported Python runtime."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Enum whose members behave and render as their string values."""

    def __str__(self) -> str:
        return str.__str__(self)

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        del start, count, last_values
        return name.lower()
