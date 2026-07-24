"""Question-independent runtime values available during lookup compilation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeValueContext:
    runtime_date: str
    timezone: str


__all__ = ["RuntimeValueContext"]
