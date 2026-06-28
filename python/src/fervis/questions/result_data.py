"""Shared result-data accessors for question/run surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def result_data_clarifications(
    result_data: Mapping[str, Any] | None,
) -> list[Any]:
    if result_data is None:
        return []
    details = result_data.get("details")
    if not isinstance(details, Mapping):
        return []
    clarifications = details.get("clarifications")
    if not isinstance(clarifications, list):
        return []
    return list(clarifications)
