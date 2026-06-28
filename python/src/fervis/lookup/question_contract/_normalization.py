"""Shared question-contract text normalization helpers."""

from __future__ import annotations


def number_text(value: object) -> str:
    return str(value).strip().replace(",", "").removesuffix("%")
