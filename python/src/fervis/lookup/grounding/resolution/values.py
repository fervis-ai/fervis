"""Grounded value identifiers and normalization."""

from __future__ import annotations

from dataclasses import replace
import re

from fervis.lookup.fact_plan.values import FactValue


def _dedupe_values(values: tuple[FactValue, ...]) -> tuple[FactValue, ...]:
    output: dict[str, FactValue] = {}
    for value in values:
        existing = output.get(value.id)
        if existing is not None and existing != value:
            output[value.id] = replace(value, id=f"{value.id}_{len(output) + 1}")
            continue
        output[value.id] = value
    return tuple(output.values())


def _grounded_value_id(known_input_id: str) -> str:
    return f"grounded_{_symbol(known_input_id)}"


def _normalize_lookup_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _symbol(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "value"
