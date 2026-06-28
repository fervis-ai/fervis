"""Value-use helpers for pattern fact-plan compilation."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding import BoundSource

from .shared import _required_dicts, _text


def _scalar_inputs(
    value: Any,
    *,
    bound_sources: dict[str, BoundSource],
) -> tuple[dict[str, str], ...]:
    output: list[dict[str, str]] = []
    for item in _required_dicts(value, "scalar_inputs"):
        source_binding_id = _text(item.get("source_binding_id"))
        bound = bound_sources.get(source_binding_id)
        if bound is None or not bound.value_id:
            raise ValueError("scalar input requires value source binding")
        output.append(
            {
                "input_id": _text(item.get("input_id")),
                "value_id": bound.value_id,
            }
        )
    return tuple(output)
