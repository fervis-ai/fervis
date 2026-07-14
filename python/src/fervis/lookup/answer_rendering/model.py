"""Rendered fact response model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from fervis.lookup.canonical_data import RuntimeValue

if TYPE_CHECKING:
    from fervis.lookup.outcomes.model import OutcomeKind


@dataclass(frozen=True)
class RenderedFact:
    kind: "OutcomeKind"
    rows: tuple[Mapping[str, RuntimeValue], ...] = ()
    row_labels: Mapping[str, str] | None = None
    scalars: Mapping[str, RuntimeValue] | None = None
    message: str = ""
    details: Mapping[str, RuntimeValue] | None = None
    proof_refs: tuple[str, ...] = ()
    render_outputs: tuple[Mapping[str, RuntimeValue], ...] = ()
