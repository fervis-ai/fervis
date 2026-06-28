"""Rendered fact response model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from fervis.lookup.outcomes.model import OutcomeKind


@dataclass(frozen=True)
class RenderedFact:
    kind: "OutcomeKind"
    rows: tuple[Mapping[str, object], ...] = ()
    row_labels: Mapping[str, str] | None = None
    scalars: Mapping[str, object] | None = None
    message: str = ""
    details: Mapping[str, object] | None = None
    proof_refs: tuple[str, ...] = ()
    render_outputs: tuple[Mapping[str, object], ...] = ()
