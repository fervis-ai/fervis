"""Grounding pipeline result model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.grounding.model import CanonicalInputLedger
from fervis.lookup.grounding.turn import GroundingTurnResult


@dataclass(frozen=True)
class GroundingOutput:
    ledger: CanonicalInputLedger
    turn: GroundingTurnResult | None = None

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self.turn.usage) if self.turn is not None else {}
