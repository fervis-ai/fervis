"""Grounding pipeline result model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.grounding.model import CanonicalInputLedger, KnownInputBindingTask
from fervis.lookup.grounding.turn import GroundingTurnResult


@dataclass(frozen=True)
class GroundingOutput:
    ledger: CanonicalInputLedger
    binding_tasks: tuple[KnownInputBindingTask, ...] = ()
    turn: GroundingTurnResult | None = None

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self.turn.usage) if self.turn is not None else {}
