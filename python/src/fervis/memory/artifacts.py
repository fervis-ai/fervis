"""Fact artifacts projected from prior Fervis executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fervis.memory.addresses import FactAddress
from fervis.memory._serialization import without_empty


class FactOutcome(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    IMPOSSIBLE = "impossible"
    NO_DATA = "no_data"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class FactArtifact:
    artifact_id: str
    outcome: FactOutcome
    addresses: tuple[FactAddress, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    source_question: str = ""
    source_answer: str = ""

    def address(self, address: str) -> FactAddress | None:
        for item in self.addresses:
            if item.address == address:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifactId": self.artifact_id,
            "outcome": self.outcome.value,
            "addresses": [item.to_dict() for item in self.addresses],
            "provenance": dict(self.provenance),
        }
        if self.source_question:
            payload["sourceQuestion"] = self.source_question
        if self.source_answer:
            payload["sourceAnswer"] = self.source_answer
        return without_empty(payload)


def build_fact_artifact(
    *,
    artifact_id: str,
    outcome: FactOutcome,
    addresses: tuple[FactAddress, ...] = (),
    provenance: dict[str, Any] | None = None,
    source_question: str = "",
    source_answer: str = "",
) -> FactArtifact:
    if not artifact_id:
        raise ValueError("fact artifact requires artifact_id")
    _require_unique_addresses(addresses)
    return FactArtifact(
        artifact_id=artifact_id,
        outcome=outcome,
        addresses=tuple(addresses),
        provenance=dict(provenance or {}),
        source_question=source_question,
        source_answer=source_answer,
    )


def _require_unique_addresses(addresses: tuple[FactAddress, ...]) -> None:
    seen: set[str] = set()
    for item in addresses:
        if not item.address:
            raise ValueError("fact address requires address")
        if item.address in seen:
            raise ValueError(f"duplicate fact address {item.address}")
        seen.add(item.address)
