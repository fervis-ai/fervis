"""Provider DTOs for semantic Plan Selection."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class SourceAlignmentAssessmentOutput(ProviderOutput):
    basis: str
    alignment: str


@dataclass(frozen=True)
class SemanticPlanSelectionOutput(ProviderOutput):
    source_assessments_by_requested_fact: dict[
        str,
        dict[str, SourceAlignmentAssessmentOutput],
    ]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
