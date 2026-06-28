"""Canonical read-eligibility retention projections for downstream turns."""

from __future__ import annotations

from typing import Protocol


class ReadAssessmentLike(Protocol):
    source_candidate_id: str
    source_candidate_signature: str
    is_retained: bool


class ReadEligibilityResultLike(Protocol):
    read_assessments: tuple[ReadAssessmentLike, ...]


def retained_source_candidate_ids_by_signature(
    read_eligibility: ReadEligibilityResultLike,
) -> dict[str, str]:
    """Return retained model-facing candidate ids keyed by stable candidate signature."""

    return {
        item.source_candidate_signature: item.source_candidate_id
        for item in read_eligibility.read_assessments
        if item.is_retained
    }
