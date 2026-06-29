"""Canonical read-eligibility retention projections for downstream turns."""

from __future__ import annotations

from typing import Protocol


class ReadAssessmentLike(Protocol):
    source_candidate_id: str
    source_candidate_signature: str
    is_retained: bool
    relevant_field_refs: tuple[str, ...]


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


def retained_relevant_field_refs_by_candidate_id(
    read_eligibility: ReadEligibilityResultLike,
) -> dict[str, frozenset[str]]:
    """Return retained field refs keyed by model-facing candidate id."""

    return {
        item.source_candidate_id: frozenset(item.relevant_field_refs)
        for item in read_eligibility.read_assessments
        if item.is_retained and item.relevant_field_refs
    }
