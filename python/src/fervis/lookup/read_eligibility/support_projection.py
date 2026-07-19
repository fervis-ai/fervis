"""Canonical read-eligibility retention projections for downstream turns."""

from __future__ import annotations

from fervis.lookup.read_eligibility.model import ResolvedRetainedReadSet


def retained_source_candidate_ids_by_signature(
    read_eligibility: ResolvedRetainedReadSet,
) -> dict[str, str]:
    """Return retained model-facing candidate ids keyed by stable candidate signature."""

    return {
        item.source_candidate_signature: item.source_candidate_id
        for item in read_eligibility.retained_reads
    }


def retained_relevant_field_refs_by_candidate_id(
    read_eligibility: ResolvedRetainedReadSet,
) -> dict[str, frozenset[str]]:
    """Return retained field refs keyed by model-facing candidate id."""

    return {
        item.source_candidate_id: frozenset(item.relevant_field_refs)
        for item in read_eligibility.retained_reads
        if item.relevant_field_refs
    }
