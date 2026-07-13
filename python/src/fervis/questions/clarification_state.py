"""Current clarification state derived from append-only lineage."""

from __future__ import annotations


def pending_clarification_ids(
    request_ids: tuple[str, ...],
    response_clarification_ids: tuple[str, ...],
) -> tuple[str, ...]:
    responded_ids = frozenset(response_clarification_ids)
    return tuple(
        clarification_id
        for clarification_id in request_ids
        if clarification_id not in responded_ids
    )
