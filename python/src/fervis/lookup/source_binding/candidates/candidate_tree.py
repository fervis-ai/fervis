"""Source-binding candidate payload traversal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ._shared import Any


@dataclass(frozen=True)
class CandidateTreeContext:
    requested_fact_id: str = ""
    top_level_key: str = ""


CandidateMapper = Callable[
    [dict[str, Any], CandidateTreeContext],
    dict[str, Any] | None,
]


def map_source_candidate_tree(
    payload: dict[str, Any],
    mapper: CandidateMapper,
    *,
    top_level_keys: tuple[str, ...] = (
        "memory_source_candidates",
        "utility_source_candidates",
        "value_source_candidates",
    ),
) -> dict[str, Any]:
    output = dict(payload)
    output["requested_fact_sources"] = [
        _map_fact_sources(fact_sources, mapper=mapper)
        for fact_sources in output.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
    ]
    for key in top_level_keys:
        if key in output:
            output[key] = [
                mapped
                for candidate in output.get(key) or ()
                if isinstance(candidate, dict)
                for mapped in (
                    mapper(
                        candidate,
                        CandidateTreeContext(top_level_key=key),
                    ),
                )
                if mapped is not None
            ]
    return output


def _map_fact_sources(
    fact_sources: dict[str, Any],
    *,
    mapper: CandidateMapper,
) -> dict[str, Any]:
    requested_fact_id = str(fact_sources.get("requested_fact_id") or "")
    output = dict(fact_sources)
    output["source_contexts"] = [
        _map_source_context(
            context,
            mapper=mapper,
            requested_fact_id=requested_fact_id,
        )
        for context in output.get("source_contexts") or ()
        if isinstance(context, dict)
    ]
    return output


def _map_source_context(
    context: dict[str, Any],
    *,
    mapper: CandidateMapper,
    requested_fact_id: str,
) -> dict[str, Any]:
    output = dict(context)
    output["source_options"] = [
        mapped
        for candidate in output.get("source_options") or ()
        if isinstance(candidate, dict)
        for mapped in (
            mapper(
                candidate, CandidateTreeContext(requested_fact_id=requested_fact_id)
            ),
        )
        if mapped is not None
    ]
    return output
