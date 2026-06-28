"""Source-candidate population binding projection."""

from __future__ import annotations

from fervis.lookup.source_binding.population_bindings import (
    PopulationBindingIndex,
)
from fervis.lookup.source_binding.model import (
    SourceBindingRequest,
    SourceCandidateDiscoveryRequest,
)

from ._shared import Any
from .candidate_tree import CandidateTreeContext, map_source_candidate_tree


def _with_population_bindings(
    payload: dict[str, Any],
    *,
    request: SourceCandidateDiscoveryRequest | SourceBindingRequest,
) -> dict[str, Any]:
    population_index = PopulationBindingIndex.from_request(request)
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _candidate_with_population_bindings(
            candidate,
            context=context,
            population_index=population_index,
        ),
        top_level_keys=("utility_source_candidates", "value_source_candidates"),
    )


def _candidate_with_population_bindings(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
    population_index: PopulationBindingIndex,
) -> dict[str, Any] | None:
    bindings = population_index.bindings_for_candidate(
        candidate,
        requested_fact_id=context.requested_fact_id,
    )
    if context.requested_fact_id and not bindings:
        return None
    output = dict(candidate)
    output["population_bindings"] = list(bindings)
    return output
