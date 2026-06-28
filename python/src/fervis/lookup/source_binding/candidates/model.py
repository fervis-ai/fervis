"""Source-binding candidate registry models."""

from ._shared import Any, EndpointParamBinding, RelationSource, dataclass


@dataclass(frozen=True)
class SourceCandidate:
    id: str
    requested_fact_id: str
    kind: str
    source: RelationSource | None = None
    value_id: str = ""
    applies_to_requested_fact_ids: tuple[str, ...] = ()
    params: tuple[dict[str, Any], ...] = ()
    applied_param_bindings: tuple[EndpointParamBinding, ...] = ()
    applied_param_binding_sets: tuple[tuple[EndpointParamBinding, ...], ...] = ()
    fields: tuple[dict[str, Any], ...] = ()
    population_bindings: tuple[dict[str, Any], ...] = ()
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceCandidateRegistry:
    prompt_payload: dict[str, Any]
    candidates_by_id: dict[str, SourceCandidate]
    prompt_candidate_ids: tuple[str, ...] = ()
