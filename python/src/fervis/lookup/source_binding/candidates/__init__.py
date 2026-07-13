"""Public source-binding candidate registry boundary."""

from .model import SourceCandidate, SourceCandidateRegistry
from .registry_builder import parse_source_candidate_registry
from .registry import (
    bound_sources_prompt_payload,
    same_scope_read_ids,
    source_binding_candidate_payload,
    source_candidate_discovery_registry,
    source_candidate_discovery_payload,
    source_binding_prompt_candidate_fulfillment_answer_output_ids,
    source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output,
    source_binding_prompt_candidate_population_binding_ids,
    source_binding_prompt_candidate_requested_fact_ids,
    source_candidate_registry,
    source_candidate_required_param_decision_ids,
    source_candidates,
)


__all__ = [
    "SourceCandidate",
    "SourceCandidateRegistry",
    "parse_source_candidate_registry",
    "bound_sources_prompt_payload",
    "same_scope_read_ids",
    "source_binding_candidate_payload",
    "source_candidate_discovery_registry",
    "source_candidate_discovery_payload",
    "source_binding_prompt_candidate_fulfillment_answer_output_ids",
    "source_binding_prompt_candidate_fulfillment_support_set_ids_by_answer_output",
    "source_binding_prompt_candidate_population_binding_ids",
    "source_binding_prompt_candidate_requested_fact_ids",
    "source_candidate_required_param_decision_ids",
    "source_candidate_registry",
    "source_candidates",
]
