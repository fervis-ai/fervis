"""Source binding public surface for lookup orchestration."""

from fervis.lookup.source_binding.candidates import (
    bound_sources_prompt_payload,
    same_scope_read_ids,
    source_binding_candidate_payload,
    source_candidate_discovery_payload,
)
from fervis.lookup.source_binding.model import (
    AnswerPopulation,
    BoundSource,
    SourceEvidenceItem,
    SourceField,
    SourceFulfillment,
    SourceMetricFitBasis,
    SourceBindingPlan,
    SourceBindingRequest,
    SourceBindingResult,
    SourceCandidateDiscoveryRequest,
)
from fervis.lookup.source_binding.prompt import (
    SourceBindingTurnPrompt,
)
from fervis.lookup.source_binding.parser import (
    parse_source_binding,
)
from fervis.lookup.source_binding.turn import (
    SourceBindingGenerationError,
    SourceBindingTurnResult,
    generate_source_binding,
)
from fervis.lookup.source_binding.terminal_outcomes import (
    backend_impossible_without_answer_candidates,
)

__all__ = [
    "AnswerPopulation",
    "BoundSource",
    "SourceEvidenceItem",
    "SourceField",
    "SourceFulfillment",
    "SourceMetricFitBasis",
    "SourceBindingGenerationError",
    "SourceBindingPlan",
    "SourceBindingRequest",
    "SourceBindingResult",
    "SourceBindingTurnResult",
    "SourceBindingTurnPrompt",
    "SourceCandidateDiscoveryRequest",
    "bound_sources_prompt_payload",
    "backend_impossible_without_answer_candidates",
    "generate_source_binding",
    "parse_source_binding",
    "same_scope_read_ids",
    "source_binding_candidate_payload",
    "source_candidate_discovery_payload",
]
