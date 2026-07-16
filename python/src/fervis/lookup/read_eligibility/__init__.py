"""Read-eligibility retention public boundary."""

from fervis.lookup.read_eligibility.filtering import (
    filter_catalog_selection_for_read_eligibility,
)
from fervis.lookup.read_eligibility.model import (
    READ_ELIGIBILITY_RECALL_READS_PER_FACT,
    RETENTION_DECISION_VALUES,
    CanonicalInputSelection,
    DroppedReadAssessment,
    ReadAssessment,
    ReadEligibilityRequest,
    ReadEligibilityResult,
    ResolvedRetainedReadSet,
    RetainedReadAssessment,
)
from fervis.lookup.read_eligibility.parser import parse_read_eligibility
from fervis.lookup.read_eligibility.prompt import (
    READ_ELIGIBILITY_TOOL_NAME,
    ReadEligibilityTurnPrompt,
)
from fervis.lookup.read_eligibility.recall import (
    prepare_catalog_selection_for_read_eligibility,
)
from fervis.lookup.read_eligibility.schema import build_read_eligibility_schema
from fervis.lookup.read_eligibility.support_projection import (
    retained_relevant_field_refs_by_candidate_id,
    retained_source_candidate_ids_by_signature,
)
from fervis.lookup.read_eligibility.surface import (
    ReadEligibilityCandidateSurface,
    read_eligibility_candidate_surface,
)
from fervis.lookup.read_eligibility.turn import (
    ReadEligibilityGenerationError,
    ReadEligibilityTurnResult,
    generate_read_eligibility,
)

__all__ = [
    "READ_ELIGIBILITY_TOOL_NAME",
    "READ_ELIGIBILITY_RECALL_READS_PER_FACT",
    "RETENTION_DECISION_VALUES",
    "CanonicalInputSelection",
    "DroppedReadAssessment",
    "ReadAssessment",
    "ReadEligibilityGenerationError",
    "ReadEligibilityRequest",
    "ReadEligibilityResult",
    "ResolvedRetainedReadSet",
    "RetainedReadAssessment",
    "ReadEligibilityCandidateSurface",
    "ReadEligibilityTurnPrompt",
    "ReadEligibilityTurnResult",
    "build_read_eligibility_schema",
    "filter_catalog_selection_for_read_eligibility",
    "generate_read_eligibility",
    "parse_read_eligibility",
    "prepare_catalog_selection_for_read_eligibility",
    "read_eligibility_candidate_surface",
    "retained_relevant_field_refs_by_candidate_id",
    "retained_source_candidate_ids_by_signature",
]
