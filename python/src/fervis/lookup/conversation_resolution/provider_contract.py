"""Provider-output DTOs for conversation resolution."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


ConversationResolutionOutput = provider_output_type(
    "ConversationResolutionOutput",
    ("kind", "current_question_text", "clause_resolutions", "unresolved"),
)
ClauseResolutionOutput = provider_output_type(
    "ClauseResolutionOutput",
    (
        "current_clause_text",
        "occurrence",
        "requested_value_frame",
        "continuation",
        "dependencies",
        "resolved_clause_text",
    ),
    optional_fields=("continuation",),
)
ContinuationOutput = provider_output_type(
    "ContinuationOutput",
    ("kind", "frame_id", "replacements"),
)
ContinuationReplacementOutput = provider_output_type(
    "ContinuationReplacementOutput",
    ("part_id", "current_text"),
)
RequestedValueFrameOutput = provider_output_type(
    "RequestedValueFrameOutput",
    ("current_value_surface", "context_frame_choices"),
)
CurrentValueSurfaceOutput = provider_output_type(
    "CurrentValueSurfaceOutput",
    ("text", "kind"),
)
ContextFrameChoiceOutput = provider_output_type(
    "ContextFrameChoiceOutput",
    ("frame_id", "current_conflict_quotes", "choice"),
)
DependencyOutput = provider_output_type(
    "DependencyOutput",
    (
        "anchor_text",
        "occurrence",
        "meaning_components",
        "resolved_text",
        "must_preserve_terms",
        "kind",
    ),
)
MeaningComponentOutput = provider_output_type(
    "MeaningComponentOutput",
    ("source_id", "source_text", "memory_id", "resolved_text", "kind"),
)
UnresolvedOutput = provider_output_type(
    "UnresolvedOutput",
    ("why_unresolved", "candidate_interpretations", "unresolved_kind"),
)
CandidateInterpretationOutput = provider_output_type(
    "CandidateInterpretationOutput",
    ("integrated_question", "supporting_evidence"),
)
SourceEvidenceOutput = provider_output_type(
    "SourceEvidenceOutput",
    ("source_id", "exact_source_texts"),
)
