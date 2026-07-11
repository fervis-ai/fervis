"""Provider-output DTOs for conversation resolution."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


ConversationResolutionOutput = provider_output_type(
    "ConversationResolutionOutput",
    (
        "kind",
        "current_question_text",
        "outcome",
    ),
)
ResolvedOutcomeOutput = provider_output_type(
    "ResolvedOutcomeOutput",
    (
        "kind",
        "resolution_basis",
        "contextualized_question",
        "clauses",
        "frame_call",
    ),
)
ResolvedClauseOutput = provider_output_type(
    "ResolvedClauseOutput",
    (
        "current_clause_text",
        "occurrence",
        "resolved_text",
        "retained_frame_parts",
        "values",
    ),
)
ResolvedValueOutput = provider_output_type(
    "ResolvedValueOutput",
    ("value_id", "resolved_text", "sources"),
)
CurrentSpanSourceOutput = provider_output_type(
    "CurrentSpanSourceOutput",
    ("kind", "text", "occurrence"),
)
ContextAnchorSourceOutput = provider_output_type(
    "ContextAnchorSourceOutput",
    ("kind", "source_id", "memory_id", "source_text"),
)
FramePartSourceOutput = provider_output_type(
    "FramePartSourceOutput",
    ("kind", "frame_id", "part_id"),
)
NoFrameCallOutput = provider_output_type("NoFrameCallOutput", ("kind",))
FrameCallOutput = provider_output_type(
    "FrameCallOutput",
    ("kind", "frame_id", "arguments"),
)
CarriedFrameArgumentOutput = provider_output_type(
    "CarriedFrameArgumentOutput",
    ("kind", "parameter_id"),
)
ResolvedValueFrameArgumentOutput = provider_output_type(
    "ResolvedValueFrameArgumentOutput",
    ("kind", "parameter_id", "value_id"),
)
UnresolvedOutcomeOutput = provider_output_type(
    "UnresolvedOutcomeOutput",
    ("kind", "why_unresolved", "candidate_interpretations"),
)
CandidateInterpretationOutput = provider_output_type(
    "CandidateInterpretationOutput",
    ("contextualized_question", "supporting_evidence"),
)
SourceEvidenceOutput = provider_output_type(
    "SourceEvidenceOutput",
    ("source_id", "exact_source_texts"),
)
