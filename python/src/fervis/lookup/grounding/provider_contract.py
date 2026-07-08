"""Provider-output DTOs for grounding."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


GroundingOutput = provider_output_type(
    "GroundingOutput",
    ("known_time_resolutions", "known_input_binding_reviews"),
)
KnownTimeResolutionOutput = provider_output_type(
    "KnownTimeResolutionOutput",
    ("date_intent",),
)
DateIntentOutput = provider_output_type(
    "DateIntentOutput",
    ("expression", "intent"),
)
KnownInputBindingReviewOutput = provider_output_type(
    "KnownInputBindingReviewOutput",
    ("option_reviews",),
)
OptionReviewOutput = provider_output_type(
    "OptionReviewOutput",
    ("resolver_fit_question", "because", "decision"),
)
