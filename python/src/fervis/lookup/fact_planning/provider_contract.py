"""Provider-output DTOs for fact planning."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


FactPlanOutput = provider_output_type("FactPlanOutput", ("outcome",))
FactPlanAnswerOutput = provider_output_type(
    "FactPlanAnswerOutput",
    ("kind", "answers"),
)
PlanImpossibleOutput = provider_output_type(
    "PlanImpossibleOutput",
    ("kind", "blocked_facts"),
)
BlockedFactOutput = provider_output_type(
    "BlockedFactOutput",
    (
        "requested_fact_id",
        "basis",
        "evidence_refs",
        "reviewed_read_ids",
        "nearest_fields",
        "explanation",
    ),
    optional_fields=("reviewed_read_ids", "nearest_fields", "explanation"),
)
BlockedFactFieldOutput = provider_output_type(
    "BlockedFactFieldOutput",
    ("read_id", "field_id"),
)
PlanClarificationOutput = provider_output_type(
    "PlanClarificationOutput",
    ("kind", "missing_catalog_inputs"),
)
MissingCatalogRequiredInputOutput = provider_output_type(
    "MissingCatalogRequiredInputOutput",
    ("kind", "id", "requested_fact_id", "required_catalog_input_id"),
)
MissingCatalogChoiceInputOutput = provider_output_type(
    "MissingCatalogChoiceInputOutput",
    ("kind", "id", "requested_fact_id", "required_catalog_choice_input_id"),
)
