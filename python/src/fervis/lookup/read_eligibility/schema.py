"""Provider schema for read-eligibility retention."""

from __future__ import annotations

from fervis.lookup.read_eligibility import provider_contract as provider_output


def build_read_eligibility_schema(
    *,
    candidate_reviews_by_requested_fact_id: dict[str, tuple[dict[str, object], ...]],
) -> dict[str, object]:
    requested_fact_assessment_variants = tuple(
        _requested_fact_assessment_schema(
            requested_fact_id=requested_fact_id,
            candidate_specs=candidate_specs,
        )
        for requested_fact_id, candidate_specs in (
            candidate_reviews_by_requested_fact_id.items()
        )
    )
    requested_fact_assessments_schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "object"},
        "minItems": len(candidate_reviews_by_requested_fact_id),
        "maxItems": len(candidate_reviews_by_requested_fact_id),
    }
    if requested_fact_assessment_variants:
        requested_fact_assessments_schema["prefixItems"] = list(
            requested_fact_assessment_variants
        )
    return provider_output.ReadEligibilityOutput.schema(
        {
            "requested_fact_assessments": requested_fact_assessments_schema,
        },
    )


def _requested_fact_assessment_schema(
    *,
    requested_fact_id: str,
    candidate_specs: tuple[dict[str, object], ...],
) -> dict[str, object]:
    read_candidate_reviews_schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "object"},
        "minItems": len(candidate_specs),
        "maxItems": len(candidate_specs),
    }
    if candidate_specs:
        read_candidate_reviews_schema["prefixItems"] = [
            _candidate_read_candidate_review_schema(
                candidate_spec=candidate_spec,
            )
            for candidate_spec in candidate_specs
        ]
    return provider_output.RequestedFactAssessmentOutput.schema(
        {
            "requested_fact_id": _string_schema(enum_values=(requested_fact_id,)),
            "read_candidate_reviews": read_candidate_reviews_schema,
        },
    )


def _candidate_read_candidate_review_schema(
    *,
    candidate_spec: dict[str, object],
) -> dict[str, object]:
    return {
        "oneOf": [
            _read_candidate_review_schema(
                candidate_id_schema=_string_schema(
                    enum_values=(str(candidate_spec["source_candidate_id"]),),
                ),
                read_id_schema=_string_schema(
                    enum_values=(str(candidate_spec["read_id"]),),
                ),
                row_path_token_schema=_string_schema(
                    enum_values=_candidate_spec_values(
                        candidate_spec,
                        key="row_path_tokens",
                    ),
                ),
                field_token_schema=_string_schema(
                    enum_values=_candidate_spec_values(
                        candidate_spec,
                        key="field_tokens",
                    ),
                ),
                retention_decision="RETAIN",
            ),
            _read_candidate_review_schema(
                candidate_id_schema=_string_schema(
                    enum_values=(str(candidate_spec["source_candidate_id"]),),
                ),
                read_id_schema=_string_schema(
                    enum_values=(str(candidate_spec["read_id"]),),
                ),
                row_path_token_schema={"type": "string"},
                field_token_schema={"type": "string"},
                retention_decision="DROP",
            ),
        ]
    }


def _read_candidate_review_schema(
    *,
    candidate_id_schema: dict[str, object],
    read_id_schema: dict[str, object],
    row_path_token_schema: dict[str, object],
    field_token_schema: dict[str, object],
    retention_decision: str,
) -> dict[str, object]:
    relevant_row_path_tokens_schema: dict[str, object] = {
        "type": "array",
        "items": row_path_token_schema,
    }
    relevant_field_tokens_schema: dict[str, object] = {
        "type": "array",
        "items": field_token_schema,
    }
    if retention_decision == "DROP":
        relevant_row_path_tokens_schema["maxItems"] = 0
        relevant_field_tokens_schema["maxItems"] = 0
    return provider_output.ReadCandidateReviewOutput.schema(
        {
            "source_candidate_id": candidate_id_schema,
            "read_id": read_id_schema,
            "relevant_row_path_tokens": relevant_row_path_tokens_schema,
            "relevant_field_tokens": relevant_field_tokens_schema,
            "retention_basis": {"type": "string", "minLength": 1},
            "retention_decision": {"enum": [retention_decision]},
        },
    )


def _string_schema(*, enum_values: tuple[str, ...]) -> dict[str, object]:
    schema: dict[str, object] = {"type": "string", "minLength": 1}
    if enum_values:
        schema["enum"] = list(enum_values)
    return schema


def _candidate_spec_values(
    candidate_spec: dict[str, object],
    *,
    key: str,
) -> tuple[str, ...]:
    raw_values = candidate_spec.get(key)
    values = raw_values if isinstance(raw_values, (list, tuple)) else ()
    return tuple(str(value) for value in values if str(value))
