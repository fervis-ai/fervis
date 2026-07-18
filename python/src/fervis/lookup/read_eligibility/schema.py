"""Provider schema for read-eligibility retention."""

from __future__ import annotations

from fervis.lookup.read_eligibility import provider_contract as provider_output


def build_read_eligibility_schema(
    *,
    canonical_options_by_requested_fact_id: dict[
        str,
        tuple[dict[str, object], ...],
    ],
    candidate_reviews_by_requested_fact_id: dict[str, tuple[dict[str, object], ...]],
) -> dict[str, object]:
    requested_fact_assessments: dict[str, object] = {
        requested_fact_id: _requested_fact_assessment_schema(
            canonical_specs=canonical_options_by_requested_fact_id.get(
                requested_fact_id,
                (),
            ),
            candidate_specs=candidate_specs,
        )
        for requested_fact_id, candidate_specs in (
            candidate_reviews_by_requested_fact_id.items()
        )
    }
    return provider_output.ReadEligibilityOutput.schema(
        {
            "requested_fact_assessments": _closed_object_schema(
                requested_fact_assessments
            ),
        },
    )


def _requested_fact_assessment_schema(
    *,
    canonical_specs: tuple[dict[str, object], ...],
    candidate_specs: tuple[dict[str, object], ...],
) -> dict[str, object]:
    read_candidate_reviews: dict[str, object] = {
        str(candidate_spec["source_candidate_id"]): (
            _candidate_read_candidate_review_schema(candidate_spec=candidate_spec)
        )
        for candidate_spec in candidate_specs
    }
    return provider_output.RequestedFactAssessmentOutput.schema(
        {
            "read_candidate_reviews": _closed_object_schema(read_candidate_reviews),
            "canonical_inputs": _canonical_inputs_schema(canonical_specs),
        },
    )


def _candidate_read_candidate_review_schema(
    *,
    candidate_spec: dict[str, object],
) -> dict[str, object]:
    return {
        "oneOf": [
            _retained_read_review_schema(
                row_path_token_schema=_string_schema(
                    enum_values=_spec_values(candidate_spec, key="row_path_tokens"),
                ),
                field_token_schema=_string_schema(
                    enum_values=_spec_values(candidate_spec, key="field_tokens"),
                ),
            ),
            _dropped_read_review_schema(),
        ]
    }


def _retained_read_review_schema(
    *,
    row_path_token_schema: dict[str, object],
    field_token_schema: dict[str, object],
) -> dict[str, object]:
    return provider_output.RetainedReadReviewOutput.schema(
        {
            "relevant_row_path_tokens": {
                "type": "array",
                "items": row_path_token_schema,
            },
            "relevant_field_tokens": {
                "type": "array",
                "items": field_token_schema,
            },
            "retention_basis": {"type": "string", "minLength": 1},
            "retention_decision": {"enum": ["RETAIN"]},
        },
    )


def _dropped_read_review_schema() -> dict[str, object]:
    return provider_output.DroppedReadReviewOutput.schema(
        {
            "retention_basis": {"type": "string", "minLength": 1},
            "retention_decision": {"enum": ["DROP"]},
        }
    )


def _string_schema(*, enum_values: tuple[str, ...]) -> dict[str, object]:
    schema: dict[str, object] = {"type": "string", "minLength": 1}
    if enum_values:
        schema["enum"] = list(enum_values)
    return schema


def _closed_object_schema(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _canonical_inputs_schema(
    canonical_specs: tuple[dict[str, object], ...],
) -> dict[str, object]:
    canonical_inputs: dict[str, object] = {
        str(spec["known_input_id"]): _canonical_input_schema(spec)
        for spec in canonical_specs
    }
    return _closed_object_schema(canonical_inputs)


def _canonical_input_schema(spec: dict[str, object]) -> dict[str, object]:
    canonical_options = _canonical_option_specs(spec)
    canonical_option_ids = tuple(
        str(option["canonical_option_id"]) for option in canonical_options
    )
    return provider_output.CanonicalInputSelectionOutput.schema(
        {
            "interpretation_question": _string_schema(
                enum_values=(str(spec["interpretation_question"]),)
            ),
            "canonical_option_assessments": _closed_object_schema(
                {
                    option_id: {"type": "string", "minLength": 1}
                    for option_id in canonical_option_ids
                }
            ),
            "because": {"type": "string", "minLength": 1},
            "canonical_option_id": _string_schema(enum_values=canonical_option_ids),
        }
    )


def _canonical_option_specs(spec: dict[str, object]) -> tuple[dict[str, object], ...]:
    value = spec.get("canonical_options")
    values = value if isinstance(value, (list, tuple)) else ()
    return tuple(item for item in values if isinstance(item, dict))


def _spec_values(
    spec: dict[str, object],
    *,
    key: str,
) -> tuple[str, ...]:
    raw_values = spec.get(key)
    values = raw_values if isinstance(raw_values, (list, tuple)) else ()
    return tuple(str(value) for value in values if str(value))
