"""Provider schema projection for model-facing fact plans."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.operation_families.fact_planning_schema_registry import (
    build_pattern_answer_schema,
)
from fervis.lookup.fact_planning.schema_helpers import (
    handle_schema as _handle_schema,
    non_empty_array_items as _non_empty_array_items,
    non_empty_string_array as _non_empty_string_array,
)
from fervis.lookup.fact_planning import provider_contract as provider_output
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFactBasis,
    MissingCatalogInputKind,
    PlanOutcomeKind,
)
from fervis.lookup.fact_planning.plan_shapes import (
    ALL_PLAN_SHAPES,
)


def build_fact_plan_schema(
    *,
    required_catalog_input_ids: tuple[str, ...] | None = None,
    required_catalog_choice_input_ids: tuple[str, ...] | None = None,
    requested_fact_ids: tuple[str, ...] | None = None,
    pattern_names: tuple[str, ...] = ALL_PLAN_SHAPES,
    require_pattern: bool = True,
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]] | None = None,
    identity_field_ids_by_source_binding_id: (
        Mapping[str, tuple[str, ...]] | None
    ) = None,
    selected_plan_shapes_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    grouped_ranked_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    scalar_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    rank_limit_value_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    plan_outcome_schema = _plan_outcome_schema(
        required_catalog_input_ids=required_catalog_input_ids,
        required_catalog_choice_input_ids=required_catalog_choice_input_ids,
        requested_fact_ids=requested_fact_ids,
        pattern_names=pattern_names,
        require_pattern=require_pattern,
        field_ids_by_source_binding_id=field_ids_by_source_binding_id,
        identity_field_ids_by_source_binding_id=(
            identity_field_ids_by_source_binding_id
        ),
        selected_plan_shapes_by_requested_fact_id=(
            selected_plan_shapes_by_requested_fact_id
        ),
        source_binding_ids_by_requested_fact_id=(
            source_binding_ids_by_requested_fact_id
        ),
        answer_output_ids_by_requested_fact_id=(answer_output_ids_by_requested_fact_id),
        answer_output_ids_by_source_binding_id=answer_output_ids_by_source_binding_id,
        source_binding_ids_by_requirement_by_requested_fact_id=(
            source_binding_ids_by_requirement_by_requested_fact_id
        ),
        grouped_ranked_choices_by_requested_fact_id=(
            grouped_ranked_choices_by_requested_fact_id
        ),
        scalar_aggregate_choices_by_requested_fact_id=(
            scalar_aggregate_choices_by_requested_fact_id
        ),
        rank_limit_value_ids=rank_limit_value_ids,
    )
    schema = provider_output.FactPlanOutput.schema(
        {
            "outcome": plan_outcome_schema,
        }
    )
    return {
        **schema,
        "modelSchemas": {
            "outcome": plan_outcome_schema,
        },
    }


def _plan_outcome_schema(
    *,
    required_catalog_input_ids: tuple[str, ...] | None,
    required_catalog_choice_input_ids: tuple[str, ...] | None,
    requested_fact_ids: tuple[str, ...] | None,
    pattern_names: tuple[str, ...],
    require_pattern: bool,
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]] | None,
    identity_field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]] | None,
    selected_plan_shapes_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    grouped_ranked_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    scalar_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    rank_limit_value_ids: tuple[str, ...] | None,
) -> dict[str, object]:
    answer_schema = _answer_schema(
        pattern_names=pattern_names,
        require_pattern=require_pattern,
        field_ids_by_source_binding_id=field_ids_by_source_binding_id,
        identity_field_ids_by_source_binding_id=(
            identity_field_ids_by_source_binding_id
        ),
        selected_plan_shapes_by_requested_fact_id=(
            selected_plan_shapes_by_requested_fact_id
        ),
        source_binding_ids_by_requested_fact_id=source_binding_ids_by_requested_fact_id,
        answer_output_ids_by_requested_fact_id=(answer_output_ids_by_requested_fact_id),
        answer_output_ids_by_source_binding_id=answer_output_ids_by_source_binding_id,
        source_binding_ids_by_requirement_by_requested_fact_id=(
            source_binding_ids_by_requirement_by_requested_fact_id
        ),
        grouped_ranked_choices_by_requested_fact_id=(
            grouped_ranked_choices_by_requested_fact_id
        ),
        scalar_aggregate_choices_by_requested_fact_id=(
            scalar_aggregate_choices_by_requested_fact_id
        ),
        rank_limit_value_ids=rank_limit_value_ids,
    )
    outcome_variants = []
    if answer_schema is not None:
        outcome_variants.append(answer_schema)
    outcome_variants.append(_impossible_schema(requested_fact_ids=requested_fact_ids))
    clarification_schema = _clarification_schema(
        required_catalog_input_ids=required_catalog_input_ids,
        required_catalog_choice_input_ids=required_catalog_choice_input_ids,
    )
    if clarification_schema is not None:
        outcome_variants.append(clarification_schema)
    return {"oneOf": outcome_variants}


def _answer_schema(
    *,
    pattern_names: tuple[str, ...],
    require_pattern: bool,
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]] | None,
    identity_field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]] | None,
    selected_plan_shapes_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    answer_output_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requirement_by_requested_fact_id: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ],
    grouped_ranked_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    scalar_aggregate_choices_by_requested_fact_id: Mapping[
        str, tuple[dict[str, object], ...]
    ],
    rank_limit_value_ids: tuple[str, ...] | None,
) -> dict[str, object] | None:
    pattern_schema = build_pattern_answer_schema(
        pattern_names=pattern_names,
        require_pattern=require_pattern,
        field_ids_by_source_binding_id=field_ids_by_source_binding_id,
        identity_field_ids_by_source_binding_id=(
            identity_field_ids_by_source_binding_id
        ),
        selected_plan_shapes_by_requested_fact_id=(
            selected_plan_shapes_by_requested_fact_id
        ),
        source_binding_ids_by_requested_fact_id=(
            source_binding_ids_by_requested_fact_id
        ),
        answer_output_ids_by_requested_fact_id=(answer_output_ids_by_requested_fact_id),
        answer_output_ids_by_source_binding_id=answer_output_ids_by_source_binding_id,
        source_binding_ids_by_requirement_by_requested_fact_id=(
            source_binding_ids_by_requirement_by_requested_fact_id
        ),
        grouped_ranked_choices_by_requested_fact_id=(
            grouped_ranked_choices_by_requested_fact_id
        ),
        scalar_aggregate_choices_by_requested_fact_id=(
            scalar_aggregate_choices_by_requested_fact_id
        ),
        rank_limit_value_ids=rank_limit_value_ids,
    )
    if pattern_schema is None:
        return None
    return provider_output.FactPlanAnswerOutput.schema(
        {
            "kind": {"enum": [PlanOutcomeKind.FACT_PLAN.value]},
            "answers": _non_empty_array_items(pattern_schema),
        },
    )


def _impossible_schema(
    *,
    requested_fact_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    requested_fact_id_schema = _handle_schema()
    if requested_fact_ids:
        requested_fact_id_schema = {"enum": list(requested_fact_ids)}
    blocked_fact_schema = provider_output.BlockedFactOutput.schema(
        {
            "requested_fact_id": requested_fact_id_schema,
            "basis": {
                "enum": [
                    BlockedFactBasis.CATALOG_ACCESS.value,
                    BlockedFactBasis.POLICY_ACCESS.value,
                ]
            },
            "evidence_refs": _non_empty_string_array(),
            "reviewed_read_ids": {
                "type": "array",
                "items": _handle_schema(),
            },
            "nearest_fields": {
                "type": "array",
                "items": provider_output.BlockedFactFieldOutput.schema(
                    {
                        "read_id": _handle_schema(),
                        "field_id": _handle_schema(),
                    },
                ),
            },
            "explanation": {"type": "string"},
        },
    )
    return provider_output.PlanImpossibleOutput.schema(
        {
            "kind": {"enum": [PlanOutcomeKind.IMPOSSIBLE.value]},
            "blocked_facts": _non_empty_array_items(blocked_fact_schema),
        },
    )


def _clarification_schema(
    *,
    required_catalog_input_ids: tuple[str, ...] | None,
    required_catalog_choice_input_ids: tuple[str, ...] | None,
) -> dict[str, object] | None:
    missing_catalog_input_variants = []
    missing_catalog_input_base = {
        "id": _handle_schema(),
        "requested_fact_id": _handle_schema(),
    }
    if required_catalog_input_ids is None or required_catalog_input_ids:
        missing_catalog_input_variants.append(
            provider_output.MissingCatalogRequiredInputOutput.schema(
                {
                    **missing_catalog_input_base,
                    "kind": {"enum": [MissingCatalogInputKind.REQUIRED_INPUT.value]},
                    "required_catalog_input_id": _clarification_id_schema(
                        required_catalog_input_ids
                    ),
                },
            )
        )
    if required_catalog_choice_input_ids is None or required_catalog_choice_input_ids:
        missing_catalog_input_variants.append(
            provider_output.MissingCatalogChoiceInputOutput.schema(
                {
                    **missing_catalog_input_base,
                    "kind": {"enum": [MissingCatalogInputKind.CHOICE_INPUT.value]},
                    "required_catalog_choice_input_id": _clarification_id_schema(
                        required_catalog_choice_input_ids
                    ),
                },
            )
        )
    if not missing_catalog_input_variants:
        return None
    return provider_output.PlanClarificationOutput.schema(
        {
            "kind": {"enum": [PlanOutcomeKind.NEEDS_CLARIFICATION.value]},
            "missing_catalog_inputs": _non_empty_array_items(
                {"oneOf": missing_catalog_input_variants}
            ),
        },
    )


def _clarification_id_schema(allowed_ids: tuple[str, ...] | None) -> dict[str, object]:
    if allowed_ids is None:
        return _handle_schema()
    return {"enum": list(allowed_ids)}
