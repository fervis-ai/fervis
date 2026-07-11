from __future__ import annotations

from typing import Any

from fervis.lookup.fact_planning.schema import build_fact_plan_schema
from fervis.lookup.plan_selection.schema import build_plan_selection_schema
from fervis.lookup.read_eligibility.schema import build_read_eligibility_schema
from fervis.lookup.source_binding.schema import build_source_binding_schema


def test_lookup_provider_schemas_do_not_emit_internal_model_schemas_metadata():
    for schema in (
        build_read_eligibility_schema(candidate_reviews_by_requested_fact_id={}),
        build_plan_selection_schema(
            requested_fact_ids=("fact_1",),
            source_candidate_ids_by_requested_fact_id={"fact_1": ()},
        ),
        build_fact_plan_schema(
            selected_plan_shapes_by_requested_fact_id={},
            source_binding_ids_by_requested_fact_id={},
            answer_output_ids_by_requested_fact_id={},
            answer_output_ids_by_source_binding_id={},
            source_binding_ids_by_requirement_by_requested_fact_id={},
            grouped_ranked_choices_by_requested_fact_id={},
            scalar_aggregate_choices_by_requested_fact_id={},
        ),
        build_source_binding_schema(
            target_param_decision_ids_by_param={},
            target_finite_choice_values={},
            target_row_predicate_values={},
            target_finite_choice_test_ids={},
            target_finite_choice_normal_instance_test_ids={},
            target_row_predicate_test_ids={},
            target_population_roles={},
            target_requested_fact_ids={},
            metric_evidence_ids_by_requested_fact={},
            target_fulfillment_support_set_ids_by_answer_output={},
            target_required_fulfillment_answer_output_ids={},
            plan_families=(),
        ),
    ):
        assert not _contains_key(schema, "modelSchemas")


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_key(item, key) for item in value)
    return False
