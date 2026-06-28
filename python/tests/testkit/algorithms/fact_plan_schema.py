from __future__ import annotations

from fervis.lookup.fact_planning.schema import build_fact_plan_schema


def run_fact_plan_schema_case(payload: dict) -> list[str]:
    input_payload = payload["input"]
    schema_text = str(
        build_fact_plan_schema(
            requested_fact_ids=tuple(input_payload["requested_fact_ids"]),
            pattern_names=tuple(input_payload["pattern_names"]),
            selected_plan_shapes_by_requested_fact_id=dict(
                input_payload["selected_plan_shapes_by_requested_fact_id"]
            ),
            source_binding_ids_by_requested_fact_id={
                key: tuple(value)
                for key, value in input_payload[
                    "source_binding_ids_by_requested_fact_id"
                ].items()
            },
            answer_output_ids_by_requested_fact_id={
                key: tuple(value)
                for key, value in input_payload[
                    "answer_output_ids_by_requested_fact_id"
                ].items()
            },
            answer_output_ids_by_source_binding_id={
                key: tuple(value)
                for key, value in input_payload[
                    "answer_output_ids_by_source_binding_id"
                ].items()
            },
            source_binding_ids_by_requirement_by_requested_fact_id=dict(
                input_payload.get(
                    "source_binding_ids_by_requirement_by_requested_fact_id",
                    {},
                )
            ),
            grouped_ranked_choices_by_requested_fact_id=dict(
                input_payload.get("grouped_ranked_choices_by_requested_fact_id", {})
            ),
            scalar_aggregate_choices_by_requested_fact_id=dict(
                input_payload.get("scalar_aggregate_choices_by_requested_fact_id", {})
            ),
            field_ids_by_source_binding_id={
                key: tuple(value)
                for key, value in input_payload[
                    "field_ids_by_source_binding_id"
                ].items()
            },
        )
    )
    return [
        f"schema unexpectedly contains {item!r}"
        for item in payload["expect"]["text_excludes"]
        if item in schema_text
    ]
