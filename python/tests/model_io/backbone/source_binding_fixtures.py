from fervis.lookup.source_binding.schema import build_source_binding_schema
from fervis.model_io.backbone.dto import ToolSpec


def _candidate_population_bindings(
    *source_candidate_ids: str,
) -> dict[str, tuple[str, ...]]:
    return {
        source_candidate_id: (f"pop.{source_candidate_id}.candidate_population",)
        for source_candidate_id in source_candidate_ids
    }


def source_binding_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="submit_source_binding",
        description="Submit source binding decisions.",
        strict=True,
        input_schema=build_source_binding_schema(
            source_candidate_param_decision_ids_by_param={
                "source_1": {
                    "start_date": (
                        "param_decision.source_1.start_date.bind.this_month",
                    ),
                    "status": ("param_decision.source_1.status.bind.completed",),
                },
                "source_2": {},
            },
            source_candidate_required_param_ids={
                "source_1": ("start_date",),
                "source_2": (),
            },
            source_candidate_finite_choice_values={
                "source_1": {"status": ("DRAFT", "COMPLETED")},
                "source_2": {},
            },
            source_candidate_row_predicate_values={
                "source_1": {},
                "source_2": {},
            },
            source_candidate_membership_test_ids={
                "source_1": ("subject_identity",),
                "source_2": ("subject_identity",),
            },
            source_candidate_normal_instance_test_ids={
                "source_1": (),
                "source_2": (),
            },
            source_candidate_population_roles={
                "source_1": ({"role_id": "role_1"},),
                "source_2": ({"role_id": "role_1"},),
            },
            metric_evidence_ids_by_requested_fact={
                "fact_1": (
                    "source_1.root.amount",
                    "source_2.root.amount",
                )
            },
            source_candidate_requested_fact_ids={
                "source_1": "fact_1",
                "source_2": "fact_1",
            },
            source_candidate_fulfillment_support_set_ids_by_answer_output={
                "source_1": {
                    "answer_1": (
                        "support.source_1.answer_1."
                        "slot.source_1.answer_1.source_1.root.amount",
                    )
                },
                "source_2": {
                    "answer_1": (
                        "support.source_2.answer_1."
                        "slot.source_2.answer_1.source_2.root.amount",
                    )
                },
            },
            source_candidate_population_binding_ids=_candidate_population_bindings(
                "source_1",
                "source_2",
            ),
        ),
    )
