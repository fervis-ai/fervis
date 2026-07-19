from fervis.lookup.source_binding.schema import build_source_binding_schema
from fervis.lookup.plan_selection.family_specs import SourceMemberConstraint
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingPlanFamily,
    SourceBindingTarget,
)
from fervis.model_io.backbone.dto import ToolSpec


def _candidate_population_bindings(
    *target_ids: str,
) -> dict[str, tuple[str, ...]]:
    return {
        target_id: (f"pop.{target_id.rsplit('.', 1)[-1]}.candidate_population",)
        for target_id in target_ids
    }


def source_binding_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="submit_source_binding",
        description="Submit source binding decisions.",
        strict=True,
        input_schema=build_source_binding_schema(
            target_param_decision_ids_by_param={
                "target.source_1": {
                    "start_date": (
                        "param_decision.source_1.start_date.bind.this_month",
                    ),
                },
                "target.source_2": {},
            },
            target_required_param_decision_ids={
                "target.source_1": ("start_date",),
                "target.source_2": (),
            },
            target_finite_choice_values={
                "target.source_1": {"status": ("DRAFT", "COMPLETED")},
                "target.source_2": {},
            },
            target_row_predicate_values={
                "target.source_1": {},
                "target.source_2": {},
            },
            target_finite_choice_test_ids={
                "target.source_1": {"status": ("subject_identity",)},
                "target.source_2": {},
            },
            target_finite_choice_normal_instance_test_ids={
                "target.source_1": {"status": ()},
                "target.source_2": {},
            },
            target_row_predicate_test_ids={
                "target.source_1": {},
                "target.source_2": {},
            },
            target_population_roles={
                "target.source_1": ({"role_id": "role_1"},),
                "target.source_2": ({"role_id": "role_1"},),
            },
            metric_evidence_ids_by_requested_fact={
                "fact_1": (
                    "source_1.root.amount",
                    "source_2.root.amount",
                )
            },
            target_requested_fact_ids={
                "target.source_1": "fact_1",
                "target.source_2": "fact_1",
            },
            target_fulfillment_support_set_ids_by_answer_output={
                "target.source_1": {
                    "answer_1": (
                        "support.source_1.answer_1."
                        "slot.source_1.answer_1.source_1.root.amount",
                    )
                },
                "target.source_2": {
                    "answer_1": (
                        "support.source_2.answer_1."
                        "slot.source_2.answer_1.source_2.root.amount",
                    )
                },
            },
            target_required_fulfillment_answer_output_ids={
                "target.source_1": ("answer_1",),
                "target.source_2": ("answer_1",),
            },
            target_population_binding_ids=_candidate_population_bindings(
                "target.source_1",
                "target.source_2",
            ),
            plan_families=(
                SourceBindingPlanFamily(
                    requested_fact_id="fact_1",
                    plan_shape="aggregate_scalar",
                    member_constraint=SourceMemberConstraint.ANY,
                    required_answer_output_ids=("answer_1",),
                    role_targets=(
                        (
                            "metric",
                            tuple(
                                SourceBindingTarget(
                                    binding_target_id=target_id,
                                    requested_fact_id="fact_1",
                                    plan_shape="aggregate_scalar",
                                    source_candidate_id=target_id.removeprefix(
                                        "target."
                                    ),
                                    requirement_id="metric",
                                    answer_output_ids=("answer_1",),
                                    required_answer_output_ids=("answer_1",),
                                )
                                for target_id in (
                                    "target.source_1",
                                    "target.source_2",
                                )
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
