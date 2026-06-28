"""Prompt sections for pattern fact planning."""

from __future__ import annotations

from fervis.lookup.fact_planning.grouped_ranked_choices import (
    GROUPED_RANKED_PLAN_SHAPES,
)
from fervis.lookup.turn_prompts import PromptSection, TurnPromptBuilder


def fact_plan_instruction_sections(
    builder: TurnPromptBuilder,
    *,
    tool_name: str,
    plan_shapes: frozenset[str],
) -> tuple[PromptSection, ...]:
    sections = [
        builder.instruction_block(
            "Decision Scope",
            (
                "Bound sources are already selected and argument-bound.",
                "Allowed answer patterns are constrained by the selected source alignment and bound source evidence.",
                "For each requested fact, use only answer patterns allowed by the output schema.",
                "When the schema allows multiple answer patterns, choose the one that matches the requested fact and bound evidence.",
                "Do not choose reads, endpoints, memory sources, source candidates, or query params.",
                "Do not add, remove, or change source params.",
            ),
        ),
    ]
    answer_identity_lines = [
        "requested_fact_id is the requested fact this answer satisfies.",
        "Copy requested_fact_id verbatim from Requested facts.",
    ]
    if plan_shapes_use_answer_output_ids(plan_shapes):
        answer_identity_lines.append(
            "For patterns that include answer_output_ids, copy answer_output_ids verbatim from Requested facts."
        )
    sections.append(
        builder.instruction_block(
            "Answer Identity",
            answer_identity_lines,
        )
    )
    source_selection_lines = []
    if plan_shapes_use_source_binding_id(plan_shapes):
        source_selection_lines.extend(
            (
                "Only output source_binding_id where the selected pattern schema includes it.",
                "When source_binding_id is included, copy its value verbatim from Bound sources.",
            )
        )
    if plan_shapes_use_grouped_ranked_choices(plan_shapes):
        source_selection_lines.append(
            "For aggregate_by_group and ranked_aggregate, copy source_binding_id from Grouped/ranked operation choices."
        )
    if plan_shapes_use_required_fulfillment_evidence(plan_shapes):
        source_selection_lines.append(
            "Required fulfillment evidence lists field evidence that must be used to produce the factual answer."
        )
    if source_selection_lines:
        sections.append(
            builder.instruction_block(
                "Source Selection",
                source_selection_lines,
            )
        )
    if plan_shapes_use_field_selection(plan_shapes):
        sections.append(
            builder.instruction_block(
                "Field Selection",
                (
                    "Only output field-selection properties where the schema includes them.",
                    "Choose operation fields from Bound sources.fields according to the selected pattern.",
                    "Copy field_id values verbatim from Bound sources.fields.",
                    "Do not rewrite, normalize, abbreviate, or invent field_id values.",
                    "label is an optional output name for a selected field.",
                ),
            )
        )
    if plan_shapes & _LIST_FIELD_PLAN_SHAPES:
        sections.append(
            builder.instruction_block(
                "List And Field Patterns",
                (
                    "For list_rows, output_fields are the raw fields to show.",
                    "For grouped_rows, group_fields define each group and output_fields are the raw fields to show inside each group.",
                    "For direct_field_value, output_field is the single direct field value to return from the bound source.",
                ),
            )
        )
    if plan_shapes & _METRIC_PLAN_SHAPES:
        metric_lines = []
        if "aggregate_scalar" in plan_shapes:
            metric_lines.extend(
                (
                    "For aggregate_scalar, choose metric and function from Scalar aggregate operation choices.",
                    "For each selected scalar aggregate part, write selection_basis before copying the selected id and field/value.",
                )
            )
        if plan_shapes_use_grouped_ranked_choices(plan_shapes):
            metric_lines.extend(
                (
                    "For aggregate_by_group and ranked_aggregate, choose group, metric, function, and rank from Grouped/ranked operation choices.",
                    "For each selected grouped/ranked part, write selection_basis before copying the selected id and field/value.",
                    "Do not output separate group fields, answer fields, or source evidence for grouped/ranked patterns.",
                )
            )
        sections.append(
            builder.instruction_block(
                "Metric Patterns",
                metric_lines,
            )
        )
    if plan_shapes_use_grouped_ranked_choices(plan_shapes):
        sections.append(
            builder.instruction_block(
                "Grouped Metric Patterns",
                (
                    "For aggregate_by_group, group defines the answer grouping, metric defines the measured value, and function defines the aggregate computation.",
                    "For ranked_aggregate, group defines the answer grouping, metric and function define the computed ranking value, and rank defines which groups to return.",
                    "rank.sort=desc returns larger metric values before smaller metric values.",
                    "rank.sort=asc returns smaller metric values before larger metric values.",
                    "rank.limit is the number of ranked groups to return.",
                    "rank.limit_value_id is optional. Include it only when Operation input values contains a value_id for that exact rank limit.",
                ),
            )
        )
    if "computed_scalar" in plan_shapes:
        sections.append(
            builder.instruction_block(
                "Computed Scalar",
                (
                    "Use computed_scalar only with bound value sources.",
                    "scalar_inputs are the bound value sources used by computed_scalar.",
                    "Copy source_binding_id values verbatim from Bound sources where kind is value.",
                    "expression is arithmetic using scalar_inputs input_id names.",
                    "output.scalar_id is the ID for the computed scalar result.",
                    "output.label is an optional output name.",
                ),
            )
        )
    if "set_difference" in plan_shapes:
        sections.append(
            builder.instruction_block(
                "Set Difference",
                (
                    "candidate is the source containing possible rows.",
                    "observed is the source containing rows already seen or present.",
                    "candidate.identity_fields and observed.identity_fields are the fields used to compare rows.",
                    "candidate.output_fields are the fields to show for missing rows.",
                ),
            )
        )
    if "joined_rows" in plan_shapes:
        sections.append(
            builder.instruction_block(
                "Joined Rows",
                (
                    "left and right are the two bound sources being joined.",
                    "left.fields and right.fields are the fields used from each side.",
                    "join_keys map left_field_id to right_field_id.",
                    "output_fields lists which joined fields to show.",
                    "Use joined_rows only when the selected pattern is joined_rows.",
                ),
            )
        )
    sections.extend(
        (
            builder.instruction_block(
                "Copying And Validity",
                (
                    "When a field needs an existing identifier, copy that identifier verbatim from the prompt JSON or from an object you already created in this tool call.",
                    "Do not rewrite, normalize, abbreviate, or invent identifiers.",
                    "Do not invent fields, sources, IDs, filters, values, joins, metrics, or calculations.",
                    "Do not choose operation details that are not allowed by the selected plan shape.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (f"Return the {tool_name} tool call only.",),
            ),
        )
    )
    return tuple(sections)


_LIST_FIELD_PLAN_SHAPES = frozenset({"list_rows", "grouped_rows", "direct_field_value"})
_METRIC_PLAN_SHAPES = frozenset(
    {"aggregate_scalar", "aggregate_by_group", "ranked_aggregate"}
)
_SOURCE_BINDING_ID_PLAN_SHAPES = frozenset(
    {
        "list_rows",
        "grouped_rows",
        "direct_field_value",
        "aggregate_scalar",
        "set_difference",
        "joined_rows",
    }
)
_FIELD_SELECTION_PLAN_SHAPES = frozenset(
    {
        "list_rows",
        "grouped_rows",
        "direct_field_value",
        "set_difference",
        "joined_rows",
    }
)


def plan_shapes_use_grouped_ranked_choices(plan_shapes: frozenset[str]) -> bool:
    return bool(plan_shapes & GROUPED_RANKED_PLAN_SHAPES)


def plan_shapes_use_answer_output_ids(plan_shapes: frozenset[str]) -> bool:
    return bool(plan_shapes - GROUPED_RANKED_PLAN_SHAPES)


def plan_shapes_use_source_binding_id(plan_shapes: frozenset[str]) -> bool:
    return bool(
        plan_shapes & (_SOURCE_BINDING_ID_PLAN_SHAPES | GROUPED_RANKED_PLAN_SHAPES)
    )


def plan_shapes_use_required_fulfillment_evidence(
    plan_shapes: frozenset[str],
) -> bool:
    return bool(plan_shapes - GROUPED_RANKED_PLAN_SHAPES)


def plan_shapes_use_field_selection(plan_shapes: frozenset[str]) -> bool:
    return bool(plan_shapes & _FIELD_SELECTION_PLAN_SHAPES)
