"""Plan-selection specs for computed-scalar answers."""

from __future__ import annotations

from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
)


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "computed_scalar",
        ("value_1",),
        complete_answer_fulfillment_requirements=frozenset(),
    ),
    PlanSelectionShapeSpec(
        "computed_scalar",
        ("value_1", "value_2"),
        distinct_members=True,
        complete_answer_fulfillment_requirements=frozenset(),
    ),
)
