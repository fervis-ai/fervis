"""Plan-selection specs for coverage-check answers."""

from __future__ import annotations

from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
)


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "set_difference",
        ("candidate_set", "observed_set"),
        distinct_members=True,
        answer_fulfillment_requirements=frozenset(("candidate_set",)),
    ),
)
