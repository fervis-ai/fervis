"""Plan-selection specs for list-row answers."""

from __future__ import annotations

from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
)


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "list_rows",
        ("primary",),
        intrinsic_source_requirements=frozenset(("primary",)),
    ),
    PlanSelectionShapeSpec(
        "grouped_rows",
        ("primary", "group_identity"),
        single_source=True,
        intrinsic_source_requirements=frozenset(("primary",)),
    ),
    PlanSelectionShapeSpec(
        "joined_rows",
        ("left", "right"),
        distinct_members=True,
        intrinsic_source_requirements=frozenset(("left", "right")),
    ),
)
