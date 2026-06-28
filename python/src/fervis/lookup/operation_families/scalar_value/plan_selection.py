"""Plan-selection specs for scalar-value answers."""

from __future__ import annotations

from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
)


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "direct_field_value",
        ("primary",),
        intrinsic_source_requirements=frozenset(("primary",)),
    ),
)
