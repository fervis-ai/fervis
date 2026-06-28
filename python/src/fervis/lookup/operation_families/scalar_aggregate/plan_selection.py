"""Plan-selection specs for scalar-aggregate answers."""

from __future__ import annotations

from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
)


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "aggregate_scalar",
        ("metric",),
        row_population_grain_requirements=frozenset(("metric",)),
    ),
)
