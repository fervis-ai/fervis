"""Physical plan-selection specs for ranked answers."""

from __future__ import annotations

from fervis.lookup.operation_families.grouped_ranked.plan_selection import (
    grouped_ranked_support_set_groups,
)
from fervis.lookup.plan_selection.family_specs import PlanSelectionShapeSpec


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "ranked_rows",
        ("primary",),
        intrinsic_source_requirements=frozenset(("primary",)),
    ),
    PlanSelectionShapeSpec(
        "ranked_aggregate",
        ("operation",),
        single_source=True,
        support_set_grouper=grouped_ranked_support_set_groups,
    ),
)
