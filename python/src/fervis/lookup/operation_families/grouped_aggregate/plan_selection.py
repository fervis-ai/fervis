"""Plan-selection specs for grouped-aggregate answers."""

from __future__ import annotations

from fervis.lookup.plan_selection.family_specs import (
    PlanSelectionShapeSpec,
)
from fervis.lookup.operation_families.grouped_aggregate.support_sets import (
    grouped_aggregate_support_set_groups,
)


PLAN_SELECTION_SHAPES = (
    PlanSelectionShapeSpec(
        "aggregate_by_group",
        ("operation",),
        single_source=True,
        support_set_grouper=grouped_aggregate_support_set_groups,
    ),
)
