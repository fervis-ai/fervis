"""Concrete plan-selection family registry."""

from __future__ import annotations

from functools import cache

from fervis.lookup.plan_selection.family_specs import PlanSelectionShapeSpec
from fervis.lookup.question_contract import (
    RequestedFactAnswerExpressionFamily,
)


def plan_selection_shape_specs_for_family(
    family: RequestedFactAnswerExpressionFamily,
) -> tuple[PlanSelectionShapeSpec, ...]:
    return _plan_shapes_by_family().get(family, ())


@cache
def _plan_shapes_by_family() -> dict[
    RequestedFactAnswerExpressionFamily, tuple[PlanSelectionShapeSpec, ...]
]:
    from fervis.lookup.operation_families.comparison_check.plan_selection import (
        PLAN_SELECTION_SHAPES as COMPARISON_CHECK_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.computed_scalar.plan_selection import (
        PLAN_SELECTION_SHAPES as COMPUTED_SCALAR_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.coverage_check.plan_selection import (
        PLAN_SELECTION_SHAPES as COVERAGE_CHECK_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.existence_check.plan_selection import (
        PLAN_SELECTION_SHAPES as EXISTENCE_CHECK_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.grouped_aggregate.plan_selection import (
        PLAN_SELECTION_SHAPES as GROUPED_AGGREGATE_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.list_rows.plan_selection import (
        PLAN_SELECTION_SHAPES as LIST_ROWS_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.scalar_aggregate.plan_selection import (
        PLAN_SELECTION_SHAPES as SCALAR_AGGREGATE_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.scalar_value.plan_selection import (
        PLAN_SELECTION_SHAPES as SCALAR_VALUE_PLAN_SELECTION_SHAPES,
    )
    from fervis.lookup.operation_families.set_difference.plan_selection import (
        PLAN_SELECTION_SHAPES as SET_DIFFERENCE_PLAN_SELECTION_SHAPES,
    )

    return {
        RequestedFactAnswerExpressionFamily.LIST_ROWS: LIST_ROWS_PLAN_SELECTION_SHAPES,
        RequestedFactAnswerExpressionFamily.SCALAR_VALUE: (
            SCALAR_VALUE_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE: (
            SCALAR_AGGREGATE_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE: (
            GROUPED_AGGREGATE_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.COMPUTED_SCALAR: (
            COMPUTED_SCALAR_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.SET_DIFFERENCE: (
            SET_DIFFERENCE_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.COVERAGE_CHECK: (
            COVERAGE_CHECK_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.EXISTENCE_CHECK: (
            EXISTENCE_CHECK_PLAN_SELECTION_SHAPES
        ),
        RequestedFactAnswerExpressionFamily.COMPARISON_CHECK: (
            COMPARISON_CHECK_PLAN_SELECTION_SHAPES
        ),
    }
