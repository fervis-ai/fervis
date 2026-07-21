"""One bounded scalar-value catalog for Fact Planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.fact_planning.compiled_patterns import CompiledMetric
from fervis.lookup.fact_planning.metric_options import (
    compiled_metric_for_choice,
    scalar_aggregate_choices_for_source,
)
from fervis.lookup.plan_selection import BoundPlanSelectionSet
from fervis.lookup.source_binding import BoundSource


@dataclass(frozen=True)
class SourceDerivedScalarValue:
    value_id: str
    requested_fact_id: str
    source_binding_id: str
    metric_id: str
    function_id: str
    metric: CompiledMetric
    payload: dict[str, Any]


def source_derived_scalar_values_by_fact(
    *,
    bound_sources: tuple[BoundSource, ...],
    plan_selection: BoundPlanSelectionSet,
) -> dict[str, tuple[SourceDerivedScalarValue, ...]]:
    """Project executable source aggregates as scalar value handles."""

    sources_by_id = {source.id: source for source in bound_sources}
    output: dict[str, tuple[SourceDerivedScalarValue, ...]] = {}
    for plan in plan_selection.plan_selections:
        if plan.plan_shape != "computed_scalar":
            continue
        values: list[SourceDerivedScalarValue] = []
        for source_binding_id in plan.source_binding_ids:
            source = sources_by_id.get(source_binding_id)
            if source is None or source.source is None:
                continue
            choice = scalar_aggregate_choices_for_source(
                source,
                requested_fact_id=plan.requested_fact_id,
                plan_shape=plan.plan_shape,
            )
            if choice is None:
                continue
            values.extend(_source_values(choice, requested_fact_id=plan.requested_fact_id))
        if values:
            output[plan.requested_fact_id] = tuple(values)
    return output


def _source_values(
    choice: dict[str, Any],
    *,
    requested_fact_id: str,
) -> tuple[SourceDerivedScalarValue, ...]:
    source_binding_id = str(choice.get("source_binding_id") or "")
    values: list[SourceDerivedScalarValue] = []
    for metric in choice.get("metric_candidates") or ():
        if not isinstance(metric, dict):
            continue
        metric_id = str(metric.get("id") or "")
        allowed_functions = set(metric.get("allowed_functions") or ())
        for function in choice.get("function_candidates") or ():
            if not isinstance(function, dict):
                continue
            function_id = str(function.get("id") or "")
            function_value = str(function.get("value") or "")
            if function_value not in allowed_functions:
                continue
            value_id = (
                f"source_scalar.{source_binding_id}.{metric_id}.{function_id}"
            )
            values.append(
                SourceDerivedScalarValue(
                    value_id=value_id,
                    requested_fact_id=requested_fact_id,
                    source_binding_id=source_binding_id,
                    metric_id=metric_id,
                    function_id=function_id,
                    metric=compiled_metric_for_choice(
                        choice,
                        metric_id=metric_id,
                        function_id=function_id,
                    ),
                    payload={
                        "value_id": value_id,
                        "kind": "source_derived_scalar",
                        "type": "number",
                        "source_binding_id": source_binding_id,
                        "metric": {
                            "kind": str(metric.get("kind") or ""),
                            **(
                                {"field_id": str(metric["field_id"])}
                                if metric.get("field_id")
                                else {}
                            ),
                            **(
                                {
                                    "source_binding_basis": dict(
                                        metric["source_binding_basis"]
                                    )
                                }
                                if isinstance(metric.get("source_binding_basis"), dict)
                                else {}
                            ),
                        },
                        "aggregation": {
                            "function": function_value,
                            "meaning": str(function.get("meaning") or ""),
                        },
                        "applies_to_requested_facts": [requested_fact_id],
                    },
                )
            )
    return tuple(values)
