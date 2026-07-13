"""Scalar aggregate choices derived from bound-source fulfillment evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fervis.lookup.fact_planning.aggregate_choice_parts import (
    AGGREGATE_FUNCTIONS,
    COUNT_FUNCTION,
    aggregate_function_candidates,
    xml_attr,
    xml_text,
)
from fervis.lookup.fact_planning.executable_support import (
    parse_count_basis,
    count_metric_payload_for_evidence_item,
    unique_count_metric_payloads,
)
from fervis.lookup.fact_planning.fulfillment_evidence import (
    evidence_is_compatible_with_plan_shape,
    row_count_basis_evidence_ids,
    field_ids_by_answer_output_from_evidence,
)
from fervis.lookup.fact_plan.field_types import field_is_numeric
from fervis.lookup.answer_program.operations import AggregationFunction
from fervis.lookup.fact_planning.source_binding_basis import (
    metric_fit_bases_by_evidence_id,
)
from fervis.lookup.source_binding import BoundSource, SourceFulfillment
from fervis.lookup.fact_planning.provider_contract import AggregateScalarAnswerOutput
from fervis.lookup.fact_planning.compiled_patterns import CompiledMetric


def scalar_aggregate_choices_for_source(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
) -> dict[str, Any] | None:
    """Return compact scalar aggregate choices for one selected source."""

    metric_measure_field_ids_by_output = _metric_measure_field_ids_by_output(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
    )
    fields_by_id = {field.field_id: field for field in source.available_fields}
    basis_by_field_id = _metric_fit_basis_by_field_id(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
    )
    numeric_fields_by_output = {
        answer_output_id: tuple(
            field_id
            for field_id in field_ids
            if field_is_numeric(fields_by_id.get(field_id))
        )
        for answer_output_id, field_ids in metric_measure_field_ids_by_output.items()
    }
    metrics: list[dict[str, Any]] = []
    for answer_output_id in _answer_output_ids(
        source,
        requested_fact_id=requested_fact_id,
    ):
        numeric_field_ids = numeric_fields_by_output.get(answer_output_id, ())
        if numeric_field_ids:
            metrics.extend(
                _aggregate_field_metric_candidates(
                    answer_output_id=answer_output_id,
                    field_ids=numeric_field_ids,
                    fields_by_id=fields_by_id,
                    bases_by_field_id=basis_by_field_id,
                )
            )
            continue
        count_metrics = _count_metric_payloads(
            source,
            requested_fact_id=requested_fact_id,
            plan_shape=plan_shape,
            answer_output_id=answer_output_id,
            fields_by_id=fields_by_id,
        )
        for metric in count_metrics:
            metrics.append({**metric, "answer_output_id": answer_output_id})
    if any(metric.get("kind") == "aggregate_field" for metric in metrics):
        metrics = [
            metric for metric in metrics if metric.get("kind") == "aggregate_field"
        ]
    if not metrics:
        return None
    metric_candidates = tuple(
        {**metric, "id": f"metric_{index}"}
        for index, metric in enumerate(metrics, start=1)
    )
    return {
        "requested_fact_id": requested_fact_id,
        "source_binding_id": source.id,
        "read_id": source.source.read_id if source.source is not None else "",
        "plan_shape": plan_shape,
        "metric_candidates": metric_candidates,
        "function_candidates": _function_candidates(metric_candidates),
    }


def scalar_aggregate_choices_prompt(
    choices_by_requested_fact_id: dict[str, tuple[dict[str, Any], ...]],
) -> str:
    lines: list[str] = []
    for requested_fact_id, choices in choices_by_requested_fact_id.items():
        if not choices:
            continue
        lines.append(f'<fact id="{_xml(requested_fact_id)}">')
        for choice in choices:
            lines.extend(_choice_xml_lines(choice, indent="  "))
        lines.append("</fact>")
    return "\n".join(lines)


def selected_scalar_aggregate_metric(
    *,
    answer: AggregateScalarAnswerOutput,
    bound_sources: dict[str, BoundSource],
) -> CompiledMetric:
    source = bound_sources.get(answer.source_binding_id.strip())
    if source is None:
        raise ValueError("fact plan references unknown relation source binding")
    choice = scalar_aggregate_choices_for_source(
        source,
        requested_fact_id=answer.requested_fact_id.strip(),
        plan_shape=answer.pattern.strip(),
    )
    if choice is None:
        raise ValueError("aggregate_scalar has no legal metric choices")
    metric = _selected_candidate(
        answer.metric.id,
        choice.get("metric_candidates"),
        label="metric",
    )
    function = _selected_candidate(
        answer.function.id,
        choice.get("function_candidates"),
        label="function",
    )
    _validate_metric_selection(answer, metric)
    _validate_function_selection(answer, function)
    if str(function.get("value") or "") not in tuple(
        metric.get("allowed_functions") or ()
    ):
        raise ValueError("function selection is not allowed for selected metric")
    compiled = _compiled_metric({**metric, "function": str(function["value"])})
    return compiled


def _aggregate_field_metric_candidates(
    *,
    answer_output_id: str,
    field_ids: tuple[str, ...],
    fields_by_id: dict[str, Any],
    bases_by_field_id: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            **{
                "kind": "aggregate_field",
                "field_id": field_id,
                "type": str(getattr(fields_by_id.get(field_id), "type", "") or ""),
                "answer_output_id": answer_output_id,
                "allowed_functions": AGGREGATE_FUNCTIONS,
            },
            **(
                {"source_binding_basis": basis}
                if (basis := (bases_by_field_id or {}).get(field_id))
                else {}
            ),
        }
        for field_id in field_ids
    )


def _answer_output_ids(
    source: BoundSource,
    *,
    requested_fact_id: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            fulfillment.answer_output_id
            for fulfillment in source.fulfillments
            if fulfillment.requested_fact_id == requested_fact_id
        )
    )


def _count_metric_payloads(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
    answer_output_id: str,
    fields_by_id: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    explicit_count_basis_items = _count_basis_evidence_items_by_output(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
        answer_output_id=answer_output_id,
    )
    if explicit_count_basis_items:
        output: list[dict[str, Any]] = []
        for item in explicit_count_basis_items:
            metric = count_metric_payload_for_evidence_item(
                item,
            )
            if metric is not None:
                output.append(
                    {
                        **metric,
                        "allowed_functions": (COUNT_FUNCTION,),
                    }
                )
        return unique_count_metric_payloads(output)
    return ()


def metric_for_selection(
    *,
    answer: AggregateScalarAnswerOutput,
    bound_sources: dict[str, BoundSource],
) -> CompiledMetric:
    return selected_scalar_aggregate_metric(
        answer=answer,
        bound_sources=bound_sources,
    )


def _compiled_metric(metric: dict[str, Any]) -> CompiledMetric:
    if metric["kind"] == "count_records":
        count_basis = parse_count_basis(_metric_count_basis(metric))
        return CompiledMetric(
            field_id="",
            row_population_basis=count_basis.row_population,
            label="count",
            output_field_id="count",
            function=AggregationFunction.COUNT,
            answer_output_id=metric["answer_output_id"],
        )
    field_id = metric["field_id"]
    function = AggregationFunction(metric["function"])
    return CompiledMetric(
        field_id=field_id,
        row_population_basis=None,
        label=field_id,
        output_field_id=field_id,
        function=function,
        answer_output_id=metric["answer_output_id"],
    )


def _metric_count_basis(metric: dict[str, Any]) -> dict[str, Any]:
    basis = metric.get("count_basis")
    if not isinstance(basis, dict):
        raise ValueError("count_records metric requires count_basis")
    return basis


def _metric_measure_field_ids_by_output(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
) -> dict[str, tuple[str, ...]]:
    return _field_ids_by_output_from_evidence(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
        evidence_ids_by_fulfillment=lambda fulfillment: (
            fulfillment.metric_measure_evidence_ids
        ),
    )


def _metric_fit_basis_by_field_id(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
) -> dict[str, dict[str, str]]:
    basis_by_evidence_id = metric_fit_bases_by_evidence_id(
        tuple(
            fulfillment
            for fulfillment in source.fulfillments
            if fulfillment.requested_fact_id == requested_fact_id
        )
    )
    output: dict[str, dict[str, str]] = {}
    for evidence_id, basis in basis_by_evidence_id.items():
        field_ids = field_ids_by_answer_output_from_evidence(
            source,
            requested_fact_id=requested_fact_id,
            plan_shape=plan_shape,
            evidence_ids_by_fulfillment=lambda fulfillment: (
                (evidence_id,)
                if evidence_id in fulfillment.metric_measure_evidence_ids
                else ()
            ),
        )
        for field_id in {
            field_id
            for field_ids_for_output in field_ids.values()
            for field_id in field_ids_for_output
        }:
            output.setdefault(field_id, basis)
    return output


def _count_basis_evidence_items_by_output(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
    answer_output_id: str,
) -> tuple[Any, ...]:
    evidence_by_id = {item.evidence_id: item for item in source.evidence_items}
    output: list[Any] = []
    for fulfillment in source.fulfillments:
        if fulfillment.requested_fact_id != requested_fact_id:
            continue
        if fulfillment.answer_output_id != answer_output_id:
            continue
        for evidence_id in row_count_basis_evidence_ids(fulfillment):
            item = evidence_by_id.get(evidence_id)
            if item is None:
                continue
            if not evidence_is_compatible_with_plan_shape(
                item.row_cardinality,
                plan_shape=plan_shape,
            ):
                continue
            output.append(item)
    return tuple(output)


def _field_ids_by_output_from_evidence(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
    evidence_ids_by_fulfillment: Callable[[SourceFulfillment], tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    return field_ids_by_answer_output_from_evidence(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
        evidence_ids_by_fulfillment=evidence_ids_by_fulfillment,
    )


def _function_candidates(
    metric_candidates: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    return aggregate_function_candidates(metric_candidates)


def _choice_xml_lines(choice: dict[str, Any], *, indent: str) -> list[str]:
    source_attrs = " ".join(
        (
            f'id="{xml_attr(choice.get("source_binding_id"))}"',
            f'read="{xml_attr(choice.get("read_id"))}"',
        )
    )
    lines = [f"{indent}<source_binding {source_attrs}>"]
    lines.append(f'{indent}  <operation family="aggregate_scalar">')
    lines.append(f"{indent}    <metric_candidates>")
    for metric in choice.get("metric_candidates") or ():
        attrs = [
            f'id="{xml_attr(metric.get("id"))}"',
            f'kind="{xml_attr(metric.get("kind"))}"',
        ]
        if metric.get("field_id"):
            attrs.append(f'field="{xml_attr(metric.get("field_id"))}"')
        if metric.get("type"):
            attrs.append(f'type="{xml_attr(metric.get("type"))}"')
        if metric.get("allowed_functions"):
            allowed_functions = " ".join(
                str(item) for item in metric.get("allowed_functions") or ()
            )
            attrs.append(f'allowed_functions="{xml_attr(allowed_functions)}"')
        basis = _dict_or_empty(metric.get("source_binding_basis"))
        if not basis:
            lines.append(f"{indent}      <metric {' '.join(attrs)} />")
            continue
        lines.extend(
            (
                f"{indent}      <metric {' '.join(attrs)}>",
                f"{indent}        <source_binding_basis>",
                f"{indent}          <metric_meaning>{xml_text(_text(basis.get('metric_meaning')))}</metric_meaning>",
                f"{indent}          <fit_basis>{xml_text(_text(basis.get('fit_basis')))}</fit_basis>",
                f"{indent}        </source_binding_basis>",
                f"{indent}      </metric>",
            )
        )
    lines.append(f"{indent}    </metric_candidates>")
    lines.append(f"{indent}    <function_candidates>")
    for function in choice.get("function_candidates") or ():
        function_attrs = " ".join(
            (
                f'id="{xml_attr(function.get("id"))}"',
                f'value="{xml_attr(function.get("value"))}"',
                f'meaning="{xml_attr(function.get("meaning"))}"',
            )
        )
        lines.append(f"{indent}      <function {function_attrs} />")
    lines.append(f"{indent}    </function_candidates>")
    lines.append(f"{indent}  </operation>")
    lines.append(f"{indent}</source_binding>")
    return lines


def _selected_candidate(
    selected_id: str,
    candidates: Any,
    *,
    label: str,
) -> dict[str, Any]:
    for candidate in candidates or ():
        if str(candidate.get("id") or "") == selected_id:
            return candidate
    raise ValueError(f"unknown {label} selection")


def _validate_metric_selection(
    answer: AggregateScalarAnswerOutput,
    metric: dict[str, Any],
) -> None:
    if answer.metric.kind != str(metric.get("kind") or ""):
        raise ValueError("metric selection kind does not match selected id")
    if metric.get("field_id") and (answer.metric.field_id or "") != str(
        metric.get("field_id") or ""
    ):
        raise ValueError("metric selection field does not match selected id")


def _validate_function_selection(
    answer: AggregateScalarAnswerOutput,
    function: dict[str, Any],
) -> None:
    if answer.function.value != str(function.get("value") or ""):
        raise ValueError("function selection value does not match selected id")


def _xml(value: str) -> str:
    return xml_attr(value)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()
