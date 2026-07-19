"""Fact-planning choices for grouped aggregate operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

from fervis.lookup.fact_planning.aggregate_choice_parts import (
    AGGREGATE_FUNCTIONS,
    COUNT_FUNCTION,
    aggregate_function_candidates,
    xml_attr,
    xml_text,
)
from fervis.lookup.fact_planning.executable_support import (
    parse_count_basis,
    count_basis_matches_evidence_item,
    count_basis_meaning,
    count_metric_payload_for_evidence_item,
    unique_count_metric_payloads,
)
from fervis.lookup.fact_plan.field_types import field_is_numeric
from fervis.lookup.fact_planning.fulfillment_evidence import (
    evidence_is_compatible_with_plan_shape,
    field_id_for_fulfillment_evidence,
    source_cardinality_by_evidence_id,
    source_field_id_by_evidence_id,
)
from fervis.lookup.answer_program.operations import AggregationFunction
from fervis.lookup.fact_planning.source_binding_basis import (
    attach_metric_fit_basis,
    metric_fit_bases_by_evidence_id,
)
from fervis.lookup.fact_planning.provider_contract import (
    GroupedAggregateAnswerOutput,
)
from fervis.lookup.fact_planning.compiled_patterns import CompiledMetric
from fervis.lookup.source_binding import (
    BoundSource,
    SourceField,
    SourceFulfillment,
    entity_evidence_entity_kind,
    entity_evidence_key_id,
)


GROUPED_AGGREGATE_PLAN_SHAPES = frozenset({"aggregate_by_group"})


@dataclass(frozen=True)
class GroupedAggregateResultOutput:
    answer_output_id: str
    role: str
    field_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()
    key_id: str = ""
    entity_kind: str = ""
    entity_components: tuple[tuple[str, str], ...] = ()

    @property
    def field_id(self) -> str:
        return self.field_ids[0] if len(self.field_ids) == 1 else ""

    @property
    def evidence_id(self) -> str:
        return self.evidence_ids[0] if len(self.evidence_ids) == 1 else ""


@dataclass(frozen=True)
class GroupedAggregateSelection:
    source_binding_id: str
    fulfills_answer_output_ids: tuple[str, ...]
    group_field_ids: tuple[str, ...]
    group_entity_key_id: str
    group_entity_kind: str
    group_entity_components: tuple[tuple[str, str], ...]
    metric: CompiledMetric
    answer_outputs: tuple[GroupedAggregateResultOutput, ...]


@dataclass(frozen=True)
class _GroupEvidence:
    evidence_ids: tuple[str, ...]
    key_id: str = ""
    entity_kind: str = ""
    components: tuple[tuple[str, str], ...] = ()


class _GroupComponentPayload(TypedDict):
    component_id: str
    field_id: str


class _GroupCandidatePayload(TypedDict):
    id: str
    field_ids: list[str]
    types: list[str]
    evidence_id: str
    field_evidence_ids: list[str]
    key_id: str
    entity_kind: str
    components: list[_GroupComponentPayload]


@dataclass(frozen=True)
class _GroupCandidate:
    field_ids: tuple[str, ...]
    types: tuple[str, ...]
    evidence_id: str
    field_evidence_ids: tuple[str, ...]
    key_id: str = ""
    entity_kind: str = ""
    components: tuple[tuple[str, str], ...] = ()

    def payload(self, *, candidate_id: str) -> _GroupCandidatePayload:
        return _GroupCandidatePayload(
            id=candidate_id,
            field_ids=list(self.field_ids),
            types=list(self.types),
            evidence_id=self.evidence_id,
            field_evidence_ids=list(self.field_evidence_ids),
            key_id=self.key_id,
            entity_kind=self.entity_kind,
            components=[
                _GroupComponentPayload(component_id=component_id, field_id=field_id)
                for component_id, field_id in self.components
            ],
        )


def grouped_aggregate_choice_payload(
    sources: tuple[BoundSource, ...],
    *,
    requested_fact_id: str,
    plan_shape: str,
    allowed_source_binding_ids: tuple[str, ...] = (),
) -> tuple[dict[str, Any], ...]:
    if plan_shape not in GROUPED_AGGREGATE_PLAN_SHAPES:
        return ()
    allowed = set(allowed_source_binding_ids)
    return tuple(
        payload
        for source in sources
        if not allowed or source.id in allowed
        for payload in (
            _choice_payload_for_source(
                source,
                requested_fact_id=requested_fact_id,
                plan_shape=plan_shape,
            ),
        )
        if payload is not None
    )


def grouped_aggregate_choices_prompt(
    choices_by_requested_fact_id: Mapping[str, tuple[dict[str, Any], ...]],
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


def grouped_aggregate_choices_by_requested_fact_id(
    sources: tuple[BoundSource, ...],
    *,
    selected_plan_shapes_by_requested_fact_id: Mapping[str, tuple[str, ...]],
    source_binding_ids_by_requested_fact_id: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for (
        requested_fact_id,
        plan_shapes,
    ) in selected_plan_shapes_by_requested_fact_id.items():
        fact_choices: list[dict[str, Any]] = []
        for plan_shape in plan_shapes:
            if plan_shape not in GROUPED_AGGREGATE_PLAN_SHAPES:
                continue
            fact_choices.extend(
                grouped_aggregate_choice_payload(
                    sources,
                    requested_fact_id=requested_fact_id,
                    plan_shape=plan_shape,
                    allowed_source_binding_ids=source_binding_ids_by_requested_fact_id.get(
                        requested_fact_id,
                        (),
                    ),
                )
            )
        if fact_choices:
            output[requested_fact_id] = tuple(fact_choices)
    return output


def selected_grouped_aggregate_operation(
    answer: GroupedAggregateAnswerOutput,
    *,
    bound_sources: dict[str, BoundSource],
) -> GroupedAggregateSelection:
    requested_fact_id = answer.requested_fact_id
    plan_shape = answer.pattern
    source_binding_id = answer.source_binding_id
    source = bound_sources.get(source_binding_id)
    if source is None:
        raise ValueError("fact plan references unknown relation source binding")
    choice = _choice_payload_for_source(
        source,
        requested_fact_id=requested_fact_id,
        plan_shape=plan_shape,
    )
    if choice is None:
        raise ValueError("fact plan references unavailable grouped aggregate choices")
    group = _parse_group_candidate(_dict(choice.get("group")))
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
    if _text(function.get("value")) not in tuple(metric.get("allowed_functions") or ()):
        raise ValueError("function selection is not allowed for selected metric")
    answer_outputs = _answer_outputs_for_selection(
        source,
        requested_fact_id=requested_fact_id,
        group_candidate=group,
        metric_candidate=metric,
        plan_shape=plan_shape,
    )
    if not answer_outputs:
        raise ValueError("grouped aggregate selection produces no answer outputs")
    return GroupedAggregateSelection(
        source_binding_id=source_binding_id,
        fulfills_answer_output_ids=tuple(
            dict.fromkeys(item.answer_output_id for item in answer_outputs)
        ),
        group_field_ids=group.field_ids,
        group_entity_key_id=group.key_id,
        group_entity_kind=group.entity_kind,
        group_entity_components=group.components,
        metric=_compiled_metric(
            metric,
            function,
            answer_output_id=_metric_answer_output_id(metric, answer_outputs),
        ),
        answer_outputs=answer_outputs,
    )


def _choice_payload_for_source(
    source: BoundSource,
    *,
    requested_fact_id: str,
    plan_shape: str,
) -> dict[str, Any] | None:
    fulfillments = tuple(
        fulfillment
        for fulfillment in source.fulfillments
        if fulfillment.requested_fact_id == requested_fact_id
    )
    if not fulfillments:
        return None
    group_candidate = _backend_owned_group(
        source,
        fulfillments=fulfillments,
        plan_shape=plan_shape,
    )
    metrics = _metric_candidates(
        source,
        fulfillments=fulfillments,
        plan_shape=plan_shape,
    )
    if group_candidate is None or not metrics:
        return None
    group = group_candidate.payload(candidate_id="group_1")
    functions = _function_candidates(metrics)
    return {
        "requested_fact_id": requested_fact_id,
        "source_binding_id": source.id,
        "read_id": source.source.read_id if source.source is not None else "",
        "plan_shape": plan_shape,
        "group": group,
        "metric_candidates": metrics,
        "function_candidates": functions,
    }


def _group_candidates(
    source: BoundSource,
    *,
    fulfillments: tuple[SourceFulfillment, ...],
    plan_shape: str,
) -> tuple[_GroupCandidate, ...]:
    fields_by_id = {field.field_id: field for field in source.available_fields}
    entity_evidence = tuple(
        dict.fromkeys(
            fulfillment.entity_evidence
            for fulfillment in fulfillments
            if fulfillment.entity_evidence is not None
        )
    )
    output: list[_GroupCandidate] = []
    for evidence in entity_evidence:
        resolved_field_ids = tuple(
            _field_id_for_evidence_id(
                source,
                component.field_evidence_id,
                plan_shape=plan_shape,
            )
            for component in evidence.components
        )
        declared_field_ids = tuple(
            component.field_id for component in evidence.components
        )
        if resolved_field_ids != declared_field_ids:
            continue
        output.append(
            _GroupCandidate(
                field_ids=declared_field_ids,
                types=_source_field_types(
                    declared_field_ids,
                    fields_by_id=fields_by_id,
                ),
                evidence_id=evidence.evidence_id,
                field_evidence_ids=tuple(
                    component.field_evidence_id for component in evidence.components
                ),
                key_id=entity_evidence_key_id(evidence),
                entity_kind=entity_evidence_entity_kind(evidence),
                components=tuple(
                    (component.component_id, component.field_id)
                    for component in evidence.components
                ),
            )
        )
    for fulfillment in fulfillments:
        for evidence_id in fulfillment.value_evidence_ids:
            field_id = _field_id_for_evidence_id(
                source,
                evidence_id,
                plan_shape=plan_shape,
            )
            if not field_id:
                continue
            output.append(
                _GroupCandidate(
                    field_ids=(field_id,),
                    types=_source_field_types((field_id,), fields_by_id=fields_by_id),
                    evidence_id=evidence_id,
                    field_evidence_ids=(evidence_id,),
                )
            )
    return _unique_group_candidates(output)


def _source_field_types(
    field_ids: tuple[str, ...],
    *,
    fields_by_id: Mapping[str, SourceField],
) -> tuple[str, ...]:
    return tuple(
        fields_by_id[field_id].type if field_id in fields_by_id else ""
        for field_id in field_ids
    )


def _backend_owned_group(
    source: BoundSource,
    *,
    fulfillments: tuple[SourceFulfillment, ...],
    plan_shape: str,
) -> _GroupCandidate | None:
    groups = _group_candidates(
        source,
        fulfillments=fulfillments,
        plan_shape=plan_shape,
    )
    return groups[0] if len(groups) == 1 else None


def _metric_candidates(
    source: BoundSource,
    *,
    fulfillments: tuple[SourceFulfillment, ...],
    plan_shape: str,
) -> tuple[dict[str, Any], ...]:
    fields_by_id = {field.field_id: field for field in source.available_fields}
    bases_by_evidence_id = metric_fit_bases_by_evidence_id(fulfillments)
    metrics: list[dict[str, Any]] = []
    for field_id, evidence_id in _metric_field_ids(source, fulfillments, plan_shape):
        field = fields_by_id.get(field_id)
        if not field_is_numeric(field):
            continue
        metrics.append(
            attach_metric_fit_basis(
                {
                    "kind": "aggregate_field",
                    "field_id": field_id,
                    "type": str(getattr(field, "type", "") or ""),
                    "evidence_id": evidence_id,
                    "allowed_functions": AGGREGATE_FUNCTIONS,
                },
                evidence_id=evidence_id,
                bases_by_evidence_id=bases_by_evidence_id,
            )
        )
    metrics.extend(
        _count_record_metric_payloads(
            source,
            fulfillments=fulfillments,
            plan_shape=plan_shape,
        )
    )
    return tuple(
        {**metric, "id": f"metric_{index}"}
        for index, metric in enumerate(metrics, start=1)
    )


def _metric_field_ids(
    source: BoundSource,
    fulfillments: tuple[SourceFulfillment, ...],
    plan_shape: str,
) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fulfillment in fulfillments:
        for evidence_id in fulfillment.metric_measure_evidence_ids:
            field_id = _field_id_for_evidence_id(
                source,
                evidence_id,
                plan_shape=plan_shape,
            )
            if not field_id or field_id in seen:
                continue
            seen.add(field_id)
            output.append((field_id, evidence_id))
    return tuple(output)


def _count_record_metric_payloads(
    source: BoundSource,
    *,
    fulfillments: tuple[SourceFulfillment, ...],
    plan_shape: str,
) -> tuple[dict[str, Any], ...]:
    evidence_by_id = {item.evidence_id: item for item in source.evidence_items}
    metrics: list[dict[str, Any]] = []
    for fulfillment in fulfillments:
        for evidence_id in fulfillment.row_count_basis_evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                continue
            if not evidence_is_compatible_with_plan_shape(
                item.row_cardinality,
                plan_shape=plan_shape,
            ):
                continue
            metric = count_metric_payload_for_evidence_item(
                item,
            )
            if metric is None:
                continue
            metrics.append(
                {
                    **metric,
                    "evidence_id": evidence_id,
                    "allowed_functions": (COUNT_FUNCTION,),
                }
            )
    return unique_count_metric_payloads(metrics)


def _function_candidates(
    metric_candidates: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    return aggregate_function_candidates(metric_candidates)


def _compiled_metric(
    metric: Mapping[str, object],
    function: Mapping[str, object],
    *,
    answer_output_id: str,
) -> CompiledMetric:
    if metric.get("kind") == "count_records":
        count_basis = parse_count_basis(_dict(metric.get("count_basis")))
        return CompiledMetric(
            field_id="",
            row_population_basis=count_basis.row_population,
            label="count",
            output_field_id="count",
            function=AggregationFunction.COUNT,
            answer_output_id=answer_output_id,
        )
    field_id = _text(metric.get("field_id"))
    return CompiledMetric(
        field_id=field_id,
        row_population_basis=None,
        label=field_id,
        output_field_id=field_id,
        function=AggregationFunction(_text(function.get("value"))),
        answer_output_id=answer_output_id,
    )


def _metric_answer_output_id(
    metric: Mapping[str, object],
    answer_outputs: tuple[GroupedAggregateResultOutput, ...],
) -> str:
    expected_role = (
        "ROW_COUNT"
        if _text(metric.get("kind")) == "count_records"
        else "MEASURED_VALUE"
    )
    evidence_id = _text(metric.get("evidence_id"))
    if not evidence_id:
        return ""
    matches = tuple(
        answer_output.answer_output_id
        for answer_output in answer_outputs
        if answer_output.role == expected_role
        and answer_output.evidence_id == evidence_id
    )
    return matches[0] if len(matches) == 1 else ""


def _answer_outputs_for_selection(
    source: BoundSource,
    *,
    requested_fact_id: str,
    group_candidate: _GroupCandidate,
    metric_candidate: Mapping[str, Any],
    plan_shape: str,
) -> tuple[GroupedAggregateResultOutput, ...]:
    group_field_ids = group_candidate.field_ids
    metric_field_id = _text(metric_candidate.get("field_id"))
    count_basis = _dict_or_empty(metric_candidate.get("count_basis"))
    output: list[GroupedAggregateResultOutput] = []
    for fulfillment in (
        item
        for item in source.fulfillments
        if item.requested_fact_id == requested_fact_id
    ):
        group_evidence = _matching_group_evidence(
            source,
            fulfillment,
            group_field_ids=group_field_ids,
            plan_shape=plan_shape,
        )
        if group_evidence is not None:
            output.append(
                GroupedAggregateResultOutput(
                    answer_output_id=fulfillment.answer_output_id,
                    role="GROUP_KEY",
                    field_ids=group_field_ids,
                    evidence_ids=group_evidence.evidence_ids,
                    key_id=group_evidence.key_id,
                    entity_kind=group_evidence.entity_kind,
                    entity_components=group_evidence.components,
                )
            )
            continue
        metric_evidence_id = _first_matching_metric_evidence_id(
            source,
            fulfillment,
            metric_field_id=metric_field_id,
            plan_shape=plan_shape,
        )
        if metric_evidence_id:
            output.append(
                GroupedAggregateResultOutput(
                    answer_output_id=fulfillment.answer_output_id,
                    role="MEASURED_VALUE",
                    field_ids=(metric_field_id,),
                    evidence_ids=(metric_evidence_id,),
                )
            )
            continue
        count_evidence_id = _first_matching_count_evidence_id(
            source,
            fulfillment,
            count_basis=count_basis,
            plan_shape=plan_shape,
        )
        if count_evidence_id:
            output.append(
                GroupedAggregateResultOutput(
                    answer_output_id=fulfillment.answer_output_id,
                    role="ROW_COUNT",
                    field_ids=("count",),
                    evidence_ids=(count_evidence_id,),
                )
            )
    return tuple(output)


def _matching_group_evidence(
    source: BoundSource,
    fulfillment: SourceFulfillment,
    *,
    group_field_ids: tuple[str, ...],
    plan_shape: str,
) -> _GroupEvidence | None:
    entity_evidence = _matching_entity_group_evidence(
        fulfillment,
        group_field_ids=group_field_ids,
    )
    scalar_evidence = None
    if entity_evidence is None:
        scalar_evidence = _matching_scalar_group_evidence(
            source,
            fulfillment,
            group_field_ids=group_field_ids,
            plan_shape=plan_shape,
        )
    return entity_evidence or scalar_evidence


def _matching_entity_group_evidence(
    fulfillment: SourceFulfillment,
    *,
    group_field_ids: tuple[str, ...],
) -> _GroupEvidence | None:
    match = None
    entity_evidence = fulfillment.entity_evidence
    if entity_evidence is not None:
        field_ids = tuple(
            component.field_id for component in entity_evidence.components
        )
        if field_ids == group_field_ids:
            match = _GroupEvidence(
                evidence_ids=tuple(
                    component.field_evidence_id
                    for component in entity_evidence.components
                ),
                key_id=entity_evidence_key_id(entity_evidence),
                entity_kind=entity_evidence_entity_kind(entity_evidence),
                components=tuple(
                    (component.component_id, component.field_id)
                    for component in entity_evidence.components
                ),
            )
    return match


def _matching_scalar_group_evidence(
    source: BoundSource,
    fulfillment: SourceFulfillment,
    *,
    group_field_ids: tuple[str, ...],
    plan_shape: str,
) -> _GroupEvidence | None:
    match = None
    if len(group_field_ids) == 1:
        matching_evidence_id = next(
            (
                evidence_id
                for evidence_id in fulfillment.value_evidence_ids
                if (
                    _field_id_for_evidence_id(
                        source,
                        evidence_id,
                        plan_shape=plan_shape,
                    )
                    == group_field_ids[0]
                )
            ),
            "",
        )
        if matching_evidence_id:
            match = _GroupEvidence(evidence_ids=(matching_evidence_id,))
    return match


def _first_matching_metric_evidence_id(
    source: BoundSource,
    fulfillment: SourceFulfillment,
    *,
    metric_field_id: str,
    plan_shape: str,
) -> str:
    for evidence_id in fulfillment.metric_measure_evidence_ids:
        if (
            _field_id_for_evidence_id(source, evidence_id, plan_shape=plan_shape)
            == metric_field_id
        ):
            return evidence_id
    return ""


def _first_matching_count_evidence_id(
    source: BoundSource,
    fulfillment: SourceFulfillment,
    *,
    count_basis: Mapping[str, Any],
    plan_shape: str,
) -> str:
    if not count_basis:
        return ""
    evidence_by_id = {item.evidence_id: item for item in source.evidence_items}
    for evidence_id in fulfillment.row_count_basis_evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        if not evidence_is_compatible_with_plan_shape(
            item.row_cardinality,
            plan_shape=plan_shape,
        ):
            continue
        if count_basis_matches_evidence_item(count_basis, item):
            return evidence_id
    return ""


def _field_id_for_evidence_id(
    source: BoundSource,
    evidence_id: str,
    *,
    plan_shape: str,
) -> str:
    cardinality = source_cardinality_by_evidence_id(source).get(evidence_id, "")
    if not evidence_is_compatible_with_plan_shape(cardinality, plan_shape=plan_shape):
        return ""
    return field_id_for_fulfillment_evidence(
        evidence_id,
        field_id_by_evidence_id=source_field_id_by_evidence_id(source),
        available_field_ids=set(source.available_field_ids),
    )


def _choice_xml_lines(choice: Mapping[str, Any], *, indent: str) -> list[str]:
    source_id = _xml(_text(choice.get("source_binding_id")))
    read_id = _xml(_text(choice.get("read_id")))
    plan_shape = _xml(_text(choice.get("plan_shape")))
    lines = [
        f'{indent}<source_binding id="{source_id}" read="{read_id}">',
        f'{indent}  <operation family="{plan_shape}">',
    ]
    group = _dict(choice.get("group"))
    lines.append(
        f'{indent}    <group fields="{_xml(" ".join(_texts(group.get("field_ids"))))}" key_id="{_xml(_text(group.get("key_id")))}" entity_kind="{_xml(_text(group.get("entity_kind")))}" source="source_binding" />'
    )
    lines.append(f"{indent}    <metric_candidates>")
    for item in choice.get("metric_candidates") or ():
        if _text(item.get("kind")) == "count_records":
            lines.append(
                f'{indent}      <metric id="{_xml(_text(item.get("id")))}" kind="count_records" count_basis="{_xml(count_basis_meaning(_dict(item.get("count_basis"))))}" allowed_functions="count" />'
            )
            continue
        lines.append(
            f'{indent}      <metric id="{_xml(_text(item.get("id")))}" kind="aggregate_field" field="{_xml(_text(item.get("field_id")))}" type="{_xml(_text(item.get("type")))}" allowed_functions="{_xml(" ".join(item.get("allowed_functions") or ()))}"{_metric_basis_suffix(item, indent=indent)}'
        )
    lines.extend(
        (f"{indent}    </metric_candidates>", f"{indent}    <function_candidates>")
    )
    for item in choice.get("function_candidates") or ():
        lines.append(
            f'{indent}      <function id="{_xml(_text(item.get("id")))}" value="{_xml(_text(item.get("value")))}" meaning="{_xml(_text(item.get("meaning")))}" />'
        )
    lines.append(f"{indent}    </function_candidates>")
    if choice.get("rank_candidates"):
        lines.append(f"{indent}    <rank_candidates>")
        for item in choice.get("rank_candidates") or ():
            lines.append(
                f'{indent}      <rank id="{_xml(_text(item.get("id")))}" sort="{_xml(_text(item.get("sort")))}" limit="{_xml(str(item.get("limit") or ""))}" meaning="{_xml(_text(item.get("meaning")))}" />'
            )
        lines.append(f"{indent}    </rank_candidates>")
    lines.extend((f"{indent}  </operation>", f"{indent}</source_binding>"))
    return lines


def _selected_candidate(
    selected_id: str,
    raw_candidates: Any,
    *,
    label: str,
) -> dict[str, Any]:
    for candidate in raw_candidates or ():
        if _text(candidate.get("id")) == selected_id:
            return candidate
    raise ValueError(f"unknown {label} selection")


def _validate_metric_selection(
    answer: GroupedAggregateAnswerOutput,
    candidate: Mapping[str, Any],
) -> None:
    if answer.metric.kind != _text(candidate.get("kind")):
        raise ValueError("metric selection mismatches candidate")
    if _text(candidate.get("kind")) == "aggregate_field" and _text(
        answer.metric.field_id
    ) != _text(candidate.get("field_id")):
        raise ValueError("metric selection mismatches candidate")


def _validate_function_selection(
    answer: GroupedAggregateAnswerOutput,
    candidate: Mapping[str, Any],
) -> None:
    if answer.function.value != _text(candidate.get("value")):
        raise ValueError("function selection mismatches candidate")


def _dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return dict(value)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _texts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected list")
    values = tuple(_text(item) for item in value)
    if not values or any(not item for item in values):
        raise ValueError("expected non-empty strings")
    return values


def _component_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected component list")
    return tuple(
        (_text(item.get("component_id")), _text(item.get("field_id")))
        for item in value
        if isinstance(item, dict)
    )


def _parse_group_candidate(payload: dict[str, Any]) -> _GroupCandidate:
    return _GroupCandidate(
        field_ids=_texts(payload.get("field_ids")),
        types=_texts(payload.get("types")),
        evidence_id=_text(payload.get("evidence_id")),
        field_evidence_ids=_texts(payload.get("field_evidence_ids")),
        key_id=_text(payload.get("key_id")),
        entity_kind=_text(payload.get("entity_kind")),
        components=_component_pairs(payload.get("components")),
    )


def _unique_group_candidates(
    candidates: list[_GroupCandidate],
) -> tuple[_GroupCandidate, ...]:
    output: list[_GroupCandidate] = []
    seen: set[
        tuple[
            tuple[str, ...],
            str,
            str,
            tuple[tuple[str, str], ...],
        ]
    ] = set()
    for candidate in candidates:
        signature = (
            candidate.field_ids,
            candidate.key_id,
            candidate.entity_kind,
            candidate.components,
        )
        if signature in seen:
            continue
        seen.add(signature)
        output.append(candidate)
    return tuple(output)


def _xml(value: str) -> str:
    return xml_attr(value)


def _metric_basis_suffix(item: Mapping[str, Any], *, indent: str) -> str:
    basis = _dict_or_empty(item.get("source_binding_basis"))
    if not basis:
        return " />"
    return (
        ">\n"
        f"{indent}        <source_binding_basis>\n"
        f"{indent}          <metric_meaning>{xml_text(_text(basis.get('metric_meaning')))}</metric_meaning>\n"
        f"{indent}          <fit_basis>{xml_text(_text(basis.get('fit_basis')))}</fit_basis>\n"
        f"{indent}        </source_binding_basis>\n"
        f"{indent}      </metric>"
    )
