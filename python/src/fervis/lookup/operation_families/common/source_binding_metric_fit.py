"""Metric-fit candidate ledger for source binding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, cast

from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.source_binding.candidates import (
    SourceCandidate,
    source_candidate_registry,
)
from fervis.lookup.source_binding.candidates.contracts import (
    EvidenceItem,
    FieldEvidence,
    RowPopulationEvidence,
)
from fervis.lookup.source_binding.evidence_types import (
    evidence_item_can_measure,
)
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.relation_catalog.model import CatalogField, EndpointRead

MetricCandidatePayload: TypeAlias = dict[str, str | list[str]]


def source_binding_metric_fit_surface_payload(
    request: SourceBindingRequest,
) -> dict[str, object]:
    """Return model-visible metric candidates grouped by requested fact."""

    surfaces: list[dict[str, object]] = []
    metric_contexts: dict[str, dict[str, object]] = {}
    candidates = _metric_candidates_by_requested_fact(
        request,
        metric_contexts=metric_contexts,
    )
    for fact in request.requested_facts:
        metric_candidates = candidates.get(fact.id, ())
        if not metric_candidates:
            continue
        surfaces.append(
            {
                "requested_fact_id": fact.id,
                "requested_measure_basis": {
                    "basis_id": f"measure_basis.{fact.id}",
                    "requested_fact_text": fact.description,
                    "answer_outputs": [
                        {
                            "answer_output_id": answer_output.id,
                            "answer_output_text": answer_output.description,
                        }
                        for answer_output in fact.answer_outputs
                    ],
                },
                "metric_candidates": list(metric_candidates),
            }
        )
    output: dict[str, object] = {"requested_fact_metric_fit_surface": surfaces}
    if metric_contexts:
        output["metric_contexts"] = list(metric_contexts.values())
    return output


def source_binding_metric_evidence_ids_by_requested_fact(
    request: SourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    """Return metric evidence ids the model must review per requested fact."""

    output: dict[str, tuple[str, ...]] = {}
    for requested_fact_id, metric_candidates in _metric_candidates_by_requested_fact(
        request
    ).items():
        output[requested_fact_id] = tuple(
            dict.fromkeys(
                str(candidate.get("metric_evidence_id") or "")
                for candidate in metric_candidates
                if str(candidate.get("metric_evidence_id") or "")
            )
        )
    return output


def _metric_candidates_by_requested_fact(
    request: SourceBindingRequest,
    *,
    metric_contexts: dict[str, dict[str, object]] | None = None,
) -> dict[str, tuple[MetricCandidatePayload, ...]]:
    registry = source_candidate_registry(request)
    output: dict[str, list[MetricCandidatePayload]] = {}
    seen_by_requested_fact: dict[str, set[str]] = {}
    relevant_field_refs = _relevant_field_refs_by_fact_candidate(request)
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    for candidate_id in registry.prompt_candidate_ids:
        candidate = registry.candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        requested_fact_ids = _requested_fact_ids_for_candidate(candidate, request)
        if not requested_fact_ids:
            continue
        read = _candidate_read(candidate, request=request)
        for requested_fact_id in requested_fact_ids:
            fact = facts_by_id.get(requested_fact_id)
            if fact is None:
                continue
            plan_shape = _plan_shape_for_candidate(
                request,
                requested_fact_id=requested_fact_id,
                source_candidate_id=candidate.id,
            )
            evidence_policy = _metric_evidence_policy(
                fact,
                plan_shape=plan_shape,
            )
            scoped_field_refs = relevant_field_refs.get(
                (requested_fact_id, candidate.id)
            )
            for item in _metric_evidence_items(
                candidate.evidence_items,
                evidence_policy=evidence_policy,
                scoped_field_refs=scoped_field_refs,
            ):
                if not _metric_evidence_is_relevant(
                    item,
                    requested_fact_id=requested_fact_id,
                    source_candidate_id=candidate.id,
                    relevant_field_refs=relevant_field_refs,
                ):
                    continue
                evidence_id = item.evidence_id
                if not evidence_id:
                    continue
                field_path = _field_path_for_evidence_item(item)
                row_path_id = _row_path_id_for_field_path(field_path)
                context_id = _metric_context_id(
                    source_candidate_id=candidate.id,
                    row_path_id=row_path_id,
                )
                if metric_contexts is not None:
                    metric_contexts[context_id] = _metric_context_payload(
                        candidate,
                        request=request,
                        read=read,
                        context_id=context_id,
                        row_path_id=row_path_id,
                        existing=metric_contexts.get(context_id),
                    )
                seen = seen_by_requested_fact.setdefault(
                    requested_fact_id,
                    set(),
                )
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                output.setdefault(requested_fact_id, []).append(
                    _metric_candidate_payload(
                        item,
                        source_candidate_id=candidate.id,
                        read=read,
                        field_path=field_path,
                        metric_context_id=context_id,
                    )
                )
    return {
        requested_fact_id: tuple(metric_candidates)
        for requested_fact_id, metric_candidates in output.items()
    }


def _relevant_field_refs_by_fact_candidate(
    request: SourceBindingRequest,
) -> dict[tuple[str, str], frozenset[str]]:
    if request.read_eligibility is None:
        return {}
    output: dict[tuple[str, str], set[str]] = {}
    for assessment in request.read_eligibility.retained_reads:
        if not assessment.is_retained:
            continue
        key = (assessment.requested_fact_id, assessment.source_candidate_id)
        output.setdefault(key, set()).update(assessment.relevant_field_refs)
    return {key: frozenset(value) for key, value in output.items()}


def _metric_evidence_is_relevant(
    item: EvidenceItem,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
    relevant_field_refs: dict[tuple[str, str], frozenset[str]],
) -> bool:
    field_ref = item.field_ref if isinstance(item, FieldEvidence) else ""
    evidence_has_no_scope_ref = not field_ref
    if evidence_has_no_scope_ref:
        return True
    allowed_refs = relevant_field_refs.get((requested_fact_id, source_candidate_id))
    if allowed_refs is None:
        return True
    evidence_ref_is_in_scope = field_ref in allowed_refs
    return evidence_ref_is_in_scope


def _requested_fact_ids_for_candidate(
    candidate: SourceCandidate,
    request: SourceBindingRequest,
) -> tuple[str, ...]:
    requested_fact_ids = {fact.id for fact in request.requested_facts}
    explicit_fact_ids = tuple(
        dict.fromkeys(
            fact_id
            for fact_id in candidate.applies_to_requested_fact_ids
            if fact_id in requested_fact_ids
        )
    )
    if explicit_fact_ids:
        return explicit_fact_ids
    if len(request.requested_facts) == 1:
        return (request.requested_facts[0].id,)
    return ()


def _metric_candidate_payload(
    item: EvidenceItem,
    *,
    source_candidate_id: str,
    read: EndpointRead | None,
    field_path: str,
    metric_context_id: str,
) -> MetricCandidatePayload:
    output: MetricCandidatePayload = {
        "metric_evidence_id": item.evidence_id,
        "source_candidate_id": source_candidate_id,
        "read_id": read.id if read is not None else "",
        "field_path": field_path,
        "field_type": item.type,
        "metric_context_id": metric_context_id,
    }
    if isinstance(item, RowPopulationEvidence):
        output["metric_operation"] = "count_rows"
    resource_names = read.resource_names if read is not None else ()
    if resource_names:
        output["resource_names"] = list(resource_names)
    return {key: value for key, value in output.items() if value not in ("", [], ())}


def _metric_context_payload(
    candidate: SourceCandidate,
    *,
    request: SourceBindingRequest,
    read: EndpointRead | None,
    context_id: str,
    row_path_id: str,
    existing: dict[str, object] | None,
) -> dict[str, object]:
    same_row_field_paths = _same_row_field_paths(
        candidate,
        request=request,
        read=read,
        row_path_id=row_path_id,
    )
    scope_field_paths: tuple[str, ...] = ()
    if existing:
        same_row_field_paths = _unique(
            *cast(tuple[str, ...], existing.get("same_row_field_paths") or ()),
            *same_row_field_paths,
        )
        scope_field_paths = _unique(
            *cast(tuple[str, ...], existing.get("scope_field_paths") or ()),
            *scope_field_paths,
        )
    output = {
        "metric_context_id": context_id,
        "source_candidate_id": candidate.id,
        "read_id": read.id if read is not None else "",
        "row_path_id": row_path_id,
        "same_row_field_paths": list(same_row_field_paths),
        "scope_field_paths": list(scope_field_paths),
    }
    return {key: value for key, value in output.items() if value not in ("", [], ())}


def _same_row_field_paths(
    candidate: SourceCandidate,
    *,
    request: SourceBindingRequest,
    read: EndpointRead | None,
    row_path_id: str,
) -> tuple[str, ...]:
    catalog_field_paths = _same_row_catalog_field_paths(
        request=request,
        read_id=read.id if read is not None else "",
        row_path_id=row_path_id,
    )
    if catalog_field_paths:
        return catalog_field_paths
    return _unique(
        *(
            field_path
            for field in candidate.evidence_items
            if isinstance(field, FieldEvidence)
            for field_path in (_field_path_for_candidate_field(field),)
            if field_path and _row_path_id_for_field_path(field_path) == row_path_id
        )
    )


def _same_row_catalog_field_paths(
    *,
    request: SourceBindingRequest,
    read_id: str,
    row_path_id: str,
) -> tuple[str, ...]:
    if not read_id:
        return ()
    try:
        read = request.relation_catalog.read(read_id)
    except KeyError:
        return ()
    return _unique(
        *(
            field_path
            for field in read.fields
            for field_path in (str(getattr(field, "path", "") or ""),)
            if field_path
            and _catalog_field_row_path_id(field, field_path) == row_path_id
        )
    )


def _catalog_field_row_path_id(field: CatalogField, field_path: str) -> str:
    row_path_id = field.row_path_id
    if row_path_id:
        return row_path_id
    return _row_path_id_for_field_path(field_path)


def _field_path_for_candidate_field(field: FieldEvidence) -> str:
    explicit_path = field.response_path or field.path
    if explicit_path:
        return explicit_path
    field_id = field.field_id
    if not field_id:
        return ""
    if "." in field_id:
        return field_id
    row_path_id = field.row_path_id
    if row_path_id and row_path_id != "root":
        return f"{row_path_id}.{field_id}"
    return field_id


def _field_path_for_evidence_item(
    item: EvidenceItem,
) -> str:
    if isinstance(item, FieldEvidence):
        explicit_path = item.response_path or item.path
        field_id = item.field_id
        row_path_id = item.row_path_id
    elif isinstance(item, RowPopulationEvidence):
        explicit_path = ""
        field_id = item.row_path_id
        row_path_id = item.row_path_id
    else:
        return ""
    if explicit_path:
        return explicit_path
    if "." in field_id:
        return field_id
    if row_path_id and row_path_id != "root" and field_id:
        return f"{row_path_id}.{field_id}"
    return field_id


def _row_path_id_for_field_path(field_path: str) -> str:
    if "." not in field_path:
        return "root"
    return field_path.rsplit(".", 1)[0]


def _metric_context_id(
    *,
    source_candidate_id: str,
    row_path_id: str,
) -> str:
    return f"metric_context.{source_candidate_id}.{row_path_id}"


def _unique(*items: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _candidate_read(
    candidate: SourceCandidate,
    *,
    request: SourceBindingRequest,
) -> EndpointRead | None:
    source = candidate.source
    if source is None or not source.read_id:
        return None
    try:
        return request.relation_catalog.read(source.read_id)
    except KeyError:
        return None


@dataclass(frozen=True)
class _MetricEvidencePolicy:
    include_measured_fields: bool
    include_scoped_measured_fields: bool
    include_row_population: bool


class _MetricEvidenceKind(Enum):
    MEASURED_FIELD = "measured_field"
    ROW_COUNT = "row_population"
    UNSUPPORTED = "unsupported"


def _metric_evidence_policy(
    fact: RequestedFact,
    *,
    plan_shape: str,
) -> _MetricEvidencePolicy:
    roles = _answer_output_roles(fact)
    supports_row_count_metric = _plan_shape_supports_row_count_metric(plan_shape)
    row_count_is_answer = "ROW_COUNT" in roles
    row_count_is_intermediate = plan_shape in {
        "aggregate_by_group",
    }
    return _MetricEvidencePolicy(
        include_measured_fields=bool(roles & {"ANSWER_VALUE", "MEASURED_VALUE"}),
        include_scoped_measured_fields=row_count_is_answer,
        include_row_population=(
            supports_row_count_metric
            and (row_count_is_answer or row_count_is_intermediate)
        ),
    )


def _answer_output_roles(fact: RequestedFact) -> frozenset[str]:
    return frozenset(output.role for output in fact.support_answer_outputs)


def _metric_evidence_items(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    evidence_policy: _MetricEvidencePolicy,
    scoped_field_refs: frozenset[str] | None,
) -> tuple[EvidenceItem, ...]:
    return tuple(
        item
        for item in evidence_items
        if _metric_evidence_policy_allows(
            evidence_policy,
            item,
            scoped_field_refs=scoped_field_refs,
        )
    )


def _metric_evidence_policy_allows(
    policy: _MetricEvidencePolicy,
    item: EvidenceItem,
    *,
    scoped_field_refs: frozenset[str] | None,
) -> bool:
    evidence_kind = _metric_evidence_kind(item)
    if evidence_kind == _MetricEvidenceKind.MEASURED_FIELD:
        measured_fields_are_allowed = policy.include_measured_fields
        scoped_measured_field_is_allowed = (
            policy.include_scoped_measured_fields
            and _evidence_item_has_scoped_field_ref(
                item,
                scoped_field_refs=scoped_field_refs,
            )
        )
        return measured_fields_are_allowed or scoped_measured_field_is_allowed
    if evidence_kind == _MetricEvidenceKind.ROW_COUNT:
        row_population_is_allowed = policy.include_row_population
        row_population_is_executable = _is_executable_row_population(item)
        return row_population_is_allowed and row_population_is_executable
    return False


def _metric_evidence_kind(item: EvidenceItem) -> _MetricEvidenceKind:
    if isinstance(item, RowPopulationEvidence):
        return _MetricEvidenceKind.ROW_COUNT
    if isinstance(item, FieldEvidence) and evidence_item_can_measure(item):
        return _MetricEvidenceKind.MEASURED_FIELD
    return _MetricEvidenceKind.UNSUPPORTED


def _evidence_item_has_scoped_field_ref(
    item: EvidenceItem,
    *,
    scoped_field_refs: frozenset[str] | None,
) -> bool:
    field_ref = item.field_ref if isinstance(item, FieldEvidence) else ""
    if scoped_field_refs is None:
        return False
    return field_ref in scoped_field_refs


def _is_executable_row_population(item: EvidenceItem) -> bool:
    return isinstance(item, RowPopulationEvidence) and item.row_cardinality == "many"


def _plan_shape_supports_row_count_metric(plan_shape: str) -> bool:
    return plan_shape in {
        "aggregate_scalar",
        "aggregate_by_group",
    }


def _plan_shape_for_candidate(
    request: SourceBindingRequest,
    *,
    requested_fact_id: str,
    source_candidate_id: str,
) -> str:
    for selection in request.plan_selection.plan_selections:
        if selection.requested_fact_id != requested_fact_id:
            continue
        for member in selection.source_members:
            if member.source_candidate_id == source_candidate_id:
                return selection.plan_shape
    return ""
