"""Metric-fit candidate ledger for source binding."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.candidates import source_candidate_registry
from fervis.lookup.source_binding.evidence_types import (
    evidence_item_can_measure,
)
from fervis.lookup.source_binding.model import SourceBindingRequest


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
) -> dict[str, tuple[dict[str, object], ...]]:
    registry = source_candidate_registry(request)
    output: dict[str, list[dict[str, object]]] = {}
    seen_by_requested_fact: dict[str, set[str]] = {}
    relevant_field_refs = _relevant_field_refs_by_fact_candidate(request)
    for candidate_id in registry.prompt_candidate_ids:
        candidate = registry.candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        requested_fact_ids = _requested_fact_ids_for_candidate(candidate, request)
        if not requested_fact_ids:
            continue
        payload = candidate.payload or {}
        read_contract = _read_contract(payload)
        for requested_fact_id in requested_fact_ids:
            include_row_count = _plan_shape_for_candidate(
                request,
                requested_fact_id=requested_fact_id,
                source_candidate_id=candidate.id,
            ) in {
                "aggregate_scalar",
                "aggregate_by_group",
                "ranked_aggregate",
            }
            for item in _metric_evidence_items(
                payload,
                include_row_count=include_row_count,
            ):
                if not _metric_evidence_is_relevant(
                    item,
                    requested_fact_id=requested_fact_id,
                    source_candidate_id=candidate.id,
                    relevant_field_refs=relevant_field_refs,
                ):
                    continue
                evidence_id = str(item.get("evidence_id") or "")
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
                        read_contract=read_contract,
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
                        read_contract=read_contract,
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
    for assessment in request.read_eligibility.read_assessments:
        if not assessment.is_retained:
            continue
        key = (assessment.requested_fact_id, assessment.source_candidate_id)
        output.setdefault(key, set()).update(assessment.relevant_field_refs)
    return {key: frozenset(value) for key, value in output.items()}


def _metric_evidence_is_relevant(
    item: dict[str, Any],
    *,
    requested_fact_id: str,
    source_candidate_id: str,
    relevant_field_refs: dict[tuple[str, str], frozenset[str]],
) -> bool:
    field_ref = str(item.get("field_ref") or "")
    if not field_ref:
        return True
    allowed_refs = relevant_field_refs.get((requested_fact_id, source_candidate_id))
    if allowed_refs is None:
        return True
    return field_ref in allowed_refs


def _requested_fact_ids_for_candidate(
    candidate: Any,
    request: SourceBindingRequest,
) -> tuple[str, ...]:
    if candidate.requested_fact_id:
        return (candidate.requested_fact_id,)
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
    item: dict[str, Any],
    *,
    source_candidate_id: str,
    read_contract: dict[str, Any],
    field_path: str,
    metric_context_id: str,
) -> dict[str, object]:
    output = {
        "metric_evidence_id": str(item.get("evidence_id") or ""),
        "source_candidate_id": source_candidate_id,
        "read_id": str(read_contract.get("read_id") or ""),
        "field_path": field_path,
        "field_type": str(item.get("type") or ""),
        "metric_context_id": metric_context_id,
    }
    resource_names = read_contract.get("resource_names")
    if resource_names:
        output["resource_names"] = list(resource_names)
    return {key: value for key, value in output.items() if value not in ("", [], ())}


def _metric_context_payload(
    candidate: Any,
    *,
    request: SourceBindingRequest,
    read_contract: dict[str, Any],
    context_id: str,
    row_path_id: str,
    existing: dict[str, object] | None,
) -> dict[str, object]:
    same_row_field_paths = _same_row_field_paths(
        candidate,
        request=request,
        read_contract=read_contract,
        row_path_id=row_path_id,
    )
    scope_field_paths: tuple[str, ...] = ()
    if existing:
        same_row_field_paths = _unique(
            *tuple(existing.get("same_row_field_paths") or ()),
            *same_row_field_paths,
        )
        scope_field_paths = _unique(
            *tuple(existing.get("scope_field_paths") or ()),
            *scope_field_paths,
        )
    output = {
        "metric_context_id": context_id,
        "source_candidate_id": candidate.id,
        "read_id": str(read_contract.get("read_id") or ""),
        "row_path_id": row_path_id,
        "same_row_field_paths": list(same_row_field_paths),
        "scope_field_paths": list(scope_field_paths),
    }
    return {key: value for key, value in output.items() if value not in ("", [], ())}


def _same_row_field_paths(
    candidate: Any,
    *,
    request: SourceBindingRequest,
    read_contract: dict[str, Any],
    row_path_id: str,
) -> tuple[str, ...]:
    catalog_field_paths = _same_row_catalog_field_paths(
        request=request,
        read_id=str(read_contract.get("read_id") or ""),
        row_path_id=row_path_id,
    )
    if catalog_field_paths:
        return catalog_field_paths
    return _unique(
        *(
            field_path
            for field in candidate.fields
            if isinstance(field, dict)
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


def _catalog_field_row_path_id(field: Any, field_path: str) -> str:
    row_path_id = str(getattr(field, "row_path_id", "") or "")
    if row_path_id:
        return row_path_id
    return _row_path_id_for_field_path(field_path)


def _field_path_for_candidate_field(field: dict[str, Any]) -> str:
    explicit_path = str(
        field.get("field_path") or field.get("response_path") or field.get("path") or ""
    )
    if explicit_path:
        return explicit_path
    field_id = str(field.get("field_id") or field.get("id") or "")
    if not field_id:
        return ""
    if "." in field_id:
        return field_id
    row_path_id = str(field.get("row_path_id") or "")
    if row_path_id and row_path_id != "root":
        return f"{row_path_id}.{field_id}"
    return field_id


def _field_path_for_evidence_item(
    item: dict[str, Any],
) -> str:
    explicit_path = str(
        item.get("field_path") or item.get("response_path") or item.get("path") or ""
    )
    if explicit_path:
        return explicit_path
    field_id = str(item.get("field_id") or item.get("id") or "")
    if "." in field_id:
        return field_id
    row_path_id = str(item.get("row_path_id") or "")
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


def _read_contract(payload: dict[str, Any]) -> dict[str, Any]:
    read_contract = payload.get("read_contract")
    if isinstance(read_contract, dict):
        return read_contract
    return {
        key: payload[key]
        for key in ("read_id", "description", "resource_names")
        if key in payload and payload[key] not in (None, "", [], ())
    }


def _metric_evidence_source(payload: dict[str, Any]) -> dict[str, Any]:
    binding_surface = payload.get("binding_surface")
    if isinstance(binding_surface, dict):
        return binding_surface
    return payload


def _metric_evidence_items(
    payload: dict[str, Any],
    *,
    include_row_count: bool,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in _metric_evidence_source(payload).get("evidence_items") or ()
        if isinstance(item, dict)
        and (
            evidence_item_can_measure(item)
            or (include_row_count and str(item.get("type") or "") == "row_population")
        )
    )


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
